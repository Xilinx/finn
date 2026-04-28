# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import pytest

import numpy as np
import os
from functools import partial
from onnx import TensorProto, helper, numpy_helper
from pathlib import Path
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames

from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.analysis.fpgadataflow.res_estimation import (
    res_estimation,
    res_estimation_complete,
)
from finn.core.onnx_exec import execute_onnx
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferAddCLSTokenLayer
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.synth_ooc import SynthOutOfContext

FPGA_PART = "xc7z020clg400-1"
CLK_NS = 10


def _make_graph(nodes, output_shape, cls_values, finn_dtype=DataType["INT8"]):
    patch_shape = [1, 3, 4]
    patches = helper.make_tensor_value_info("patches", TensorProto.FLOAT, patch_shape)
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, output_shape)
    cls_init = numpy_helper.from_array(cls_values.astype(np.float32), name="cls")
    graph = helper.make_graph(nodes, "addclstoken_test", [patches], [output], [cls_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model = ModelWrapper(model)
    for tensor_name in ["patches", "cls", "out"]:
        model.set_tensor_datatype(tensor_name, finn_dtype)
    return model


def _make_concat_model():
    cls_values = np.asarray([[[1, -2, 3, -4]]], dtype=np.float32)
    concat = helper.make_node(
        "Concat",
        ["cls", "patches"],
        ["out"],
        axis=1,
        name="concat_cls",
    )
    model = _make_graph([concat], [1, 4, 4], cls_values)
    return model, cls_values


def _make_addclstoken_model(
    pad_tokens=0,
    simd=1,
    finn_dtype=DataType["INT8"],
    cls_values=None,
):
    if cls_values is None:
        cls_values = np.asarray([[[1, -2, 3, -4]]], dtype=np.float32)
    addcls = helper.make_node(
        "AddCLSToken",
        ["patches", "cls"],
        ["out"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        name="AddCLSToken_0",
        NumTokens=3,
        NumChannels=4,
        PadTokens=pad_tokens,
        SIMD=simd,
        inputDataType=finn_dtype.name,
        outputDataType=finn_dtype.name,
    )
    model = _make_graph([addcls], [1, 4 + pad_tokens, 4], cls_values, finn_dtype)
    return model, cls_values


def _prepare_addclstoken_stitched_ip_model(simd=1, pad_tokens=0):
    model, cls_values = _make_addclstoken_model(pad_tokens=pad_tokens, simd=simd)
    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(InsertFIFO(create_shallow_fifos=True))
    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(FPGA_PART, CLK_NS))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(FPGA_PART, CLK_NS, vitis=False))
    return model, cls_values


@pytest.mark.fpgadataflow
def test_convert_concat_to_addclstoken():
    model, cls_values = _make_concat_model()
    patches = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    expected = np.concatenate([cls_values, patches], axis=1)

    ret = execute_onnx(model, {"patches": patches})
    assert (ret["out"] == expected).all()

    model = model.transform(InferAddCLSTokenLayer())
    node = model.graph.node[0]
    assert node.op_type == "AddCLSToken"
    assert node.domain == "finn.custom_op.fpgadataflow"
    assert list(node.input) == ["patches", "cls"]

    inst = getCustomOp(node)
    assert inst.get_normal_output_shape() == (1, 4, 4)
    assert inst.get_exp_cycles() == 16

    ret = execute_onnx(model, {"patches": patches})
    assert (ret["out"] == expected).all()

    model = model.transform(SpecializeLayers("xc7z020clg400-1"))
    assert model.graph.node[0].op_type == "AddCLSToken_rtl"
    assert model.graph.node[0].domain == "finn.custom_op.fpgadataflow.rtl"
    assert model.graph.node[0].name == "AddCLSToken_concat_cls"


@pytest.mark.fpgadataflow
def test_addclstoken_python_execution_with_padding():
    model, cls_values = _make_addclstoken_model(pad_tokens=2)
    patches = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    expected = np.concatenate(
        [cls_values, patches, np.zeros((1, 2, 4), dtype=np.float32)],
        axis=1,
    )

    ret = execute_onnx(model, {"patches": patches})
    assert (ret["out"] == expected).all()


@pytest.mark.fpgadataflow
@pytest.mark.parametrize(
    "finn_dtype,cls_values,expected_cls_data",
    [
        (DataType["INT8"], np.asarray([[[1, -2, 3, -4]]], dtype=np.float32), "32'hfc03fe01"),
        (DataType["UINT4"], np.asarray([[[1, 2, 3, 4]]], dtype=np.float32), "16'h4321"),
        (DataType["BIPOLAR"], np.asarray([[[1, -1, 1, -1]]], dtype=np.float32), "4'h5"),
    ],
)
def test_addclstoken_rtl_codegen(tmp_path, monkeypatch, finn_dtype, cls_values, expected_cls_data):
    if "FINN_ROOT" not in os.environ:
        monkeypatch.setenv("FINN_ROOT", str(Path(__file__).resolve().parents[2]))

    model, _ = _make_addclstoken_model(
        pad_tokens=1,
        simd=2,
        finn_dtype=finn_dtype,
        cls_values=cls_values,
    )
    model = model.transform(SpecializeLayers("xc7z020clg400-1"))

    node = model.graph.node[0]
    inst = getCustomOp(node)
    inst.set_nodeattr("code_gen_dir_ipgen", str(tmp_path))
    inst.code_generation_ipgen(model, "xc7z020clg400-1", 10)

    topname = inst.get_nodeattr("gen_top_module")
    assert topname == "AddCLSToken_0"
    wrapper = tmp_path / (topname + ".v")
    core = tmp_path / "addclstoken.sv"
    assert wrapper.is_file()
    assert core.is_file()
    wrapper_text = wrapper.read_text()
    assert "parameter FOLD_WIDTH = %d" % (2 * finn_dtype.bitwidth()) in wrapper_text
    assert ".SIMD(2)" in wrapper_text
    assert ".PAD_TOKENS(1)" in wrapper_text
    assert "CLS_DATA = %s" % expected_cls_data in wrapper_text
    assert "= '0" not in wrapper_text

    ipi_cmds = inst.code_generation_ipi()
    assert any("addclstoken.sv" in cmd for cmd in ipi_cmds)
    assert any("create_bd_cell" in cmd and topname in cmd for cmd in ipi_cmds)


@pytest.mark.fpgadataflow
def test_addclstoken_resource_estimation():
    model, _ = _make_addclstoken_model(pad_tokens=1, simd=2)
    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())

    expected = {
        "BRAM_18K": 0,
        "BRAM_efficiency": 1,
        "LUT": 132,
        "URAM": 0,
        "URAM_efficiency": 1,
        "DSP": 0,
    }
    resources = model.analysis(partial(res_estimation, fpgapart=FPGA_PART))
    assert len(resources) == 1
    assert list(resources.values())[0] == expected

    complete_resources = model.analysis(partial(res_estimation_complete, fpgapart=FPGA_PART))
    assert len(complete_resources) == 1
    assert list(complete_resources.values())[0] == [expected]


@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
@pytest.mark.parametrize("simd,pad_tokens", [(1, 0), (2, 1)])
def test_addclstoken_rtlsim(simd, pad_tokens):
    model, cls_values = _make_addclstoken_model(pad_tokens=pad_tokens, simd=simd)
    patches = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    expected_values = [cls_values, patches]
    if pad_tokens > 0:
        expected_values.append(np.zeros((1, pad_tokens, 4), dtype=np.float32))
    expected = np.concatenate(expected_values, axis=1)

    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(FPGA_PART, CLK_NS))
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareRTLSim())

    ret = execute_onnx(model, {"patches": patches})
    assert (ret["out"] == expected).all()

    node = model.get_nodes_by_op_type("AddCLSToken_rtl")[0]
    inst = getCustomOp(node)
    cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
    exp_cycles_dict = model.analysis(exp_cycles_per_layer)
    exp_cycles = exp_cycles_dict[node.name]
    assert np.isclose(exp_cycles, cycles_rtlsim, atol=10)
    assert exp_cycles != 0


@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
@pytest.mark.parametrize("simd,pad_tokens", [(1, 0), (2, 1)])
def test_addclstoken_stitched_ip_rtlsim(simd, pad_tokens):
    model, cls_values = _prepare_addclstoken_stitched_ip_model(
        simd=simd,
        pad_tokens=pad_tokens,
    )
    patches = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    expected_values = [cls_values, patches]
    if pad_tokens > 0:
        expected_values.append(np.zeros((1, pad_tokens, 4), dtype=np.float32))
    expected = np.concatenate(expected_values, axis=1)

    model.set_metadata_prop("exec_mode", "rtlsim")
    model.set_metadata_prop("extra_verilator_args", str(["-Wno-TIMESCALEMOD"]))

    ret = execute_onnx(model, {"patches": patches})
    assert (ret["out"] == expected).all()


@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_addclstoken_stitched_ip_synth_ooc():
    model, _ = _prepare_addclstoken_stitched_ip_model(simd=2, pad_tokens=1)
    model = model.transform(SynthOutOfContext(FPGA_PART, CLK_NS))
    ret = model.get_metadata_prop("res_total_ooc_synth")
    assert ret is not None
    ret = eval(ret)

    assert ret["LUT"] > 0
    assert ret["FF"] > 0
    assert ret["DSP"] == 0
    assert ret["BRAM"] == 0
    assert ret["WNS"] >= 0
