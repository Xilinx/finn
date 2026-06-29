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
from functools import partial
from onnx import TensorProto, helper, numpy_helper
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
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferSelectTokenLayer
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.vivado import parse_ooc_synth_results

FPGA_PART = "xc7z020clg400-1"
CLK_NS = 10


def _make_graph(nodes, output_shape, idx_values=None, finn_dtype=DataType["INT8"]):
    tokens_shape = [1, 4, 4]
    tokens = helper.make_tensor_value_info("tokens", TensorProto.FLOAT, tokens_shape)
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, output_shape)
    initializers = []
    if idx_values is not None:
        initializers.append(numpy_helper.from_array(idx_values, name="idx"))
    graph = helper.make_graph(nodes, "selecttoken_test", [tokens], [output], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model = ModelWrapper(model)
    for tensor_name in ["tokens", "out"]:
        model.set_tensor_datatype(tensor_name, finn_dtype)
    return model


def _make_gather_model(token_index=0):
    idx_values = np.asarray(token_index, dtype=np.int64)
    gather = helper.make_node(
        "Gather",
        ["tokens", "idx"],
        ["out"],
        axis=1,
        name="gather_token",
    )
    return _make_graph([gather], [1, 4], idx_values)


def _make_selecttoken_model(token_index=0, simd=1, finn_dtype=DataType["INT8"]):
    select = helper.make_node(
        "SelectToken",
        ["tokens"],
        ["out"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        name="SelectToken_0",
        NumTokens=4,
        NumChannels=4,
        TokenIndex=token_index,
        SIMD=simd,
        inputDataType=finn_dtype.name,
        outputDataType=finn_dtype.name,
    )
    return _make_graph([select], [1, 4], None, finn_dtype)


def _prepare_selecttoken_stitched_ip_model(simd=1, token_index=0, run_pnr=False):
    model = _make_selecttoken_model(token_index=token_index, simd=simd)
    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(InsertFIFO(create_shallow_fifos=True))
    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(FPGA_PART, CLK_NS))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(FPGA_PART, CLK_NS, run_pnr=run_pnr))
    return model


def _make_input_dict(model, tokens):
    return {model.graph.input[0].name: tokens}


@pytest.mark.fpgadataflow
def test_convert_gather_to_selecttoken():
    model = _make_gather_model(token_index=2)
    tokens = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    expected = tokens[:, 2, :]

    ret = execute_onnx(model, _make_input_dict(model, tokens))
    assert (ret["out"] == expected).all()

    model = model.transform(InferSelectTokenLayer())
    node = model.graph.node[0]
    assert node.op_type == "SelectToken"
    assert node.domain == "finn.custom_op.fpgadataflow"
    assert list(node.input) == ["tokens"]

    inst = getCustomOp(node)
    assert inst.get_normal_output_shape() == (1, 4)
    assert inst.get_exp_cycles() == 16
    assert inst.get_nodeattr("TokenIndex") == 2

    ret = execute_onnx(model, _make_input_dict(model, tokens))
    assert (ret["out"] == expected).all()

    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())
    assert model.graph.node[0].op_type == "SelectToken_rtl"
    assert model.graph.node[0].domain == "finn.custom_op.fpgadataflow.rtl"


@pytest.mark.fpgadataflow
@pytest.mark.parametrize("token_index", [0, 1, 3])
def test_selecttoken_python_execution(token_index):
    model = _make_selecttoken_model(token_index=token_index)
    tokens = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    expected = tokens[:, token_index, :]

    ret = execute_onnx(model, _make_input_dict(model, tokens))
    assert (ret["out"] == expected).all()


@pytest.mark.fpgadataflow
@pytest.mark.parametrize(
    "finn_dtype,fold_width",
    [(DataType["INT8"], 16), (DataType["UINT4"], 8), (DataType["BIPOLAR"], 2)],
)
def test_selecttoken_rtl_codegen(tmp_path, finn_dtype, fold_width):
    model = _make_selecttoken_model(token_index=3, simd=2, finn_dtype=finn_dtype)
    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())

    node = model.graph.node[0]
    inst = getCustomOp(node)
    inst.set_nodeattr("code_gen_dir_ipgen", str(tmp_path))
    inst.code_generation_ipgen(model, FPGA_PART, CLK_NS)

    topname = inst.get_nodeattr("gen_top_module")
    assert topname == node.name
    wrapper = tmp_path / (topname + ".v")
    core = tmp_path / "select_token.sv"
    assert wrapper.is_file()
    assert core.is_file()
    wrapper_text = wrapper.read_text()
    assert "parameter FOLD_WIDTH = %d" % fold_width in wrapper_text
    assert ".TOKEN_BEATS(2)" in wrapper_text
    assert ".DATA_WIDTH(FOLD_WIDTH)" in wrapper_text
    assert ".TOKEN_INDEX(3)" in wrapper_text
    assert "select_token #(" in wrapper_text
    assert "out0_V_TVALID" in wrapper_text

    ipi_cmds = inst.code_generation_ipi()
    assert any("select_token.sv" in cmd for cmd in ipi_cmds)
    assert any("create_bd_cell" in cmd and topname in cmd for cmd in ipi_cmds)


@pytest.mark.fpgadataflow
def test_selecttoken_resource_estimation():
    model = _make_selecttoken_model(token_index=1, simd=2)
    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())

    expected = {
        "BRAM_18K": 0,
        "BRAM_efficiency": 1,
        "LUT": 200,
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
@pytest.mark.parametrize("simd,token_index", [(1, 0), (2, 3)])
def test_selecttoken_rtlsim(simd, token_index):
    model = _make_selecttoken_model(token_index=token_index, simd=simd)
    tokens = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    expected = tokens[:, token_index, :]

    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(FPGA_PART, CLK_NS))
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareRTLSim())

    ret = execute_onnx(model, _make_input_dict(model, tokens))
    assert (ret["out"] == expected).all()

    node = model.get_nodes_by_op_type("SelectToken_rtl")[0]
    inst = getCustomOp(node)
    cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
    exp_cycles_dict = model.analysis(exp_cycles_per_layer)
    exp_cycles = exp_cycles_dict[node.name]
    assert np.isclose(exp_cycles, cycles_rtlsim, atol=10)
    assert exp_cycles != 0


@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
@pytest.mark.parametrize("simd,token_index", [(1, 0), (2, 3)])
def test_selecttoken_stitched_ip_rtlsim(simd, token_index):
    model = _prepare_selecttoken_stitched_ip_model(simd=simd, token_index=token_index)
    tokens = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    expected = tokens[:, token_index, :]

    model.set_metadata_prop("exec_mode", "rtlsim")

    ret = execute_onnx(model, _make_input_dict(model, tokens))
    assert (ret["out"] == expected).all()


@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_selecttoken_stitched_ip_synth_ooc():
    model = _prepare_selecttoken_stitched_ip_model(simd=2, token_index=1, run_pnr=True)
    ret = parse_ooc_synth_results(model.get_metadata_prop("vivado_stitch_proj"))
    assert ret is not None

    assert ret["LUT"] > 0
    assert ret["FF"] > 0
    assert ret.get("DSP", 0) == 0
    assert ret.get("BRAM_18K", 0) == 0
    assert ret.get("BRAM_36K", 0) == 0
    assert ret["WNS"] >= 0
