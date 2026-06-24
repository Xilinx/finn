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
from onnx import AttributeProto, TensorProto, helper
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
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferWhereLayer
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


def _numel(shape):
    return int(np.prod(shape)) if len(shape) > 0 else 1


def _make_graph(
    nodes,
    shape=None,
    finn_dtype=DataType["INT8"],
    cond_is_bool=False,
    cond_shape=None,
    x_shape=None,
    y_shape=None,
    out_shape=None,
):
    if shape is None:
        shape = [1, 2, 4]
    cond_shape = shape if cond_shape is None else cond_shape
    x_shape = shape if x_shape is None else x_shape
    y_shape = shape if y_shape is None else y_shape
    out_shape = shape if out_shape is None else out_shape
    cond_proto = TensorProto.BOOL if cond_is_bool else TensorProto.FLOAT
    cond = helper.make_tensor_value_info("cond", cond_proto, cond_shape)
    xval = helper.make_tensor_value_info("xval", TensorProto.FLOAT, x_shape)
    yval = helper.make_tensor_value_info("yval", TensorProto.FLOAT, y_shape)
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, out_shape)
    graph = helper.make_graph(nodes, "where_test", [cond, xval, yval], [output])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model = ModelWrapper(model)
    if not cond_is_bool:
        model.set_tensor_datatype("cond", DataType["BINARY"])
    for tensor_name in ["xval", "yval", "out"]:
        model.set_tensor_datatype(tensor_name, finn_dtype)
    return model


def _make_onnx_where_model(
    shape=None,
    finn_dtype=DataType["INT8"],
    cond_shape=None,
    x_shape=None,
    y_shape=None,
    out_shape=None,
):
    if shape is None:
        shape = [1, 2, 4]
    where = helper.make_node("Where", ["cond", "xval", "yval"], ["out"], name="where_select")
    return _make_graph(
        [where],
        shape,
        finn_dtype=finn_dtype,
        cond_is_bool=True,
        cond_shape=cond_shape,
        x_shape=x_shape,
        y_shape=y_shape,
        out_shape=out_shape,
    )


def _make_where_model(
    shape=None,
    pe=1,
    finn_dtype=DataType["INT8"],
    cond_shape=None,
    x_shape=None,
    y_shape=None,
    out_shape=None,
):
    if shape is None:
        shape = [1, 2, 4]
    cond_shape = shape if cond_shape is None else cond_shape
    x_shape = shape if x_shape is None else x_shape
    y_shape = shape if y_shape is None else y_shape
    out_shape = shape if out_shape is None else out_shape
    where = helper.make_node(
        "Where",
        ["cond", "xval", "yval"],
        ["out"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        name="Where_0",
        CondRank=len(cond_shape),
        XRank=len(x_shape),
        YRank=len(y_shape),
        PE=pe,
        conditionDataType="BINARY",
        inputDataType=finn_dtype.name,
        outputDataType=finn_dtype.name,
        inFIFODepths=[2, 2, 2],
        outFIFODepths=[2],
    )
    for attr_name, attr_value in [
        ("Shape", out_shape),
        ("CondShape", cond_shape),
        ("XShape", x_shape),
        ("YShape", y_shape),
    ]:
        where.attribute.append(
            helper.make_attribute(attr_name, attr_value, attr_type=AttributeProto.INTS)
        )
    return _make_graph(
        [where],
        shape,
        finn_dtype=finn_dtype,
        cond_shape=cond_shape,
        x_shape=x_shape,
        y_shape=y_shape,
        out_shape=out_shape,
    )


def _prepare_where_stitched_ip_model(
    pe=1,
    shape=None,
    cond_shape=None,
    x_shape=None,
    y_shape=None,
    out_shape=None,
):
    model = _make_where_model(
        pe=pe,
        shape=shape,
        cond_shape=cond_shape,
        x_shape=x_shape,
        y_shape=y_shape,
        out_shape=out_shape,
    )
    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(InsertFIFO(create_shallow_fifos=True))
    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(FPGA_PART, CLK_NS))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(FPGA_PART, CLK_NS, vitis=False))
    return model


def _make_inputs(shape=None, cond_shape=None, x_shape=None, y_shape=None):
    if shape is None:
        shape = [1, 2, 4]
    cond_shape = shape if cond_shape is None else cond_shape
    x_shape = shape if x_shape is None else x_shape
    y_shape = shape if y_shape is None else y_shape
    cond = (np.arange(_numel(cond_shape), dtype=np.float32) % 2).reshape(cond_shape)
    xval = np.arange(_numel(x_shape), dtype=np.float32).reshape(x_shape)
    yval = (100 + np.arange(_numel(y_shape), dtype=np.float32)).reshape(y_shape)
    return cond, xval, yval


@pytest.mark.fpgadataflow
def test_convert_onnx_where_to_where():
    model = _make_onnx_where_model()
    cond, xval, yval = _make_inputs()
    expected = np.where(cond.astype(bool), xval, yval)

    ret = execute_onnx(model, {"cond": cond.astype(bool), "xval": xval, "yval": yval})
    assert (ret["out"] == expected).all()

    model.set_tensor_datatype("cond", DataType["BINARY"])
    model = model.transform(InferWhereLayer())
    node = model.graph.node[0]
    assert node.op_type == "Where"
    assert node.domain == "finn.custom_op.fpgadataflow"
    assert list(node.input) == ["cond", "xval", "yval"]

    inst = getCustomOp(node)
    assert inst.get_normal_output_shape() == (1, 2, 4)
    assert inst.get_exp_cycles() == 20

    ret = execute_onnx(model, {"cond": cond, "xval": xval, "yval": yval})
    assert (ret["out"] == expected).all()

    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())
    assert model.graph.node[0].op_type == "Where_rtl"
    assert model.graph.node[0].domain == "finn.custom_op.fpgadataflow.rtl"


@pytest.mark.fpgadataflow
def test_convert_onnx_where_broadcast_to_where():
    cond_shape = [3, 1]
    x_shape = [4]
    y_shape = [2, 1, 1]
    out_shape = [2, 3, 4]
    model = _make_onnx_where_model(
        cond_shape=cond_shape, x_shape=x_shape, y_shape=y_shape, out_shape=out_shape
    )
    cond, xval, yval = _make_inputs(cond_shape=cond_shape, x_shape=x_shape, y_shape=y_shape)
    expected = np.where(cond.astype(bool), xval, yval)

    ret = execute_onnx(model, {"cond": cond.astype(bool), "xval": xval, "yval": yval})
    assert (ret["out"] == expected).all()

    model.set_tensor_datatype("cond", DataType["BINARY"])
    model = model.transform(InferWhereLayer())
    node = model.graph.node[0]
    assert node.op_type == "Where"
    assert node.domain == "finn.custom_op.fpgadataflow"

    inst = getCustomOp(node)
    assert inst.get_normal_input_shape(0) == tuple(cond_shape)
    assert inst.get_normal_input_shape(1) == tuple(x_shape)
    assert inst.get_normal_input_shape(2) == tuple(y_shape)
    assert inst.get_normal_output_shape() == tuple(out_shape)
    assert inst.get_folded_input_shape(0) == (3, 1, 1)
    assert inst.get_folded_input_shape(1) == (4, 1)
    assert inst.get_folded_input_shape(2) == (2, 1, 1, 1)

    ret = execute_onnx(model, {"cond": cond, "xval": xval, "yval": yval})
    assert (ret["out"] == expected).all()


@pytest.mark.fpgadataflow
def test_convert_onnx_where_scalar_broadcast_to_where():
    cond_shape = []
    x_shape = [2, 3]
    y_shape = [1, 3]
    out_shape = [2, 3]
    model = _make_onnx_where_model(
        cond_shape=cond_shape, x_shape=x_shape, y_shape=y_shape, out_shape=out_shape
    )
    cond, xval, yval = _make_inputs(cond_shape=cond_shape, x_shape=x_shape, y_shape=y_shape)
    expected = np.where(cond.astype(bool), xval, yval)

    ret = execute_onnx(model, {"cond": cond.astype(bool), "xval": xval, "yval": yval})
    assert (ret["out"] == expected).all()

    model.set_tensor_datatype("cond", DataType["BINARY"])
    model = model.transform(InferWhereLayer())
    node = model.graph.node[0]
    inst = getCustomOp(node)

    assert inst.get_normal_input_shape(0) == tuple(cond_shape)
    assert inst.get_normal_input_shape(1) == tuple(x_shape)
    assert inst.get_normal_input_shape(2) == tuple(y_shape)
    assert inst.get_normal_output_shape() == tuple(out_shape)
    assert inst.get_folded_input_shape(0) == (1, 1)
    assert inst.get_nodeattr("CondRank") == 0

    ret = execute_onnx(model, {"cond": cond, "xval": xval, "yval": yval})
    assert (ret["out"] == expected).all()


@pytest.mark.fpgadataflow
def test_convert_onnx_where_float32_to_where():
    model = _make_onnx_where_model(finn_dtype=DataType["FLOAT32"])
    cond, xval, yval = _make_inputs()
    xval = xval + 0.25
    yval = yval + 0.5
    expected = np.where(cond.astype(bool), xval, yval)

    model.set_tensor_datatype("cond", DataType["BINARY"])
    model = model.transform(InferWhereLayer())
    node = model.graph.node[0]
    inst = getCustomOp(node)

    assert inst.get_input_datatype(1) == DataType["FLOAT32"]
    assert inst.get_output_datatype() == DataType["FLOAT32"]

    ret = execute_onnx(model, {"cond": cond, "xval": xval, "yval": yval})
    assert (ret["out"] == expected).all()


@pytest.mark.fpgadataflow
@pytest.mark.parametrize(
    "finn_dtype",
    [DataType["INT8"], DataType["UINT4"], DataType["BIPOLAR"], DataType["FLOAT32"]],
)
def test_where_python_execution(finn_dtype):
    model = _make_where_model(finn_dtype=finn_dtype)
    cond, xval, yval = _make_inputs()
    if finn_dtype == DataType["BIPOLAR"]:
        xval = np.where(xval % 2 == 0, -1, 1).astype(np.float32)
        yval = -xval
    elif finn_dtype == DataType["UINT4"]:
        yval = (15 - xval).astype(np.float32)
    expected = np.where(cond.astype(bool), xval, yval)

    ret = execute_onnx(model, {"cond": cond, "xval": xval, "yval": yval})
    assert (ret["out"] == expected).all()


@pytest.mark.fpgadataflow
def test_where_python_execution_broadcast():
    cond_shape = [3, 1]
    x_shape = [4]
    y_shape = [2, 1, 1]
    out_shape = [2, 3, 4]
    model = _make_where_model(
        pe=2,
        cond_shape=cond_shape,
        x_shape=x_shape,
        y_shape=y_shape,
        out_shape=out_shape,
    )
    cond, xval, yval = _make_inputs(cond_shape=cond_shape, x_shape=x_shape, y_shape=y_shape)
    expected = np.where(cond.astype(bool), xval, yval)

    ret = execute_onnx(model, {"cond": cond, "xval": xval, "yval": yval})
    assert (ret["out"] == expected).all()


@pytest.mark.fpgadataflow
@pytest.mark.parametrize(
    "finn_dtype,fold_width",
    [
        (DataType["INT8"], 16),
        (DataType["UINT4"], 8),
        (DataType["BIPOLAR"], 2),
        (DataType["FLOAT32"], 64),
    ],
)
def test_where_rtl_codegen(tmp_path, finn_dtype, fold_width):
    model = _make_where_model(pe=2, finn_dtype=finn_dtype)
    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())

    node = model.graph.node[0]
    inst = getCustomOp(node)
    inst.set_nodeattr("code_gen_dir_ipgen", str(tmp_path))
    inst.code_generation_ipgen(model, FPGA_PART, CLK_NS)

    topname = inst.get_nodeattr("gen_top_module")
    assert topname == node.name
    wrapper = tmp_path / (topname + ".v")
    core_wrapper = tmp_path / (topname + "_core.sv")
    core = tmp_path / "where.sv"
    input_gen = tmp_path / "input_gen.sv"
    assert wrapper.is_file()
    assert core_wrapper.is_file()
    assert core.is_file()
    assert input_gen.is_file()
    wrapper_text = wrapper.read_text()
    core_wrapper_text = core_wrapper.read_text()
    assert "parameter COND_WIDTH = 2" in wrapper_text
    assert "parameter X_WIDTH = %d" % fold_width in wrapper_text
    assert "parameter Y_WIDTH = %d" % fold_width in wrapper_text
    assert "parameter OUT_WIDTH = %d" % fold_width in wrapper_text
    assert ".DATA_WIDTH(%d)" % finn_dtype.bitwidth() in core_wrapper_text
    assert ".PE(2)" in core_wrapper_text
    assert ".NDIMS(3)" in core_wrapper_text
    assert '.RAM_STYLE("auto")' in core_wrapper_text
    assert "in2_V_TDATA" in wrapper_text
    assert "out0_V_TVALID" in wrapper_text

    ipi_cmds = inst.code_generation_ipi()
    assert any("input_gen.sv" in cmd for cmd in ipi_cmds)
    assert any("where.sv" in cmd for cmd in ipi_cmds)
    assert any(topname + "_core.sv" in cmd for cmd in ipi_cmds)
    assert any(topname + ".v" in cmd for cmd in ipi_cmds)
    assert any("create_bd_cell" in cmd and topname in cmd for cmd in ipi_cmds)


@pytest.mark.fpgadataflow
def test_where_rtl_codegen_broadcast(tmp_path):
    model = _make_where_model(
        pe=2,
        cond_shape=[3, 1],
        x_shape=[4],
        y_shape=[2, 1, 1],
        out_shape=[2, 3, 4],
    )
    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())

    node = model.graph.node[0]
    inst = getCustomOp(node)
    inst.set_nodeattr("code_gen_dir_ipgen", str(tmp_path))
    inst.code_generation_ipgen(model, FPGA_PART, CLK_NS)

    topname = inst.get_nodeattr("gen_top_module")
    wrapper_text = (tmp_path / (topname + ".v")).read_text()
    core_wrapper_text = (tmp_path / (topname + "_core.sv")).read_text()
    assert "parameter COND_WIDTH = 1" in wrapper_text
    assert "parameter X_WIDTH = 16" in wrapper_text
    assert "parameter Y_WIDTH = 8" in wrapper_text
    assert "parameter OUT_WIDTH = 16" in wrapper_text
    assert ".NDIMS(3)" in core_wrapper_text
    assert ".COND_NDIMS(2)" in core_wrapper_text
    assert ".X_NDIMS(1)" in core_wrapper_text
    assert ".Y_NDIMS(3)" in core_wrapper_text
    assert ".OUT_SHAPE('{ 2, 3, 4 })" in core_wrapper_text
    assert ".COND_SHAPE('{ 3, 1 })" in core_wrapper_text
    assert ".X_SHAPE('{ 4 })" in core_wrapper_text
    assert ".Y_SHAPE('{ 2, 1, 1 })" in core_wrapper_text


@pytest.mark.fpgadataflow
def test_where_rtl_codegen_scalar_broadcast(tmp_path):
    model = _make_where_model(
        pe=3,
        cond_shape=[],
        x_shape=[2, 3],
        y_shape=[1, 3],
        out_shape=[2, 3],
    )
    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())

    node = model.graph.node[0]
    inst = getCustomOp(node)
    inst.set_nodeattr("code_gen_dir_ipgen", str(tmp_path))
    inst.code_generation_ipgen(model, FPGA_PART, CLK_NS)

    topname = inst.get_nodeattr("gen_top_module")
    wrapper_text = (tmp_path / (topname + ".v")).read_text()
    core_wrapper_text = (tmp_path / (topname + "_core.sv")).read_text()
    assert "parameter COND_WIDTH = 1" in wrapper_text
    assert "parameter X_WIDTH = 24" in wrapper_text
    assert "parameter Y_WIDTH = 24" in wrapper_text
    assert "parameter OUT_WIDTH = 24" in wrapper_text
    assert ".NDIMS(2)" in core_wrapper_text
    assert ".COND_NDIMS(1)" in core_wrapper_text
    assert ".COND_SHAPE('{ 1 })" in core_wrapper_text


@pytest.mark.fpgadataflow
def test_where_resource_estimation():
    model = _make_where_model(pe=2)
    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())

    expected = {
        "BRAM_18K": 0,
        "BRAM_efficiency": 1,
        "LUT": 80,
        "URAM": 0,
        "URAM_efficiency": 1,
        "DSP": 0,
    }
    resources = model.analysis(partial(res_estimation, fpgapart=FPGA_PART))
    assert len(resources) == 1
    assert list(resources.values())[0] == expected

    complete_resources = model.analysis(partial(res_estimation_complete, fpgapart=FPGA_PART))
    assert len(complete_resources) == 1
    complete_node_resources = list(complete_resources.values())[0]
    assert len(complete_node_resources) == 3
    assert all(x == expected for x in complete_node_resources)


@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
@pytest.mark.parametrize("pe", [1, 2])
def test_where_rtlsim(pe):
    model = _make_where_model(pe=pe)
    cond, xval, yval = _make_inputs()
    expected = np.where(cond.astype(bool), xval, yval)

    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(FPGA_PART, CLK_NS))
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareRTLSim())

    ret = execute_onnx(model, {"cond": cond, "xval": xval, "yval": yval})
    assert (ret["out"] == expected).all()

    node = model.get_nodes_by_op_type("Where_rtl")[0]
    inst = getCustomOp(node)
    cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
    exp_cycles_dict = model.analysis(exp_cycles_per_layer)
    exp_cycles = exp_cycles_dict[node.name]
    assert np.isclose(exp_cycles, cycles_rtlsim, atol=10)
    assert exp_cycles != 0


@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_where_rtlsim_broadcast():
    cond_shape = [3, 1]
    x_shape = [4]
    y_shape = [2, 1, 1]
    out_shape = [2, 3, 4]
    model = _make_where_model(
        pe=2,
        cond_shape=cond_shape,
        x_shape=x_shape,
        y_shape=y_shape,
        out_shape=out_shape,
    )
    cond, xval, yval = _make_inputs(cond_shape=cond_shape, x_shape=x_shape, y_shape=y_shape)
    expected = np.where(cond.astype(bool), xval, yval)

    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(FPGA_PART, CLK_NS))
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareRTLSim())

    ret = execute_onnx(model, {"cond": cond, "xval": xval, "yval": yval})
    assert (ret["out"] == expected).all()

    node = model.get_nodes_by_op_type("Where_rtl")[0]
    inst = getCustomOp(node)
    cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
    exp_cycles_dict = model.analysis(exp_cycles_per_layer)
    exp_cycles = exp_cycles_dict[node.name]
    assert np.isclose(exp_cycles, cycles_rtlsim, atol=15)
    assert exp_cycles != 0


@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
@pytest.mark.parametrize("pe", [1, 2])
def test_where_stitched_ip_rtlsim(pe):
    model = _prepare_where_stitched_ip_model(pe=pe)
    cond, xval, yval = _make_inputs()
    expected = np.where(cond.astype(bool), xval, yval)

    model.set_metadata_prop("exec_mode", "rtlsim")

    ret = execute_onnx(model, {"cond": cond, "xval": xval, "yval": yval})
    assert (ret["out"] == expected).all()


@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_where_stitched_ip_rtlsim_broadcast():
    cond_shape = [1, 3, 1]
    x_shape = [1, 1, 4]
    y_shape = [1, 2, 1, 1]
    out_shape = [1, 2, 3, 4]
    model = _prepare_where_stitched_ip_model(
        pe=2,
        cond_shape=cond_shape,
        x_shape=x_shape,
        y_shape=y_shape,
        out_shape=out_shape,
    )
    cond, xval, yval = _make_inputs(cond_shape=cond_shape, x_shape=x_shape, y_shape=y_shape)
    expected = np.where(cond.astype(bool), xval, yval)

    model.set_metadata_prop("exec_mode", "rtlsim")

    ret = execute_onnx(model, {"cond": cond, "xval": xval, "yval": yval})
    assert (ret["out"] == expected).all()


@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_where_stitched_ip_synth_ooc():
    model = _prepare_where_stitched_ip_model(pe=2)
    model = model.transform(CreateStitchedIP(FPGA_PART, CLK_NS, run_pnr=True))
    ret = parse_ooc_synth_results(model.get_metadata_prop("vivado_stitch_proj"))
    assert ret is not None

    assert ret["LUT"] > 0
    assert ret["FF"] > 0
    assert ret["DSP"] == 0
    assert ret.get("BRAM_18K", 0) == 0
    assert ret.get("BRAM_36K", 0) == 0
    assert ret["WNS"] >= 0
