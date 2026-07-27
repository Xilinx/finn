# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import numpy as np
from functools import partial
from importlib import import_module
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
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferPad1DLayer
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

FPGA_PART = "xc7z020clg400-1"
CLK_NS = 10


def _make_graph(nodes, output_shape, initializers, finn_dtype=DataType["INT8"]):
    patch_shape = [1, 3, 4]
    patches = helper.make_tensor_value_info("patches", TensorProto.FLOAT, patch_shape)
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, output_shape)
    graph = helper.make_graph(nodes, "pad1d_test", [patches], [output], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model = ModelWrapper(model)
    for tensor_name in ["patches", "out"]:
        model.set_tensor_datatype(tensor_name, finn_dtype)
    for init in initializers:
        model.set_tensor_datatype(init.name, finn_dtype)
    return model


def _make_concat_cls_model():
    cls_values = np.asarray([[[1, -2, 3, -4]]], dtype=np.float32)
    cls_init = numpy_helper.from_array(cls_values, name="cls")
    concat = helper.make_node(
        "Concat",
        ["cls", "patches"],
        ["out"],
        axis=1,
        name="concat_cls",
    )
    model = _make_graph([concat], [1, 4, 4], [cls_init])
    return model, cls_values


def _make_concat_custom_pad_model():
    left_values = np.asarray([[[10, 11, 12, 13], [20, 21, 22, 23]]], dtype=np.float32)
    right_values = np.asarray(
        [[[30, 31, 32, 33], [40, 41, 42, 43], [50, 51, 52, 53]]],
        dtype=np.float32,
    )
    left_init = numpy_helper.from_array(left_values, name="left_pad")
    right_init = numpy_helper.from_array(right_values, name="right_pad")
    concat = helper.make_node(
        "Concat",
        ["left_pad", "patches", "right_pad"],
        ["out"],
        axis=1,
        name="concat_pad",
    )
    model = _make_graph([concat], [1, 8, 4], [left_init, right_init])
    return model, left_values, right_values


def _make_pad1d_model(
    pad_left=1,
    pad_right=1,
    simd=1,
    finn_dtype=DataType["INT8"],
    left_values=None,
    right_values=None,
):
    if left_values is None:
        left_values = np.asarray([[[1, -2, 3, -4]]], dtype=np.float32)
    if right_values is None:
        right_values = np.asarray([[[5, 6, -7, 8]]], dtype=np.float32)

    left_init = numpy_helper.from_array(left_values.astype(np.float32), name="left_pad")
    right_init = numpy_helper.from_array(right_values.astype(np.float32), name="right_pad")
    pad1d = helper.make_node(
        "Pad1D",
        ["patches", "left_pad", "right_pad"],
        ["out"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        name="Pad1D_0",
        NumTokens=3,
        NumChannels=4,
        PadLeft=pad_left,
        PadRight=pad_right,
        SIMD=simd,
        inputDataType=finn_dtype.name,
        outputDataType=finn_dtype.name,
    )
    model = _make_graph([pad1d], [1, 3 + pad_left + pad_right, 4], [left_init, right_init])
    return model, left_values, right_values


def _prepare_pad1d_stitched_ip_model(simd=1, pad_left=1, pad_right=1):
    model, left_values, right_values = _make_pad1d_model(
        pad_left=pad_left,
        pad_right=pad_right,
        simd=simd,
    )
    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(InsertFIFO(create_shallow_fifos=True))
    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(FPGA_PART, CLK_NS))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(FPGA_PART, CLK_NS, vitis=False))
    return model, left_values, right_values


def _make_input_dict(model, patches):
    return {model.graph.input[0].name: patches}


def _expanded_pad(values, count):
    if count == 0:
        return np.zeros((1, 0, values.shape[-1]), dtype=np.float32)
    if values.shape[1] == 1 and count > 1:
        return np.repeat(values, count, axis=1)
    return values


@pytest.mark.fpgadataflow
@pytest.mark.parametrize("finn_dtype", [DataType["INT8"], DataType["FLOAT32"]])
def test_convert_concat_cls_to_pad1d(finn_dtype):
    model, cls_values = _make_concat_cls_model()
    for tensor_name in ["patches", "cls", "out"]:
        model.set_tensor_datatype(tensor_name, finn_dtype)
    patches = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    expected = np.concatenate([cls_values, patches], axis=1)

    ret = execute_onnx(model, _make_input_dict(model, patches))
    assert (ret["out"] == expected).all()

    model = model.transform(InferPad1DLayer())
    node = model.graph.node[0]
    assert node.op_type == "Pad1D"
    assert node.domain == "finn.custom_op.fpgadataflow"
    assert list(node.input)[0] == "patches"

    inst = getCustomOp(node)
    assert inst.get_nodeattr("PadLeft") == 1
    assert inst.get_nodeattr("PadRight") == 0
    assert inst.get_normal_output_shape() == (1, 4, 4)
    assert inst.get_exp_cycles() == 16

    ret = execute_onnx(model, _make_input_dict(model, patches))
    assert (ret["out"] == expected).all()

    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())
    assert model.graph.node[0].op_type == "Pad1D_rtl"
    assert model.graph.node[0].domain == "finn.custom_op.fpgadataflow.rtl"


@pytest.mark.fpgadataflow
def test_convert_concat_custom_padding_to_pad1d():
    model, left_values, right_values = _make_concat_custom_pad_model()
    patches = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    expected = np.concatenate([left_values, patches, right_values], axis=1)

    model = model.transform(InferPad1DLayer())
    node = model.graph.node[0]
    assert node.op_type == "Pad1D"

    inst = getCustomOp(node)
    assert inst.get_nodeattr("PadLeft") == 2
    assert inst.get_nodeattr("PadRight") == 3
    assert inst.get_normal_output_shape() == (1, 8, 4)

    ret = execute_onnx(model, _make_input_dict(model, patches))
    assert (ret["out"] == expected).all()


@pytest.mark.fpgadataflow
def test_convert_concat_does_not_mutate_rejected_node(monkeypatch):
    model, _, _ = _make_concat_custom_pad_model()
    model.set_tensor_datatype("right_pad", DataType["UINT8"])

    original_get_tensor_datatype = model.get_tensor_datatype
    original_set_tensor_datatype = model.set_tensor_datatype
    datatype_updates = []

    def get_tensor_datatype(tensor_name):
        if tensor_name == "left_pad":
            return None
        return original_get_tensor_datatype(tensor_name)

    def set_tensor_datatype(tensor_name, datatype):
        datatype_updates.append((tensor_name, datatype))
        original_set_tensor_datatype(tensor_name, datatype)

    monkeypatch.setattr(model, "get_tensor_datatype", get_tensor_datatype)
    monkeypatch.setattr(model, "set_tensor_datatype", set_tensor_datatype)

    model, modified = InferPad1DLayer().apply(model)

    assert not modified
    assert model.graph.node[0].op_type == "Concat"
    assert datatype_updates == []


@pytest.mark.fpgadataflow
def test_pad1d_python_execution_with_repeated_padding():
    left_values = np.asarray([[[1, 2, 3, 4]]], dtype=np.float32)
    right_values = np.asarray([[[-1, -2, -3, -4]]], dtype=np.float32)
    model, _, _ = _make_pad1d_model(
        pad_left=2,
        pad_right=1,
        left_values=left_values,
        right_values=right_values,
    )
    patches = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    expected = np.concatenate([np.repeat(left_values, 2, axis=1), patches, right_values], axis=1)

    ret = execute_onnx(model, _make_input_dict(model, patches))
    assert (ret["out"] == expected).all()


@pytest.mark.fpgadataflow
@pytest.mark.parametrize(
    "finn_dtype,left_values,expected_left_data,right_values,expected_right_data",
    [
        (
            DataType["INT8"],
            np.asarray([[[1, -2, 3, -4]]], dtype=np.float32),
            "32'hfc03fe01",
            np.asarray([[[5, 6, -7, 8]]], dtype=np.float32),
            "32'h08f90605",
        ),
        (
            DataType["UINT4"],
            np.asarray([[[1, 2, 3, 4]]], dtype=np.float32),
            "16'h4321",
            np.asarray([[[4, 3, 2, 1]]], dtype=np.float32),
            "16'h1234",
        ),
        (
            DataType["BIPOLAR"],
            np.asarray([[[1, -1, 1, -1]]], dtype=np.float32),
            "4'h5",
            np.asarray([[[-1, 1, -1, 1]]], dtype=np.float32),
            "4'ha",
        ),
        (
            DataType["FLOAT32"],
            np.asarray([[[1, -2, 3, -4]]], dtype=np.float32),
            "128'hc080000040400000c00000003f800000",
            np.asarray([[[5, 6, -7, 8]]], dtype=np.float32),
            "128'h41000000c0e0000040c0000040a00000",
        ),
    ],
)
def test_pad1d_rtl_codegen(
    tmp_path,
    finn_dtype,
    left_values,
    expected_left_data,
    right_values,
    expected_right_data,
):
    model, _, _ = _make_pad1d_model(
        pad_left=1,
        pad_right=1,
        simd=2,
        finn_dtype=finn_dtype,
        left_values=left_values,
        right_values=right_values,
    )
    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())

    node = model.graph.node[0]
    inst = getCustomOp(node)
    inst.set_nodeattr("code_gen_dir_ipgen", str(tmp_path))
    inst.code_generation_ipgen(model, FPGA_PART, CLK_NS)

    topname = inst.get_nodeattr("gen_top_module")
    assert topname == node.name
    wrapper = tmp_path / (topname + ".v")
    assert wrapper.is_file()
    assert inst.get_rtl_file_list(abspath=True)[0].endswith("/pad1d.sv")
    wrapper_text = wrapper.read_text()
    assert "parameter FOLD_WIDTH = %d" % (2 * finn_dtype.bitwidth()) in wrapper_text
    assert ".SIMD(2)" in wrapper_text
    assert ".PAD_LEFT_TOKENS(1)" in wrapper_text
    assert ".PAD_RIGHT_TOKENS(1)" in wrapper_text
    assert "PAD_LEFT_DATA = %s" % expected_left_data in wrapper_text
    assert "PAD_RIGHT_DATA = %s" % expected_right_data in wrapper_text
    assert ".PAD_LEFT_DATA(PAD_LEFT_DATA)" in wrapper_text
    assert ".PAD_RIGHT_DATA(PAD_RIGHT_DATA)" in wrapper_text
    assert "out0_V_TVALID" in wrapper_text
    assert "= '0" not in wrapper_text

    ipi_cmds = inst.code_generation_ipi()
    assert any("add_files -copy_to" in cmd and "pad1d.sv" in cmd for cmd in ipi_cmds)
    assert any("create_bd_cell" in cmd and topname in cmd for cmd in ipi_cmds)


@pytest.mark.fpgadataflow
@pytest.mark.parametrize(
    "pad_left,pad_right,expected_left_data,expected_right_data",
    [
        (1, 0, "4'h5", "4'h0"),
        (0, 1, "4'h0", "4'ha"),
    ],
)
def test_pad1d_bipolar_one_sided_rtl_codegen(
    tmp_path,
    pad_left,
    pad_right,
    expected_left_data,
    expected_right_data,
):
    left_values = np.asarray([[[1, -1, 1, -1]]], dtype=np.float32)
    right_values = np.asarray([[[-1, 1, -1, 1]]], dtype=np.float32)
    model, _, _ = _make_pad1d_model(
        pad_left=pad_left,
        pad_right=pad_right,
        simd=2,
        finn_dtype=DataType["BIPOLAR"],
        left_values=left_values,
        right_values=right_values,
    )
    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())

    node = model.graph.node[0]
    inst = getCustomOp(node)
    inst.set_nodeattr("code_gen_dir_ipgen", str(tmp_path))
    inst.code_generation_ipgen(model, FPGA_PART, CLK_NS)

    wrapper_text = (tmp_path / (inst.get_nodeattr("gen_top_module") + ".v")).read_text()
    assert "PAD_LEFT_DATA = %s" % expected_left_data in wrapper_text
    assert "PAD_RIGHT_DATA = %s" % expected_right_data in wrapper_text


@pytest.mark.fpgadataflow
def test_pad1d_resource_estimation():
    model, _, _ = _make_pad1d_model(pad_left=1, pad_right=1, simd=2)
    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())

    expected = {
        "BRAM_18K": 0,
        "BRAM_efficiency": 1,
        "LUT": 136,
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
@pytest.mark.parametrize("simd,pad_left,pad_right", [(1, 1, 0), (2, 2, 1)])
def test_pad1d_rtlsim(simd, pad_left, pad_right):
    model, left_values, right_values = _make_pad1d_model(
        pad_left=pad_left,
        pad_right=pad_right,
        simd=simd,
    )
    patches = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    expected = np.concatenate(
        [_expanded_pad(left_values, pad_left), patches, _expanded_pad(right_values, pad_right)],
        axis=1,
    )

    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(FPGA_PART, CLK_NS))
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareRTLSim())

    ret = execute_onnx(model, _make_input_dict(model, patches))
    assert (ret["out"] == expected).all()

    node = model.get_nodes_by_op_type("Pad1D_rtl")[0]
    inst = getCustomOp(node)
    cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
    exp_cycles_dict = model.analysis(exp_cycles_per_layer)
    exp_cycles = exp_cycles_dict[node.name]
    assert np.isclose(exp_cycles, cycles_rtlsim, atol=10)
    assert exp_cycles != 0


@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
@pytest.mark.parametrize("simd,pad_left,pad_right", [(1, 1, 0), (2, 2, 1)])
def test_pad1d_stitched_ip_rtlsim(simd, pad_left, pad_right):
    model, left_values, right_values = _prepare_pad1d_stitched_ip_model(
        simd=simd,
        pad_left=pad_left,
        pad_right=pad_right,
    )
    patches = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    expected = np.concatenate(
        [_expanded_pad(left_values, pad_left), patches, _expanded_pad(right_values, pad_right)],
        axis=1,
    )

    model.set_metadata_prop("exec_mode", "rtlsim")

    ret = execute_onnx(model, _make_input_dict(model, patches))
    assert (ret["out"] == expected).all()


@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_pad1d_stitched_ip_synth_ooc():
    synth_ooc = import_module("finn.transformation.fpgadataflow.synth_ooc")
    model, _, _ = _prepare_pad1d_stitched_ip_model(simd=2, pad_left=1, pad_right=1)
    model = model.transform(synth_ooc.SynthOutOfContext(FPGA_PART, CLK_NS))
    ret = model.get_metadata_prop("res_total_ooc_synth")
    assert ret is not None
    ret = eval(ret)

    assert ret["LUT"] > 0
    assert ret["FF"] > 0
    assert ret["DSP"] == 0
    assert ret["BRAM"] == 0
    assert ret["WNS"] >= 0
