# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import numpy as np
from functools import partial
from onnx import TensorProto, helper, numpy_helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.analysis.fpgadataflow.res_estimation import (
    res_estimation,
    res_estimation_complete,
)
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferPad1DLayer
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
PATCHES = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
LEFT_PAD_VALUES = {
    1: np.asarray([[[1, -2, 3, -4]]], dtype=np.float32),
    2: np.asarray([[[10, 11, 12, 13], [20, 21, 22, 23]]], dtype=np.float32),
}
RIGHT_PAD_VALUES = {
    0: None,
    1: np.asarray([[[5, 6, -7, 8]]], dtype=np.float32),
    3: np.asarray(
        [[[30, 31, 32, 33], [40, 41, 42, 43], [50, 51, 52, 53]]],
        dtype=np.float32,
    ),
}
RTL_CODEGEN_CASES = {
    "INT8": {
        "left_values": LEFT_PAD_VALUES[1],
        "right_values": RIGHT_PAD_VALUES[1],
        "left_data": "32'hfc03fe01",
        "right_data": "32'h08f90605",
        "zero_data": "32'h00000000",
    },
    "UINT4": {
        "left_values": np.asarray([[[1, 2, 3, 4]]], dtype=np.float32),
        "right_values": np.asarray([[[4, 3, 2, 1]]], dtype=np.float32),
        "left_data": "16'h4321",
        "right_data": "16'h1234",
        "zero_data": "16'h0000",
    },
    "BIPOLAR": {
        "left_values": np.asarray([[[1, -1, 1, -1]]], dtype=np.float32),
        "right_values": np.asarray([[[-1, 1, -1, 1]]], dtype=np.float32),
        "left_data": "4'h5",
        "right_data": "4'ha",
        "zero_data": "4'h0",
    },
}


def make_modelwrapper(nodes, output_shape, initializers, finn_dtype=DataType["INT8"]):
    patch_shape = [1, 3, 4]
    patches = helper.make_tensor_value_info("patches", TensorProto.FLOAT, patch_shape)
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, output_shape)
    graph = helper.make_graph(nodes, "pad1d_test", [patches], [output], initializers)
    model = qonnx_make_model(
        graph,
        producer_name="pad1d-model",
        opset_imports=[helper.make_opsetid("", 11)],
    )
    model = ModelWrapper(model)
    for tensor_name in ["patches", "out"]:
        model.set_tensor_datatype(tensor_name, finn_dtype)
    for init in initializers:
        model.set_tensor_datatype(init.name, finn_dtype)
    return model


def make_concat_modelwrapper(pad_left, pad_right):
    left_values = LEFT_PAD_VALUES[pad_left]
    right_values = RIGHT_PAD_VALUES[pad_right]
    left_init = numpy_helper.from_array(left_values, name="left_pad")
    concat_inputs = ["left_pad", "patches"]
    initializers = [left_init]
    if right_values is not None:
        right_init = numpy_helper.from_array(right_values, name="right_pad")
        concat_inputs.append("right_pad")
        initializers.append(right_init)
    concat = helper.make_node(
        "Concat",
        concat_inputs,
        ["out"],
        axis=1,
        name="concat_pad",
    )
    output_shape = [1, 3 + pad_left + pad_right, 4]
    return make_modelwrapper([concat], output_shape, initializers)


def make_pad1d_modelwrapper(
    pad_left=1,
    pad_right=1,
    simd=1,
    finn_dtype=DataType["INT8"],
    left_values=None,
    right_values=None,
):
    if left_values is None:
        left_values = LEFT_PAD_VALUES[1]
    if right_values is None:
        right_values = RIGHT_PAD_VALUES[1]

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
    output_shape = [1, 3 + pad_left + pad_right, 4]
    return make_modelwrapper([pad1d], output_shape, [left_init, right_init], finn_dtype)


def prepare_pad1d_stitched_ip_model(model, run_pnr=False):
    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(InsertFIFO(create_shallow_fifos=True))
    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(FPGA_PART, CLK_NS))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(FPGA_PART, CLK_NS, run_pnr=run_pnr))
    return model


def prepare_inputs(patches):
    return {"patches": patches}


def expanded_pad(values, count):
    if count == 0:
        return np.zeros((1, 0, values.shape[-1]), dtype=np.float32)
    if values.shape[1] == 1 and count > 1:
        return np.repeat(values, count, axis=1)
    return values


def prepare_expected(patches, pad_left, pad_right, left_values, right_values):
    return np.concatenate(
        [expanded_pad(left_values, pad_left), patches, expanded_pad(right_values, pad_right)],
        axis=1,
    )


# left padding
@pytest.mark.parametrize("pad_left", [1, 2])
# right padding
@pytest.mark.parametrize("pad_right", [0, 3])
@pytest.mark.fpgadataflow
def test_convert_concat_to_pad1d(pad_left, pad_right):
    left_values = LEFT_PAD_VALUES[pad_left]
    right_values = RIGHT_PAD_VALUES[pad_right]
    model = make_concat_modelwrapper(pad_left, pad_right)
    expected_values = [left_values, PATCHES]
    if right_values is not None:
        expected_values.append(right_values)
    expected = np.concatenate(expected_values, axis=1)

    y_produced = oxe.execute_onnx(model, prepare_inputs(PATCHES))["out"]
    assert (y_produced == expected).all()

    model = model.transform(InferPad1DLayer())
    node = model.graph.node[0]
    assert node.op_type == "Pad1D"
    assert node.domain == "finn.custom_op.fpgadataflow"
    assert list(node.input)[0] == "patches"

    inst = getCustomOp(node)
    assert inst.get_nodeattr("PadLeft") == pad_left
    assert inst.get_nodeattr("PadRight") == pad_right
    assert inst.get_normal_output_shape() == (1, 3 + pad_left + pad_right, 4)
    assert inst.get_exp_cycles() == (3 + pad_left + pad_right) * 4

    y_produced = oxe.execute_onnx(model, prepare_inputs(PATCHES))["out"]
    assert (y_produced == expected).all()

    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())
    assert model.graph.node[0].op_type == "Pad1D_rtl"
    assert model.graph.node[0].domain == "finn.custom_op.fpgadataflow.rtl"


@pytest.mark.fpgadataflow
def test_convert_concat_does_not_mutate_rejected_node(monkeypatch):
    model = make_concat_modelwrapper(pad_left=2, pad_right=3)
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
    model = make_pad1d_modelwrapper(
        pad_left=2,
        pad_right=1,
        left_values=left_values,
        right_values=right_values,
    )
    expected = prepare_expected(PATCHES, 2, 1, left_values, right_values)

    y_produced = oxe.execute_onnx(model, prepare_inputs(PATCHES))["out"]
    assert (y_produced == expected).all()


# data types
@pytest.mark.parametrize("finn_dtype", [DataType["INT8"], DataType["UINT4"], DataType["BIPOLAR"]])
# padded sides
@pytest.mark.parametrize("pad_sides", ["both", "left", "right"])
@pytest.mark.fpgadataflow
def test_pad1d_rtl_codegen(tmp_path, finn_dtype, pad_sides):
    pad_left = int(pad_sides in ["both", "left"])
    pad_right = int(pad_sides in ["both", "right"])
    codegen_case = RTL_CODEGEN_CASES[finn_dtype.name]
    left_values = codegen_case["left_values"]
    right_values = codegen_case["right_values"]
    expected_left_data = codegen_case["left_data"] if pad_left else codegen_case["zero_data"]
    expected_right_data = codegen_case["right_data"] if pad_right else codegen_case["zero_data"]

    model = make_pad1d_modelwrapper(
        pad_left=pad_left,
        pad_right=pad_right,
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
    assert ".PAD_LEFT_TOKENS(%d)" % pad_left in wrapper_text
    assert ".PAD_RIGHT_TOKENS(%d)" % pad_right in wrapper_text
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
def test_pad1d_resource_estimation():
    model = make_pad1d_modelwrapper(pad_left=1, pad_right=1, simd=2)
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


# Keep these settings coupled to preserve the original slow Vivado matrix.
# SIMD and padding configuration
@pytest.mark.parametrize(
    "config",
    [
        pytest.param((1, 1, 0), id="simd1-left1-right0"),
        pytest.param((2, 2, 1), id="simd2-left2-right1"),
    ],
)
# execution mode
# Pad1D is RTL-only, so cppsim does not apply.
@pytest.mark.parametrize("exec_mode", ["rtlsim", "stitched_rtlsim"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_fpgadataflow_pad1d(config, exec_mode):
    simd, pad_left, pad_right = config
    left_values = LEFT_PAD_VALUES[1]
    right_values = RIGHT_PAD_VALUES[1]
    model = make_pad1d_modelwrapper(
        pad_left=pad_left,
        pad_right=pad_right,
        simd=simd,
        left_values=left_values,
        right_values=right_values,
    )
    input_dict = prepare_inputs(PATCHES)
    y_expected = prepare_expected(PATCHES, pad_left, pad_right, left_values, right_values)

    # golden reference before specializing
    y_produced = oxe.execute_onnx(model, input_dict)["out"]
    assert (y_produced == y_expected).all(), "Execution of hw layer failed"

    if exec_mode == "rtlsim":
        model = model.transform(SpecializeLayers(FPGA_PART))
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareIP(FPGA_PART, CLK_NS))
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(PrepareRTLSim())
    elif exec_mode == "stitched_rtlsim":
        model = prepare_pad1d_stitched_ip_model(model)
        model.set_metadata_prop("exec_mode", "rtlsim")
    else:
        raise Exception("Unknown exec_mode")

    y_produced = oxe.execute_onnx(model, input_dict)["out"].reshape(y_expected.shape)
    assert (y_produced == y_expected).all(), exec_mode + " failed"

    if exec_mode == "rtlsim":
        node = model.get_nodes_by_op_type("Pad1D_rtl")[0]
        cycles_rtlsim = getCustomOp(node).get_nodeattr("cycles_rtlsim")
        exp_cycles = model.analysis(exp_cycles_per_layer)[node.name]
        assert np.isclose(exp_cycles, cycles_rtlsim, atol=10)
        assert exp_cycles != 0


@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_pad1d_stitched_ip_synth_ooc():
    model = make_pad1d_modelwrapper(simd=2, pad_left=1, pad_right=1)
    model = prepare_pad1d_stitched_ip_model(model, run_pnr=True)

    vivado_stitch_proj = model.get_metadata_prop("vivado_stitch_proj")
    ret = parse_ooc_synth_results(vivado_stitch_proj)
    assert ret is not None

    assert ret["LUT"] > 0
    assert ret["FF"] > 0
    assert ret["DSP"] == 0
    assert ret["BRAM_18K"] == 0
    assert ret["BRAM_36K"] == 0
    assert ret["WNS"] >= 0
