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
from finn.util.test import tree_model_test
from finn.util.vivado import parse_ooc_synth_results

FPGA_PART = "xc7z020clg400-1"
CLK_NS = 10
PATCHES = np.arange(12, dtype=np.float32).reshape(1, 3, 4)


def make_pad1d_modelwrapper(pad_left, pad_right, pad_tokens, finn_dtype):
    left_pad_token, right_pad_token = pad_tokens
    patch_shape = [1, 3, 4]
    patches = helper.make_tensor_value_info("patches", TensorProto.FLOAT, patch_shape)
    output_shape = [1, patch_shape[1] + pad_left + pad_right, patch_shape[2]]
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, output_shape)

    concat_inputs = []
    initializers = []
    if pad_left > 0:
        left_values = np.repeat(left_pad_token, pad_left, axis=1)
        left_init = numpy_helper.from_array(left_values, name="left_pad")
        concat_inputs.append("left_pad")
        initializers.append(left_init)
    concat_inputs.append("patches")
    if pad_right > 0:
        right_values = np.repeat(right_pad_token, pad_right, axis=1)
        right_init = numpy_helper.from_array(right_values, name="right_pad")
        concat_inputs.append("right_pad")
        initializers.append(right_init)

    concat = helper.make_node("Concat", concat_inputs, ["out"], axis=1, name="concat_pad")
    graph = helper.make_graph([concat], "pad1d_test", [patches], [output], initializer=initializers)
    model = ModelWrapper(
        qonnx_make_model(
            graph,
            producer_name="pad1d-model",
            opset_imports=[helper.make_opsetid("", 11)],
        )
    )
    model.set_tensor_datatype("patches", finn_dtype)
    model.set_tensor_datatype("out", finn_dtype)
    for init in initializers:
        model.set_tensor_datatype(init.name, finn_dtype)
    return model


def prepare_inputs(input_tensor):
    return {"patches": input_tensor}


def prepare_expected(pad_left, pad_right, pad_tokens):
    left_pad_token, right_pad_token = pad_tokens
    values = [np.repeat(left_pad_token, pad_left, axis=1), PATCHES]
    values.append(np.repeat(right_pad_token, pad_right, axis=1))
    return np.concatenate(values, axis=1)


def infer_and_specialize_pad1d(model, simd):
    model = model.transform(InferPad1DLayer())
    pad1d_nodes = model.get_nodes_by_op_type("Pad1D")
    assert len(pad1d_nodes) == 1
    getCustomOp(pad1d_nodes[0]).set_nodeattr("SIMD", simd)
    return model.transform(SpecializeLayers(FPGA_PART))


def prepare_pad1d_stitched_ip_model(model, run_pnr=False):
    model = model.transform(InsertFIFO(create_shallow_fifos=True))
    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(FPGA_PART, CLK_NS))
    model = model.transform(HLSSynthIP())
    return model.transform(CreateStitchedIP(FPGA_PART, CLK_NS, run_pnr=run_pnr))


def expected_resources(pad_left, pad_right):
    return {
        "BRAM_18K": 0,
        "BRAM_efficiency": 1,
        "LUT": 128 + PATCHES.shape[-1] * max(1, pad_left + pad_right),
        "URAM": 0,
        "URAM_efficiency": 1,
        "DSP": 0,
    }


# SIMD and padding configuration
@pytest.mark.parametrize(
    "config",
    [
        pytest.param((1, 1, 0), id="simd1-left1-right0"),
        pytest.param((2, 2, 1), id="simd2-left2-right1"),
    ],
)
# datatype and pad token values (combined so values fit in dtype range)
@pytest.mark.parametrize(
    "dtype_and_tokens",
    [
        pytest.param(
            (
                DataType["INT8"],
                np.asarray([[[1, -2, 3, -4]]], dtype=np.float32),
                np.asarray([[[-5, 6, -7, 8]]], dtype=np.float32),
            ),
            id="INT8-mixed",
        ),
        pytest.param(
            (
                DataType["UINT4"],
                np.asarray([[[1, 2, 3, 4]]], dtype=np.float32),
                np.asarray([[[5, 6, 7, 8]]], dtype=np.float32),
            ),
            id="UINT4-positive",
        ),
        pytest.param(
            (
                DataType["INT6"],
                np.asarray([[[-1, -2, -3, -4]]], dtype=np.float32),
                np.asarray([[[5, 6, 7, 8]]], dtype=np.float32),
            ),
            id="INT6-negative-left",
        ),
    ],
)
# execution mode
@pytest.mark.parametrize("exec_mode", ["rtlsim", "stitched_rtlsim"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_fpgadataflow_pad1d(config, dtype_and_tokens, exec_mode):
    simd, pad_left, pad_right = config
    finn_dtype, left_pad_token, right_pad_token = dtype_and_tokens
    pad_tokens = (left_pad_token, right_pad_token)
    model = make_pad1d_modelwrapper(pad_left, pad_right, pad_tokens, finn_dtype)
    input_dict = prepare_inputs(PATCHES)
    y_expected = prepare_expected(pad_left, pad_right, pad_tokens)

    # Golden reference from the original Concat graph.
    y_produced = oxe.execute_onnx(model, input_dict)["out"]
    assert (y_produced == y_expected).all(), "Execution of Concat model failed"

    model = infer_and_specialize_pad1d(model, simd)

    expected = expected_resources(pad_left, pad_right)
    resources = model.analysis(partial(res_estimation, fpgapart=FPGA_PART))
    assert list(resources.values()) == [expected]
    complete_resources = model.analysis(partial(res_estimation_complete, fpgapart=FPGA_PART))
    assert list(complete_resources.values()) == [[expected]]

    if exec_mode == "rtlsim":
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
    simd, pad_left, pad_right = 2, 1, 1
    finn_dtype = DataType["INT8"]
    pad_tokens = (
        np.asarray([[[1, -2, 3, -4]]], dtype=np.float32),
        np.asarray([[[5, 6, -7, 8]]], dtype=np.float32),
    )
    model = make_pad1d_modelwrapper(pad_left, pad_right, pad_tokens, finn_dtype)
    input_dict = prepare_inputs(PATCHES)
    y_expected = prepare_expected(pad_left, pad_right, pad_tokens)

    # Golden reference from the original Concat graph.
    y_produced = oxe.execute_onnx(model, input_dict)["out"]
    assert (y_produced == y_expected).all(), "Execution of Concat model failed"

    model = infer_and_specialize_pad1d(model, simd)
    model = prepare_pad1d_stitched_ip_model(model, run_pnr=True)
    model.set_metadata_prop("exec_mode", "rtlsim")

    y_produced = oxe.execute_onnx(model, input_dict)["out"].reshape(y_expected.shape)
    assert (y_produced == y_expected).all(), "stitched_rtlsim failed"

    vivado_stitch_proj = model.get_metadata_prop("vivado_stitch_proj")
    ret = parse_ooc_synth_results(vivado_stitch_proj)
    assert ret is not None
    assert ret["LUT"] > 0
    assert ret["FF"] > 0
    assert ret["DSP"] == 0
    assert ret["BRAM_18K"] == 0
    assert ret["BRAM_36K"] == 0
    assert ret["WNS"] >= 0


def make_pad1d_hw_modelwrapper(num_tokens, num_channels, simd, pad_left, pad_right, finn_dtype):
    """A single Pad1D node with constant pad tokens, for schedule characterization."""
    ishape = [1, num_tokens, num_channels]
    oshape = [1, num_tokens + pad_left + pad_right, num_channels]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, ishape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, oshape)

    initializers = []
    node_inputs = ["inp"]
    if pad_left > 0:
        left = np.ones((1, pad_left, num_channels), dtype=np.float32)
        initializers.append(numpy_helper.from_array(left, name="left_pad"))
        node_inputs.append("left_pad")
    else:
        node_inputs.append("")
    if pad_right > 0:
        right = np.full((1, pad_right, num_channels), 2, dtype=np.float32)
        initializers.append(numpy_helper.from_array(right, name="right_pad"))
        node_inputs.append("right_pad")

    node = helper.make_node(
        "Pad1D",
        node_inputs,
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        NumTokens=num_tokens,
        NumChannels=num_channels,
        PadLeft=pad_left,
        PadRight=pad_right,
        SIMD=simd,
        inputDataType=finn_dtype.name,
        outputDataType=finn_dtype.name,
    )
    graph = helper.make_graph([node], "pad1d_hw", [inp], [outp], initializer=initializers)
    model = ModelWrapper(qonnx_make_model(graph, producer_name="pad1d-hw-model"))
    model.set_tensor_datatype("inp", finn_dtype)
    model.set_tensor_datatype("outp", finn_dtype)
    for init in initializers:
        model.set_tensor_datatype(init.name, finn_dtype)
    return model


# NumTokens, NumChannels, SIMD, PadLeft, PadRight
@pytest.mark.parametrize(
    "config",
    [
        (3, 4, 1, 2, 3),
        (8, 8, 2, 1, 0),
        (5, 16, 16, 3, 2),
        (12, 6, 3, 0, 4),
        (6, 4, 4, 0, 0),
        # a frame of several thousand folded words, so that the fixed part of the
        # error is visibly fixed rather than hidden by a short frame
        (256, 16, 1, 2, 2),
    ],
)
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.node_tree_modeling
def test_fpgadataflow_analytical_characterization_pad1d(config):
    num_tokens, num_channels, simd, pad_left, pad_right = config
    part = "xczu7ev-ffvc1156-2-e"
    target_clk_ns = 5.0
    model = make_pad1d_hw_modelwrapper(
        num_tokens, num_channels, simd, pad_left, pad_right, DataType["INT8"]
    )

    # The schedule is exact: same period, same token counts, same phase. What is
    # left is the output register between reading a word and writing it, worth at
    # most one token of cumulative divergence. That is a fixed cost, not a
    # proportional one -- the last config's frame is two orders of magnitude
    # longer than the first's and diverges by no more -- so the budget is a
    # single-digit floor with a fraction that only matters if the error ever
    # starts scaling.
    max_allowed_volume_frac = 0.005
    volume_const = 2
    max_allowed_length_frac = 0.005
    length_const = 2

    assert tree_model_test(
        model,
        ("Pad1D", config),
        part,
        target_clk_ns,
        max_allowed_volume_frac,
        max_allowed_length_frac,
        volume_const,
        length_const,
    ), "characterized TAV does not match RTLsim'd one!"
