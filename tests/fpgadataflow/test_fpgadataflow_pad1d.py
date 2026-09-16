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
from finn import xsi as finnxsi
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
from finn.util.data_packing import npy_to_rtlsim_input
from finn.util.vivado import parse_ooc_synth_results

FPGA_PART = "xc7z020clg400-1"
CLK_NS = 10
PATCHES = np.arange(12, dtype=np.float32).reshape(1, 3, 4)


def make_pad1d_modelwrapper(
    pad_left,
    pad_right,
    pad_tokens,
    finn_dtype,
    patch_shape=(1, 3, 4),
):
    left_pad_token, right_pad_token = pad_tokens
    patch_shape = list(patch_shape)
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


def assert_rtlsim_quiescent_after_frame(inst):
    """A left pad must not announce another frame while the input is idle."""
    code_gen_dir = inst.get_nodeattr("code_gen_dir_ipgen")
    packed_input = npy_to_rtlsim_input(
        code_gen_dir + "/input_0.npy",
        inst.get_input_datatype(),
        inst.get_instream_width(),
    )
    io_dict = {"inputs": {"in0": packed_input}, "outputs": {"out0": []}}

    sim = inst.get_rtlsim()
    try:
        inst.reset_rtlsim(sim)
        inst.rtlsim_multi_io(sim, io_dict)
        output_valid = sim.get_bus_port("out0_V", "tvalid")

        class IdleProbe:
            def __init__(self, cycles):
                self.remaining = cycles
                self.saw_valid = False

            def __bool__(self):
                return True

            def __call__(self, _sim):
                self.saw_valid |= output_valid.read().as_bool()
                self.remaining -= 1
                return None if self.remaining == 0 else {}

        probe = IdleProbe(8)
        sim.enlist(probe)
        sim.run()
        assert not probe.saw_valid
    finally:
        finnxsi.close_rtlsim(sim)


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
        inst = getCustomOp(node)
        cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
        exp_cycles = model.analysis(exp_cycles_per_layer)[node.name]
        assert np.isclose(exp_cycles, cycles_rtlsim, atol=10)
        assert exp_cycles != 0
        assert_rtlsim_quiescent_after_frame(inst)


@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_pad1d_tinydeit_shape_rtlsim():
    """Exercise TinyDeiT's exact large FLOAT32 class-token insertion."""
    patches = np.arange(196 * 192, dtype=np.float32).reshape(1, 196, 192)
    cls_token = np.arange(192, dtype=np.float32).reshape(1, 1, 192)
    unused_right = np.zeros((1, 1, 192), dtype=np.float32)
    model = make_pad1d_modelwrapper(
        1,
        0,
        (cls_token, unused_right),
        DataType["FLOAT32"],
        patch_shape=patches.shape,
    )
    expected = np.concatenate((cls_token, patches), axis=1)
    model = infer_and_specialize_pad1d(model, simd=3)
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(FPGA_PART, CLK_NS))
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareRTLSim())

    produced = oxe.execute_onnx(model, {"patches": patches})["out"]
    assert np.array_equal(produced.reshape(expected.shape), expected)
    inst = getCustomOp(model.get_nodes_by_op_type("Pad1D_rtl")[0])
    assert_rtlsim_quiescent_after_frame(inst)


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
