############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright for portions of this file is held by AMD and Microsoft under
# MIT license as part of project Brainsmith.
# All other copyright is held by AMD and is provided under BSD-3-Clause license.
#
# Note: This test was originally written by Josh Monson and was adjusted.
#
############################################################################

import pytest

import numpy as np
from onnx import TensorProto, helper, numpy_helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferCrop
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

FPGA_PART = "xc7z020clg400-1"
CLK_NS = 10


def make_crop_modelwrapper(ishape, axis, indices, idt):
    indices = np.asarray(indices, dtype=np.int64)
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, ishape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, None)
    gather = helper.make_node("Gather", ["inp", "indices"], ["outp"], axis=axis)
    graph = helper.make_graph(
        [gather],
        "crop-model",
        [inp],
        [outp],
        [numpy_helper.from_array(indices, name="indices")],
    )
    model = ModelWrapper(
        qonnx_make_model(graph, producer_name="crop-model"),
        fix_missing_initializer_valueinfo=True,
    )
    model = model.transform(InferShapes())
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", idt)
    return model


def prepare_crop_stitched_ip_model(model):
    model = model.transform(InsertFIFO(create_shallow_fifos=True))
    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(FPGA_PART, CLK_NS))
    model = model.transform(HLSSynthIP())
    return model.transform(CreateStitchedIP(FPGA_PART, CLK_NS))


# Input shape, Gather axis/indices, and SIMD are coupled to cover both spatial
# axes, all channel-folding extremes, and the special 2D/4D representations.
@pytest.mark.parametrize(
    "config",
    [
        pytest.param(([3, 4, 4], 0, [1, 2], 1), id="3d-height-simd1"),
        pytest.param(([3, 4, 4], 1, [1, 2], 2), id="3d-width-simd2"),
        pytest.param(([4, 4], 0, [0, 1, 2], 4), id="2d-width-simd4"),
        pytest.param(([1, 3, 4, 4], 2, [1, 2], 4), id="4d-width-simd4"),
    ],
)
@pytest.mark.parametrize(
    "impl_style,idt,exec_mode",
    [
        pytest.param("hls", DataType["INT8"], "cppsim", id="hls-INT8-cppsim"),
        pytest.param("hls", DataType["INT8"], "rtlsim", id="hls-INT8-rtlsim"),
        pytest.param("hls", DataType["FLOAT32"], "cppsim", id="hls-FLOAT32-cppsim"),
        pytest.param("hls", DataType["FLOAT32"], "rtlsim", id="hls-FLOAT32-rtlsim"),
        pytest.param("rtl", DataType["INT8"], "rtlsim", id="rtl-INT8-rtlsim"),
        pytest.param("rtl", DataType["INT8"], "stitched_rtlsim", id="rtl-INT8-stitched-rtlsim"),
        pytest.param("rtl", DataType["UINT4"], "rtlsim", id="rtl-UINT4-rtlsim"),
        pytest.param("rtl", DataType["UINT4"], "stitched_rtlsim", id="rtl-UINT4-stitched-rtlsim"),
        pytest.param("rtl", DataType["INT6"], "rtlsim", id="rtl-INT6-rtlsim"),
        pytest.param("rtl", DataType["INT6"], "stitched_rtlsim", id="rtl-INT6-stitched-rtlsim"),
    ],
)
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_fpgadataflow_crop(config, idt, impl_style, exec_mode):
    ishape, axis, indices, simd = config
    # Stitched execution reserves the leading dimension for batch size. Promote
    # unbatched Crop inputs while preserving the spatial axis under test.
    if exec_mode == "stitched_rtlsim" and len(ishape) < 4:
        ishape = [1] + ishape
        axis += 1
    input_tensor = gen_finn_dt_tensor(idt, ishape)
    input_dict = {"inp": input_tensor}
    y_expected = np.take(input_tensor, indices, axis=axis)
    model = make_crop_modelwrapper(ishape, axis, indices, idt)

    # Golden reference from the original Gather graph.
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    assert np.array_equal(y_produced, y_expected), "Execution of Gather model failed"

    model = model.transform(InferCrop())
    crop_nodes = model.get_nodes_by_op_type("Crop")
    assert len(crop_nodes) == 1
    crop = getCustomOp(crop_nodes[0])
    crop.set_nodeattr("preferred_impl_style", impl_style)
    crop.set_nodeattr("SIMD", simd)

    # Check the inferred hardware-independent node before specialization.
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    assert np.array_equal(y_produced, y_expected), "Execution of inferred Crop failed"

    model = model.transform(SpecializeLayers(FPGA_PART))
    expected_op_type = f"Crop_{impl_style}"
    assert len(model.get_nodes_by_op_type(expected_op_type)) == 1

    if exec_mode == "cppsim":
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))
    elif exec_mode == "rtlsim":
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(PrepareIP(FPGA_PART, CLK_NS))
        if impl_style == "hls":
            model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())
    elif exec_mode == "stitched_rtlsim":
        model = prepare_crop_stitched_ip_model(model)
        model.set_metadata_prop("exec_mode", "rtlsim")
    else:
        raise ValueError("Unknown exec_mode")

    y_produced = oxe.execute_onnx(model, input_dict)["outp"].reshape(y_expected.shape)
    assert np.array_equal(y_produced, y_expected), exec_mode + " failed"

    if exec_mode == "rtlsim":
        node = model.get_nodes_by_op_type(expected_op_type)[0]
        cycles_rtlsim = getCustomOp(node).get_nodeattr("cycles_rtlsim")
        exp_cycles = model.analysis(exp_cycles_per_layer)[node.name]
        assert np.isclose(exp_cycles, cycles_rtlsim, atol=15)
        assert exp_cycles != 0
