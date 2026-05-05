# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import numpy as np
from onnx import TensorProto
from onnx import helper as oh
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

from finn.core.onnx_exec import execute_onnx
from finn.transformation.fpgadataflow.convert_to_hw_layers import (
    InferElementwiseBinaryOperation,
)
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.minimize_accumulator_width import (
    MinimizeAccumulatorWidth,
)
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.set_fifo_depths import InsertAndSetFIFODepths
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

# Versal part for RTL elementwise support
VERSAL_PART = "xcvc1902-vsva2197-2MP-e-S"

# RTL operations and their numpy references
NUMPY_REFERENCES = {
    "ElementwiseAdd": np.add,
    "ElementwiseMul": np.multiply,
}


def create_elementwise_model(op_type, lhs_dtype, rhs_dtype, lhs_shape, rhs_shape, out_dtype=None):
    """Create an ONNX model with a single elementwise binary operation.

    If out_dtype is not specified, FLOAT32 is used as placeholder and
    MinimizeAccumulatorWidth should be used to derive the correct output dtype.
    """
    onnx_op_type = op_type[11:]  # Remove "Elementwise" prefix
    out_shape = np.broadcast_shapes(lhs_shape, rhs_shape)
    if out_dtype is None:
        out_dtype = "FLOAT32"

    node = oh.make_node(
        op_type=onnx_op_type,
        inputs=["in_x", "in_y"],
        outputs=["out"],
    )

    lhs = oh.make_tensor_value_info("in_x", TensorProto.FLOAT, lhs_shape)
    rhs = oh.make_tensor_value_info("in_y", TensorProto.FLOAT, rhs_shape)
    out = oh.make_tensor_value_info("out", TensorProto.FLOAT, out_shape)

    graph = oh.make_graph([node], inputs=[lhs, rhs], outputs=[out], name="elementwise-binary")
    model = ModelWrapper(qonnx_make_model(graph, producer_name="elementwise-binary"))

    model.set_tensor_datatype("in_x", DataType[lhs_dtype])
    model.set_tensor_datatype("in_y", DataType[rhs_dtype])
    model.set_tensor_datatype("out", DataType[out_dtype])

    return model


# =============================================================================
# Main RTL Simulation Test
# =============================================================================


@pytest.mark.parametrize("op_type", ["ElementwiseAdd", "ElementwiseMul"])
@pytest.mark.parametrize(
    "lhs_dtype,rhs_dtype",
    [
        ("FLOAT32", "FLOAT32"),
        ("INT8", "INT8"),
        ("UINT8", "UINT8"),
        ("INT16", "INT16"),
        ("INT8", "FLOAT32"),
    ],
)
@pytest.mark.parametrize(
    "lhs_shape,rhs_shape,rhs_is_const,pe",
    [
        # Simple shapes with const rhs
        ([1, 4], [1, 4], True, 1),
        ([1, 4], [1, 4], True, 2),
        # Simple shapes with both dynamic inputs
        ([1, 4], [1, 4], False, 1),
        ([1, 4], [1, 4], False, 2),
        # Broadcast constant - PE=1,4,8
        ([128, 384], [384], True, 1),
        ([128, 384], [384], True, 4),
        ([128, 384], [384], True, 8),
    ],
)
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_elementwise_rtl(op_type, lhs_dtype, rhs_dtype, lhs_shape, rhs_shape, rhs_is_const, pe):
    """Test RTL elementwise operations across various configurations."""

    # Check PE divides last dimension
    if lhs_shape[-1] % pe != 0:
        pytest.skip(f"PE ({pe}) must divide last dimension ({lhs_shape[-1]})")

    # Skip input/input mode for broadcast shapes (not supported)
    if not rhs_is_const and lhs_shape != rhs_shape:
        pytest.skip("input/input mode requires matching shapes")

    # Skip input/input for non-float (would need matching dtypes check in RTL)
    if not rhs_is_const and lhs_dtype != rhs_dtype:
        pytest.skip("input/input mode requires matching dtypes")

    model = create_elementwise_model(op_type, lhs_dtype, rhs_dtype, lhs_shape, rhs_shape)

    lhs_data = gen_finn_dt_tensor(DataType[lhs_dtype], lhs_shape)
    rhs_data = gen_finn_dt_tensor(DataType[rhs_dtype], rhs_shape)

    if rhs_is_const:
        model.set_initializer("in_y", rhs_data)
        context = {"in_x": lhs_data}
    else:
        context = {"in_x": lhs_data, "in_y": rhs_data}

    # Transform pipeline
    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(InferElementwiseBinaryOperation())

    node_inst = getCustomOp(model.graph.node[0])
    node_inst.set_nodeattr("PE", pe)

    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(SpecializeLayers(VERSAL_PART))

    # Derive output dtype
    model = model.transform(MinimizeAccumulatorWidth())

    # Verify RTL backend was selected
    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == f"{op_type}_rtl"

    # Run RTL simulation
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(VERSAL_PART, 10))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim(behav=True))

    o_produced = execute_onnx(model, context)["out"]

    # Compute expected output
    lhs_ref = (
        lhs_data.astype(np.float32)
        if lhs_dtype.startswith("INT") and rhs_dtype == "FLOAT32"
        else lhs_data
    )
    o_expected = NUMPY_REFERENCES[op_type](lhs_ref, rhs_data)

    # Compare results
    out_dtype = model.get_tensor_datatype("out")
    if out_dtype == DataType["FLOAT32"]:
        assert np.allclose(o_produced, o_expected, rtol=1e-5, atol=1e-6)
    else:
        assert np.all(o_produced == o_expected)


# =============================================================================
# Stitched IP RTL Simulation Test (tests memstream wrapper)
# =============================================================================


@pytest.mark.parametrize("op_type", ["ElementwiseAdd", "ElementwiseMul"])
@pytest.mark.parametrize("pe", [1, 8, 64])  # min, middle, max
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_elementwise_rtl_stitched_ip(op_type, pe):
    """Test RTL elementwise with stitched IP to verify memstream wrapper.

    The memstream wrapper is only inserted during CreateStitchedIP, so this test
    verifies that broadcast constants work correctly with the full stitched flow.
    """
    lhs_dtype = "FLOAT32"
    rhs_dtype = "FLOAT32"
    lhs_shape = [32, 64]
    rhs_shape = [64]  # Broadcast constant

    model = create_elementwise_model(op_type, lhs_dtype, rhs_dtype, lhs_shape, rhs_shape)

    lhs_data = gen_finn_dt_tensor(DataType[lhs_dtype], lhs_shape)
    rhs_data = gen_finn_dt_tensor(DataType[rhs_dtype], rhs_shape)

    model.set_initializer("in_y", rhs_data)

    # Transform pipeline
    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(InferElementwiseBinaryOperation())

    node_inst = getCustomOp(model.graph.node[0])
    node_inst.set_nodeattr("PE", pe)

    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(SpecializeLayers(VERSAL_PART))
    model = model.transform(MinimizeAccumulatorWidth())

    assert model.graph.node[0].op_type == f"{op_type}_rtl"

    # Prepare for stitched IP rtlsim
    model = model.transform(InsertAndSetFIFODepths(VERSAL_PART, 10))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(VERSAL_PART, 10))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(VERSAL_PART, 10, vitis=False))

    # Run stitched IP rtlsim
    model.set_metadata_prop("exec_mode", "rtlsim")
    o_produced = execute_onnx(model, {model.graph.input[0].name: lhs_data})[
        model.graph.output[0].name
    ]

    o_expected = NUMPY_REFERENCES[op_type](lhs_data, rhs_data)

    assert np.allclose(o_produced, o_expected, rtol=1e-5, atol=1e-6)


# =============================================================================
# Fallback to HLS Tests (no Vivado required)
# =============================================================================


@pytest.mark.parametrize(
    "scenario,op_type,lhs_dtype,rhs_dtype,out_dtype,expected_backend",
    [
        # INT25 signed MUL exceeds DSP58 capacity (max 24-bit for signed)
        ("wide_mul_fallback", "ElementwiseMul", "INT25", "INT25", "INT50", "hls"),
        # Mismatched bitwidths -> fallback to HLS
        ("mismatched_width_fallback", "ElementwiseAdd", "INT8", "INT16", "INT32", "hls"),
        # Mismatched signedness -> fallback to HLS
        ("mismatched_sign_fallback", "ElementwiseAdd", "INT8", "UINT8", "INT32", "hls"),
        # INT24 signed MUL within DSP58 capacity -> should use RTL
        ("int24_mul_rtl", "ElementwiseMul", "INT24", "INT24", "INT48", "rtl"),
    ],
)
@pytest.mark.fpgadataflow
def test_elementwise_rtl_backend_selection(
    scenario, op_type, lhs_dtype, rhs_dtype, out_dtype, expected_backend
):
    """Test that SpecializeLayers correctly routes operations to RTL or HLS."""
    lhs_shape = [1, 4]
    rhs_shape = [1, 4]

    model = create_elementwise_model(
        op_type, lhs_dtype, rhs_dtype, lhs_shape, rhs_shape, out_dtype=out_dtype
    )

    rhs_data = gen_finn_dt_tensor(DataType[rhs_dtype], rhs_shape)
    model.set_initializer("in_y", rhs_data)

    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(InferElementwiseBinaryOperation())

    node_inst = getCustomOp(model.graph.node[0])
    node_inst.set_nodeattr("out_dtype", out_dtype)

    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(SpecializeLayers(VERSAL_PART))

    assert len(model.graph.node) == 1
    assert (
        model.graph.node[0].op_type == f"{op_type}_{expected_backend}"
    ), f"Scenario '{scenario}': expected {expected_backend}, got {model.graph.node[0].op_type}"
