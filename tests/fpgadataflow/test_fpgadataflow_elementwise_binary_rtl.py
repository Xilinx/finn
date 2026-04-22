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
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

# RTL operations and their numpy references
RTL_NUMPY_REFERENCES = {
    "ElementwiseAdd": np.add,
    "ElementwiseSub": np.subtract,
    "ElementwiseMul": np.multiply,
}


def create_elementwise_binary_operation_onnx(
    op_type, lhs_dtype, rhs_dtype, out_dtype, lhs_shape, rhs_shape
):
    onnx_op_type = op_type[11:]  # Remove "Elementwise" prefix
    out_shape = np.broadcast_shapes(lhs_shape, rhs_shape)

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


@pytest.mark.parametrize("op_type", ["ElementwiseAdd", "ElementwiseSub", "ElementwiseMul"])
@pytest.mark.parametrize("pe", [1, 2])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_elementwise_binary_operation_rtl(op_type, pe):
    """Test RTL elementwise operations for FLOAT32 using RTL simulation."""

    lhs_dtype = "FLOAT32"
    rhs_dtype = "FLOAT32"
    out_dtype = "FLOAT32"
    lhs_shape = [1, 4]
    rhs_shape = [1, 4]

    model = create_elementwise_binary_operation_onnx(
        op_type, lhs_dtype, rhs_dtype, out_dtype, lhs_shape, rhs_shape
    )

    # Generate test data
    lhs_data = gen_finn_dt_tensor(DataType[lhs_dtype], lhs_shape)
    rhs_data = gen_finn_dt_tensor(DataType[rhs_dtype], rhs_shape)

    # Set the second input as an initializer (constant) for RTL constraints
    model.set_initializer("in_y", rhs_data)

    context = {
        "in_x": lhs_data,
    }

    numpy_reference = RTL_NUMPY_REFERENCES[op_type]

    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(InferElementwiseBinaryOperation())

    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == f"{op_type}"

    node_inst = getCustomOp(model.graph.node[0])
    node_inst.set_nodeattr("preferred_impl_style", "rtl")
    node_inst.set_nodeattr("PE", pe)
    node_inst.set_nodeattr("lhs_style", "input")  # dynamic data
    node_inst.set_nodeattr("rhs_style", "const")  # constant data
    node_inst.set_nodeattr("mem_mode", "internal_decoupled")

    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(SpecializeLayers("xcvc1902-vsva2197-2MP-e-S"))

    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == f"{op_type}_rtl"

    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(GiveUniqueNodeNames())

    model = model.transform(PrepareIP("xcvc1902-vsva2197-2MP-e-S", 10))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim(behav=True))

    lhs = context["in_x"]
    rhs = rhs_data  # Use the constant data we set as initializer
    o_expected = numpy_reference(lhs, rhs)
    o_produced = execute_onnx(model, context)["out"]

    assert np.all(o_produced == o_expected)


@pytest.mark.parametrize("op_type", ["ElementwiseAdd", "ElementwiseSub", "ElementwiseMul"])
@pytest.mark.parametrize("pe", [1, 4, 8])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_elementwise_binary_operation_rtl_with_memstream(op_type, pe):
    """Test RTL elementwise operations with memstream for broadcast constants.

    Dynamic input: [1, 384] - streamed during operation
    Constant input: [384] - stored in memstream, broadcast to match dynamic input
    """

    lhs_dtype = "FLOAT32"
    rhs_dtype = "FLOAT32"
    out_dtype = "FLOAT32"
    lhs_shape = [128, 384]  # Large dynamic input
    rhs_shape = [384]  # Broadcast constant

    model = create_elementwise_binary_operation_onnx(
        op_type, lhs_dtype, rhs_dtype, out_dtype, lhs_shape, rhs_shape
    )

    # Generate test data
    lhs_data = gen_finn_dt_tensor(DataType[lhs_dtype], lhs_shape)
    rhs_data = gen_finn_dt_tensor(DataType[rhs_dtype], rhs_shape)

    # Set the second input as an initializer (constant) for RTL constraints
    model.set_initializer("in_y", rhs_data)

    context = {
        "in_x": lhs_data,
    }

    numpy_reference = RTL_NUMPY_REFERENCES[op_type]

    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(InferElementwiseBinaryOperation())

    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == f"{op_type}"

    node_inst = getCustomOp(model.graph.node[0])
    node_inst.set_nodeattr("preferred_impl_style", "rtl")
    node_inst.set_nodeattr("PE", pe)
    node_inst.set_nodeattr("lhs_style", "input")  # dynamic data
    node_inst.set_nodeattr("rhs_style", "const")  # constant data stored in memstream
    node_inst.set_nodeattr("mem_mode", "internal_decoupled")

    # Verify PE divides into the last dimension
    assert lhs_shape[-1] % pe == 0, f"PE ({pe}) must divide last dimension ({lhs_shape[-1]})"

    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(SpecializeLayers("xcvc1902-vsva2197-2MP-e-S"))

    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == f"{op_type}_rtl"

    # Verify memstream parameters are set correctly
    node_inst_rtl = getCustomOp(model.graph.node[0])
    expected_wmem = node_inst_rtl.calc_wmem()
    print(f"Expected wmem for constant shape {rhs_shape} with PE={pe}: {expected_wmem}")

    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(GiveUniqueNodeNames())

    model = model.transform(PrepareIP("xcvc1902-vsva2197-2MP-e-S", 10))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim(behav=True))

    lhs = context["in_x"]
    rhs = rhs_data  # Use the constant data we set as initializer
    o_expected = numpy_reference(lhs, rhs)
    o_produced = execute_onnx(model, context)["out"]

    assert np.allclose(o_produced, o_expected, rtol=1e-5, atol=1e-6)


@pytest.mark.fpgadataflow
def test_elementwise_binary_operation_rtl_fallback_to_hls_wide_mul():
    """Test that int MUL exceeding DSP58 width falls back to HLS."""

    # INT25 signed MUL exceeds max width of 24 for signed
    lhs_dtype = "INT25"
    rhs_dtype = "INT25"
    out_dtype = "INT50"
    lhs_shape = [1, 4]
    rhs_shape = [1, 4]

    model = create_elementwise_binary_operation_onnx(
        "ElementwiseMul", lhs_dtype, rhs_dtype, out_dtype, lhs_shape, rhs_shape
    )

    # Set rhs as const to satisfy style requirements
    rhs_data = gen_finn_dt_tensor(DataType[rhs_dtype], rhs_shape)
    model.set_initializer("in_y", rhs_data)

    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(InferElementwiseBinaryOperation())

    node_inst = getCustomOp(model.graph.node[0])
    node_inst.set_nodeattr("preferred_impl_style", "rtl")
    node_inst.set_nodeattr("lhs_style", "input")
    node_inst.set_nodeattr("rhs_style", "const")
    # Override output dtype to match RTL expectation
    node_inst.set_nodeattr("out_dtype", out_dtype)

    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(SpecializeLayers("xcvc1902-vsva2197-2MP-e-S"))

    # Should fall back to HLS because INT25 MUL exceeds DSP58 capacity
    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == "ElementwiseMul_hls"


@pytest.mark.fpgadataflow
def test_elementwise_binary_operation_rtl_fallback_mismatched_width():
    """Test that int/int with mismatched widths falls back to HLS."""

    lhs_dtype = "INT8"
    rhs_dtype = "INT16"
    out_dtype = "INT32"
    lhs_shape = [1, 4]
    rhs_shape = [1, 4]

    model = create_elementwise_binary_operation_onnx(
        "ElementwiseAdd", lhs_dtype, rhs_dtype, out_dtype, lhs_shape, rhs_shape
    )

    rhs_data = gen_finn_dt_tensor(DataType[rhs_dtype], rhs_shape)
    model.set_initializer("in_y", rhs_data)

    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(InferElementwiseBinaryOperation())

    node_inst = getCustomOp(model.graph.node[0])
    node_inst.set_nodeattr("preferred_impl_style", "rtl")
    node_inst.set_nodeattr("lhs_style", "input")
    node_inst.set_nodeattr("rhs_style", "const")

    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(SpecializeLayers("xcvc1902-vsva2197-2MP-e-S"))

    # Should fall back to HLS because widths don't match
    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == "ElementwiseAdd_hls"


@pytest.mark.fpgadataflow
def test_elementwise_binary_operation_rtl_fallback_mismatched_sign():
    """Test that int/int with mismatched signedness falls back to HLS."""

    lhs_dtype = "INT8"
    rhs_dtype = "UINT8"
    out_dtype = "INT32"
    lhs_shape = [1, 4]
    rhs_shape = [1, 4]

    model = create_elementwise_binary_operation_onnx(
        "ElementwiseAdd", lhs_dtype, rhs_dtype, out_dtype, lhs_shape, rhs_shape
    )

    rhs_data = gen_finn_dt_tensor(DataType[rhs_dtype], rhs_shape)
    model.set_initializer("in_y", rhs_data)

    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(InferElementwiseBinaryOperation())

    node_inst = getCustomOp(model.graph.node[0])
    node_inst.set_nodeattr("preferred_impl_style", "rtl")
    node_inst.set_nodeattr("lhs_style", "input")
    node_inst.set_nodeattr("rhs_style", "const")

    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(SpecializeLayers("xcvc1902-vsva2197-2MP-e-S"))

    # Should fall back to HLS because signedness doesn't match
    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == "ElementwiseAdd_hls"


@pytest.mark.parametrize(
    "op_type,out_dtype",
    [
        ("ElementwiseAdd", "INT9"),
        ("ElementwiseSub", "INT9"),
        ("ElementwiseMul", "INT16"),
    ],
)
@pytest.mark.parametrize("pe", [1, 2])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_elementwise_binary_operation_rtl_int_signed(op_type, out_dtype, pe):
    """Test RTL elementwise operations for signed INT8 inputs."""

    lhs_dtype = "INT8"
    rhs_dtype = "INT8"
    lhs_shape = [1, 4]
    rhs_shape = [1, 4]

    model = create_elementwise_binary_operation_onnx(
        op_type, lhs_dtype, rhs_dtype, out_dtype, lhs_shape, rhs_shape
    )

    lhs_data = gen_finn_dt_tensor(DataType[lhs_dtype], lhs_shape)
    rhs_data = gen_finn_dt_tensor(DataType[rhs_dtype], rhs_shape)
    model.set_initializer("in_y", rhs_data)

    context = {"in_x": lhs_data}
    numpy_reference = RTL_NUMPY_REFERENCES[op_type]

    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(InferElementwiseBinaryOperation())

    assert len(model.graph.node) == 1
    node_inst = getCustomOp(model.graph.node[0])
    node_inst.set_nodeattr("preferred_impl_style", "rtl")
    node_inst.set_nodeattr("PE", pe)
    node_inst.set_nodeattr("lhs_style", "input")
    node_inst.set_nodeattr("rhs_style", "const")
    node_inst.set_nodeattr("mem_mode", "internal_decoupled")
    # Set output dtype to match RTL O_WIDTH
    node_inst.set_nodeattr("out_dtype", out_dtype)

    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(SpecializeLayers("xcvc1902-vsva2197-2MP-e-S"))

    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == f"{op_type}_rtl"

    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP("xcvc1902-vsva2197-2MP-e-S", 10))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim(behav=True))

    o_expected = numpy_reference(lhs_data, rhs_data)
    o_produced = execute_onnx(model, context)["out"]

    assert np.all(o_produced == o_expected)


@pytest.mark.parametrize(
    "op_type,out_dtype",
    [
        ("ElementwiseAdd", "UINT9"),
        ("ElementwiseSub", "INT9"),
        ("ElementwiseMul", "UINT16"),
    ],
)
@pytest.mark.parametrize("pe", [1, 2])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_elementwise_binary_operation_rtl_int_unsigned(op_type, out_dtype, pe):
    """Test RTL elementwise operations for unsigned UINT8 inputs."""

    lhs_dtype = "UINT8"
    rhs_dtype = "UINT8"
    lhs_shape = [1, 4]
    rhs_shape = [1, 4]

    model = create_elementwise_binary_operation_onnx(
        op_type, lhs_dtype, rhs_dtype, out_dtype, lhs_shape, rhs_shape
    )

    lhs_data = gen_finn_dt_tensor(DataType[lhs_dtype], lhs_shape)
    rhs_data = gen_finn_dt_tensor(DataType[rhs_dtype], rhs_shape)
    model.set_initializer("in_y", rhs_data)

    context = {"in_x": lhs_data}
    numpy_reference = RTL_NUMPY_REFERENCES[op_type]

    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(InferElementwiseBinaryOperation())

    assert len(model.graph.node) == 1
    node_inst = getCustomOp(model.graph.node[0])
    node_inst.set_nodeattr("preferred_impl_style", "rtl")
    node_inst.set_nodeattr("PE", pe)
    node_inst.set_nodeattr("lhs_style", "input")
    node_inst.set_nodeattr("rhs_style", "const")
    node_inst.set_nodeattr("mem_mode", "internal_decoupled")
    node_inst.set_nodeattr("out_dtype", out_dtype)

    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(SpecializeLayers("xcvc1902-vsva2197-2MP-e-S"))

    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == f"{op_type}_rtl"

    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP("xcvc1902-vsva2197-2MP-e-S", 10))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim(behav=True))

    o_expected = numpy_reference(lhs_data, rhs_data)
    o_produced = execute_onnx(model, context)["out"]

    assert np.all(o_produced == o_expected)


@pytest.mark.parametrize(
    "op_type,out_dtype",
    [
        ("ElementwiseAdd", "INT17"),
        ("ElementwiseMul", "INT32"),
    ],
)
@pytest.mark.parametrize("pe", [1, 2])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_elementwise_binary_operation_rtl_int16(op_type, out_dtype, pe):
    """Test RTL elementwise operations for INT16 inputs."""

    lhs_dtype = "INT16"
    rhs_dtype = "INT16"
    lhs_shape = [1, 4]
    rhs_shape = [1, 4]

    model = create_elementwise_binary_operation_onnx(
        op_type, lhs_dtype, rhs_dtype, out_dtype, lhs_shape, rhs_shape
    )

    lhs_data = gen_finn_dt_tensor(DataType[lhs_dtype], lhs_shape)
    rhs_data = gen_finn_dt_tensor(DataType[rhs_dtype], rhs_shape)
    model.set_initializer("in_y", rhs_data)

    context = {"in_x": lhs_data}
    numpy_reference = RTL_NUMPY_REFERENCES[op_type]

    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(InferElementwiseBinaryOperation())

    assert len(model.graph.node) == 1
    node_inst = getCustomOp(model.graph.node[0])
    node_inst.set_nodeattr("preferred_impl_style", "rtl")
    node_inst.set_nodeattr("PE", pe)
    node_inst.set_nodeattr("lhs_style", "input")
    node_inst.set_nodeattr("rhs_style", "const")
    node_inst.set_nodeattr("mem_mode", "internal_decoupled")
    node_inst.set_nodeattr("out_dtype", out_dtype)

    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(SpecializeLayers("xcvc1902-vsva2197-2MP-e-S"))

    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == f"{op_type}_rtl"

    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP("xcvc1902-vsva2197-2MP-e-S", 10))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim(behav=True))

    o_expected = numpy_reference(lhs_data, rhs_data)
    o_produced = execute_onnx(model, context)["out"]

    assert np.all(o_produced == o_expected)


@pytest.mark.parametrize("op_type", ["ElementwiseAdd", "ElementwiseMul"])
@pytest.mark.parametrize("pe", [1, 2])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_elementwise_binary_operation_rtl_mixed_int_float(op_type, pe):
    """Test RTL elementwise operations with mixed int/float inputs.

    LHS is INT8 (converted to fp32 internally), RHS is FLOAT32.
    Output is FLOAT32.
    """

    lhs_dtype = "INT8"
    rhs_dtype = "FLOAT32"
    out_dtype = "FLOAT32"
    lhs_shape = [1, 4]
    rhs_shape = [1, 4]

    model = create_elementwise_binary_operation_onnx(
        op_type, lhs_dtype, rhs_dtype, out_dtype, lhs_shape, rhs_shape
    )

    lhs_data = gen_finn_dt_tensor(DataType[lhs_dtype], lhs_shape)
    rhs_data = gen_finn_dt_tensor(DataType[rhs_dtype], rhs_shape)
    model.set_initializer("in_y", rhs_data)

    context = {"in_x": lhs_data}
    numpy_reference = RTL_NUMPY_REFERENCES[op_type]

    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(InferElementwiseBinaryOperation())

    assert len(model.graph.node) == 1
    node_inst = getCustomOp(model.graph.node[0])
    node_inst.set_nodeattr("preferred_impl_style", "rtl")
    node_inst.set_nodeattr("PE", pe)
    node_inst.set_nodeattr("lhs_style", "input")
    node_inst.set_nodeattr("rhs_style", "const")
    node_inst.set_nodeattr("mem_mode", "internal_decoupled")

    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(SpecializeLayers("xcvc1902-vsva2197-2MP-e-S"))

    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == f"{op_type}_rtl"

    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP("xcvc1902-vsva2197-2MP-e-S", 10))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim(behav=True))

    o_expected = numpy_reference(lhs_data.astype(np.float32), rhs_data)
    o_produced = execute_onnx(model, context)["out"]

    assert np.allclose(o_produced, o_expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("op_type", ["ElementwiseAdd", "ElementwiseSub"])
@pytest.mark.parametrize("pe", [1, 2])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_elementwise_binary_operation_rtl_input_input(op_type, pe):
    """Test RTL elementwise operations with both inputs dynamic (no const).

    Both LHS and RHS are streaming inputs, no memstream wrapper needed.
    """

    lhs_dtype = "FLOAT32"
    rhs_dtype = "FLOAT32"
    out_dtype = "FLOAT32"
    lhs_shape = [1, 4]
    rhs_shape = [1, 4]

    model = create_elementwise_binary_operation_onnx(
        op_type, lhs_dtype, rhs_dtype, out_dtype, lhs_shape, rhs_shape
    )

    lhs_data = gen_finn_dt_tensor(DataType[lhs_dtype], lhs_shape)
    rhs_data = gen_finn_dt_tensor(DataType[rhs_dtype], rhs_shape)

    # Do NOT set in_y as initializer — both are dynamic inputs
    context = {
        "in_x": lhs_data,
        "in_y": rhs_data,
    }

    numpy_reference = RTL_NUMPY_REFERENCES[op_type]

    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(InferElementwiseBinaryOperation())

    assert len(model.graph.node) == 1
    node_inst = getCustomOp(model.graph.node[0])
    node_inst.set_nodeattr("preferred_impl_style", "rtl")
    node_inst.set_nodeattr("PE", pe)
    node_inst.set_nodeattr("lhs_style", "input")
    node_inst.set_nodeattr("rhs_style", "input")

    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(SpecializeLayers("xcvc1902-vsva2197-2MP-e-S"))

    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == f"{op_type}_rtl"

    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP("xcvc1902-vsva2197-2MP-e-S", 10))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim(behav=True))

    o_expected = numpy_reference(lhs_data, rhs_data)
    o_produced = execute_onnx(model, context)["out"]

    assert np.all(o_produced == o_expected)


@pytest.mark.fpgadataflow
def test_elementwise_binary_operation_rtl_int24_signed_mul_passes():
    """Test that INT24 signed MUL (within DSP58 capacity) routes to RTL."""

    lhs_dtype = "INT24"
    rhs_dtype = "INT24"
    out_dtype = "INT48"
    lhs_shape = [1, 4]
    rhs_shape = [1, 4]

    model = create_elementwise_binary_operation_onnx(
        "ElementwiseMul", lhs_dtype, rhs_dtype, out_dtype, lhs_shape, rhs_shape
    )

    rhs_data = gen_finn_dt_tensor(DataType[rhs_dtype], rhs_shape)
    model.set_initializer("in_y", rhs_data)

    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(InferElementwiseBinaryOperation())

    node_inst = getCustomOp(model.graph.node[0])
    node_inst.set_nodeattr("preferred_impl_style", "rtl")
    node_inst.set_nodeattr("lhs_style", "input")
    node_inst.set_nodeattr("rhs_style", "const")
    node_inst.set_nodeattr("mem_mode", "internal_decoupled")
    node_inst.set_nodeattr("out_dtype", out_dtype)

    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(SpecializeLayers("xcvc1902-vsva2197-2MP-e-S"))

    # INT24 signed MUL should route to RTL (max_w=24 for signed)
    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == "ElementwiseMul_rtl"
