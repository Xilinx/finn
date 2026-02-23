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
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.convert_to_hw_layers import (
    InferElementwiseBinaryOperation,
)
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
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
    model = model.transform(SpecializeLayers("xcv80-lsva4737-2MHP-e-s"))

    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == f"{op_type}_rtl"

    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(GiveUniqueNodeNames())
    
    model = model.transform(PrepareIP("xcv80-lsva4737-2MHP-e-s", 10))
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
    rhs_shape = [384]       # Broadcast constant
    
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
    model = model.transform(SpecializeLayers("xcv80-lsva4737-2MHP-e-s"))

    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == f"{op_type}_rtl"
    
    # Verify memstream parameters are set correctly
    node_inst_rtl = getCustomOp(model.graph.node[0])
    expected_wmem = node_inst_rtl.calc_wmem()
    print(f"Expected wmem for constant shape {rhs_shape} with PE={pe}: {expected_wmem}")

    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(GiveUniqueNodeNames())
    
    model = model.transform(PrepareIP("xcv80-lsva4737-2MHP-e-s", 10))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim(behav=True))

    lhs = context["in_x"]
    rhs = rhs_data  # Use the constant data we set as initializer
    o_expected = numpy_reference(lhs, rhs)
    o_produced = execute_onnx(model, context)["out"]

    assert np.allclose(o_produced, o_expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("op_type", ["ElementwiseAdd", "ElementwiseSub", "ElementwiseMul"])
def test_elementwise_binary_operation_rtl_fallback_to_hls(op_type):
    """Test that non-FLOAT32 datatypes fall back to HLS."""
    
    lhs_dtype = "INT8"  # Non-FLOAT32 should fallback to HLS
    rhs_dtype = "INT8"
    out_dtype = "INT8"
    lhs_shape = [1, 4]
    rhs_shape = [1, 4]
    
    model = create_elementwise_binary_operation_onnx(
        op_type, lhs_dtype, rhs_dtype, out_dtype, lhs_shape, rhs_shape
    )

    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(InferElementwiseBinaryOperation())

    # Don't set preferred_impl_style - let automatic selection work
    # RTL should be tried first but constraints should force fallback to HLS

    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(SpecializeLayers("xcv80-lsva4737-2MHP-e-s"))

    # Should fall back to HLS for non-FLOAT32
    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == f"{op_type}_hls"
