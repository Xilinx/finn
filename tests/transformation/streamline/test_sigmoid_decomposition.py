# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from onnx import TensorProto, helper as oh
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

from finn.core.onnx_exec import execute_onnx
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.streamline.sigmoid_decomposition import DecomposeSigmoid


def sigmoid_reference(x):
    """Reference sigmoid implementation."""
    return 1.0 / (1.0 + np.exp(-x))


def create_sigmoid_model(input_shape, dtype="FLOAT32"):
    """Create an ONNX model with a Sigmoid node."""
    inp = oh.make_tensor_value_info("inp", TensorProto.FLOAT, input_shape)
    outp = oh.make_tensor_value_info("outp", TensorProto.FLOAT, input_shape)
    
    sigmoid_node = oh.make_node("Sigmoid", ["inp"], ["outp"])
    
    graph = oh.make_graph(
        nodes=[sigmoid_node],
        name="sigmoid-test",
        inputs=[inp],
        outputs=[outp]
    )
    
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_tensor_datatype("inp", DataType[dtype])
    model.set_tensor_datatype("outp", DataType[dtype])
    
    return model


@pytest.mark.parametrize("input_shape", [(1, 4), (2, 8), (1, 3, 3)])
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
def test_sigmoid_decomposition(input_shape, exec_mode):
    """Test sigmoid decomposition transformation with RTLsim."""
    
    # Create model with Sigmoid node
    model = create_sigmoid_model(input_shape)
    
    # Generate test input data
    np.random.seed(42)
    test_input = np.random.uniform(-3, 3, input_shape).astype(np.float32)
    context = {"inp": test_input}
    
    # Compute reference output before decomposition
    reference_output = sigmoid_reference(test_input)
    
    # Apply sigmoid decomposition
    model = model.transform(DecomposeSigmoid())
    
    # Verify decomposition happened
    sigmoid_nodes = [n for n in model.graph.node if n.op_type == "Sigmoid"]
    assert len(sigmoid_nodes) == 0, "Sigmoid nodes should be decomposed"
    
    # Verify we have the expected decomposed operations
    op_types = [n.op_type for n in model.graph.node]
    assert "Mul" in op_types, "Should have Mul node for negation"
    assert "Exp" in op_types, "Should have Exp node"
    assert "Add" in op_types, "Should have Add node"
    assert "Div" in op_types, "Should have Div node"
    
    # Test basic execution first (no hardware)
    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    
    # Execute decomposed model in software
    sw_output = execute_onnx(model, context)["outp"]
    assert np.allclose(sw_output, reference_output, rtol=1e-5), \
        "Decomposed model output doesn't match reference"
    
    # Now test with hardware simulation if requested
    if exec_mode in ["cppsim", "rtlsim"]:
        # Convert to FINN-style operations
        model = model.transform(ConvertQONNXtoFINN())
        
        # Specialize layers for FPGA
        fpga_part = "xc7z020clg400-1"  # Zynq-7020 part
        model = model.transform(SpecializeLayers(fpga_part))
        
        # Set execution mode
        model = model.transform(SetExecMode(exec_mode))
        model = model.transform(GiveUniqueNodeNames())
        
        if exec_mode == "cppsim":
            model = model.transform(PrepareCppSim())
            model = model.transform(CompileCppSim())
        else:  # rtlsim
            model = model.transform(PrepareIP(fpga_part, 10))  # 10ns clock period
            model = model.transform(HLSSynthIP())
            model = model.transform(PrepareRTLSim())
        
        # Execute in hardware simulation
        hw_output = execute_onnx(model, context)["outp"]
        
        # Verify hardware output matches reference
        assert np.allclose(hw_output, reference_output, rtol=1e-4), \
            f"{exec_mode} output doesn't match reference"


@pytest.mark.parametrize("test_values", [
    np.array([0.0]),      # Test at zero
    np.array([1.0]),      # Test positive
    np.array([-1.0]),     # Test negative
    np.array([5.0]),      # Test large positive (should be ~1)
    np.array([-5.0]),     # Test large negative (should be ~0)
])
def test_sigmoid_decomposition_specific_values(test_values):
    """Test sigmoid decomposition with specific test values."""
    
    input_shape = test_values.shape
    model = create_sigmoid_model(input_shape)
    
    # Apply decomposition
    model = model.transform(DecomposeSigmoid())
    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    
    # Execute
    context = {"inp": test_values.astype(np.float32)}
    output = execute_onnx(model, context)["outp"]
    
    # Compare with reference
    expected = sigmoid_reference(test_values)
    assert np.allclose(output, expected, rtol=1e-5), \
        f"Failed for input {test_values}: got {output}, expected {expected}"


if __name__ == "__main__":
    # Run a simple test
    test_sigmoid_decomposition((1, 4), "cppsim")
    print("Sigmoid decomposition test passed!")