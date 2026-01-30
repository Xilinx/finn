#!/usr/bin/env python
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Simple test script for ElementwisePow operation."""

import numpy as np
from onnx import TensorProto, helper as oh
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model

from finn.core.onnx_exec import execute_onnx
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferElementwiseFunctionOperation


def create_pow_model(input_shape, exponent=2.0, dtype="FLOAT32"):
    """Create an ONNX model with a Pow node. Only FLOAT32 is supported."""
    assert dtype == "FLOAT32", "ElementwisePow only supports FLOAT32"
    inp = oh.make_tensor_value_info("inp", TensorProto.FLOAT, input_shape)
    outp = oh.make_tensor_value_info("outp", TensorProto.FLOAT, input_shape)
    
    # Create constant exponent tensor
    exp_tensor = oh.make_tensor(
        name="exponent",
        data_type=TensorProto.FLOAT,
        dims=[1],
        vals=[exponent]
    )
    
    pow_node = oh.make_node("Pow", ["inp", "exponent"], ["outp"])
    
    graph = oh.make_graph(
        nodes=[pow_node],
        name="pow-test",
        inputs=[inp],
        outputs=[outp],
        initializer=[exp_tensor]
    )
    
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_tensor_datatype("inp", DataType[dtype])
    model.set_tensor_datatype("outp", DataType[dtype])
    
    return model


def test_elementwise_pow():
    """Test ElementwisePow operation."""
    
    print("Creating ONNX model with Pow operation (exponent=2)...")
    
    # Create model with Pow node
    input_shape = [1, 4]
    exponent = 2.0
    model = create_pow_model(input_shape, exponent)
    
    # Test data
    test_values = np.array([[1.0, 2.0, 3.0, -2.0]], dtype=np.float32)
    context = {"inp": test_values}
    
    print(f"Test input values: {test_values}")
    print(f"Exponent: {exponent}")
    
    # Execute original model
    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    original_output = execute_onnx(model, context)["outp"]
    
    # Apply transformation to convert to ElementwisePow
    print("\nApplying InferElementwiseFunctionOperation transformation...")
    model = model.transform(InferElementwiseFunctionOperation())
    
    # Check transformation
    print("\nNodes after transformation:")
    for i, node in enumerate(model.graph.node):
        print(f"  {i}: {node.op_type} - domain: {node.domain}")
        if node.op_type == "ElementwisePow":
            print("     Attributes:")
            for j, attr in enumerate(node.attribute):
                print(f"       {j}: {attr.name} = {attr}")
            # Try to get custom op and check nodeattr
            from qonnx.custom_op.registry import getCustomOp
            custom_op = getCustomOp(node)
            print(f"     Custom op exponent: {custom_op.get_nodeattr('exponent')}")
    
    # Execute transformed model
    print("\nExecuting transformed model...")
    transformed_output = execute_onnx(model, context)["outp"]
    
    # Reference calculation
    reference_output = np.power(test_values, exponent)
    
    # Display results
    print("\nResults:")
    print(f"Original Pow output:      {original_output}")
    print(f"ElementwisePow output:    {transformed_output}")
    print(f"NumPy reference output:   {reference_output}")
    
    # Verify correctness
    original_match = np.allclose(original_output, reference_output, rtol=1e-5)
    transformed_match = np.allclose(transformed_output, reference_output, rtol=1e-5)
    
    print(f"\nOriginal matches reference: {original_match}")
    print(f"Transformed matches reference: {transformed_match}")
    
    if transformed_match:
        print("\n✓ ElementwisePow test PASSED!")
    else:
        print("\n✗ ElementwisePow test FAILED!")
        print(f"Max error: {np.max(np.abs(transformed_output - reference_output))}")
    
    # Test with different exponents
    print("\n--- Testing with different exponents ---")
    for exp in [0.5, 1.0, 3.0]:
        model_exp = create_pow_model(input_shape, exp)
        model_exp = model_exp.transform(InferDataTypes())
        model_exp = model_exp.transform(InferShapes())
        model_exp = model_exp.transform(InferElementwiseFunctionOperation())
        
        output = execute_onnx(model_exp, context)["outp"]
        expected = np.power(test_values, exp)
        
        match = np.allclose(output, expected, rtol=1e-5)
        print(f"Exponent {exp}: {'PASS' if match else 'FAIL'}")
    
    # Save model
    model.save("pow_elementwise.onnx")
    print("\nModel saved to 'pow_elementwise.onnx'")
    
    return transformed_match


if __name__ == "__main__":
    test_elementwise_pow()