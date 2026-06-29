# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import numpy as np
import onnx
import onnx.helper as oh
from qonnx.core.modelwrapper import ModelWrapper

from finn.core.onnx_exec import execute_onnx


def make_simple_model(input_name="in_x", output_name="out_y"):
    """Create a simple ONNX model with an Identity node for testing."""
    inp = oh.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, [1, 4])
    outp = oh.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, [1, 4])

    identity_node = oh.make_node("Identity", [input_name], [output_name])

    graph = oh.make_graph([identity_node], "test_graph", [inp], [outp])
    onnx_model = oh.make_model(
        graph, producer_name="finn-test", opset_imports=[oh.make_opsetid("", 11)]
    )
    return ModelWrapper(onnx_model)


@pytest.mark.util
def test_execute_onnx_valid_input():
    """Test that execute_onnx works with valid input tensor names."""
    model = make_simple_model(input_name="in_x", output_name="out_y")
    input_data = np.random.randn(1, 4).astype(np.float32)

    # Valid input name should work
    result = execute_onnx(model, {"in_x": input_data})

    assert "out_y" in result
    assert np.allclose(result["out_y"], input_data)


@pytest.mark.util
def test_execute_onnx_invalid_input_name():
    """Test that execute_onnx raises ValueError for invalid input tensor names.

    This catches common bugs like using outdated tensor names after model
    transformations that rename tensors.
    """
    model = make_simple_model(input_name="in_x", output_name="out_y")
    input_data = np.random.randn(1, 4).astype(np.float32)

    # Invalid input name should raise ValueError with helpful message
    with pytest.raises(ValueError) as excinfo:
        execute_onnx(model, {"wrong_name": input_data})

    error_msg = str(excinfo.value)
    assert "wrong_name" in error_msg, "Error should mention the invalid input name"
    assert "in_x" in error_msg, "Error should list valid input names"


@pytest.mark.util
def test_execute_onnx_multiple_invalid_inputs():
    """Test that execute_onnx catches invalid names even when mixed with valid ones."""
    model = make_simple_model(input_name="in_x", output_name="out_y")
    input_data = np.random.randn(1, 4).astype(np.float32)

    # Mix of valid and invalid names - should fail on the invalid one
    with pytest.raises(ValueError) as excinfo:
        execute_onnx(model, {"in_x": input_data, "invalid_tensor": input_data})

    error_msg = str(excinfo.value)
    assert "invalid_tensor" in error_msg
