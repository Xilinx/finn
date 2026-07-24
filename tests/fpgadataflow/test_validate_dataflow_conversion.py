# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for dataflow conversion validation analysis pass."""

import pytest

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

from finn.analysis.fpgadataflow.validate_dataflow_conversion import (
    validate_dataflow_conversion,
)
from finn.transformation.fpgadataflow.convert_to_hw_layers import (
    InferElementwiseBinaryOperation,
    InferHWSoftmax,
    InferQuantizedMatrixVectorActivation,
    InferThresholdingLayer,
)
from finn.util.fpgadataflow import is_fpgadataflow_node


def make_test_model():
    """Create a small model with different layer types for testing validation.

    Model structure (all non-fpgadataflow initially):
    - Layer 0: Transpose
    - Layer 1: MatMul (INT4 weights)
    - Layer 2: MultiThreshold
    - Layer 3: Mul
    - Layer 4: Softmax
    """
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 4, 4])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4, 4])

    # Layer 0: Transpose
    node0 = helper.make_node("Transpose", ["inp"], ["t0"], perm=[0, 1, 2])

    # Layer 1: MatMul with INT4 weights
    W1_data = gen_finn_dt_tensor(DataType["INT4"], (4, 4))
    W1 = helper.make_tensor("W1", TensorProto.FLOAT, [4, 4], W1_data.flatten().tolist())
    node1 = helper.make_node("MatMul", ["t0", "W1"], ["t1"])

    # Layer 2: MultiThreshold (QONNX custom op)
    # UINT4 has 16 values (0-15), so we need 15 thresholds per channel
    T2_data = gen_finn_dt_tensor(DataType["INT16"], (4, 15))
    T2_data = np.sort(T2_data, axis=1)  # Sort thresholds in increasing order
    T2 = helper.make_tensor("T2", TensorProto.FLOAT, [4, 15], T2_data.flatten().tolist())
    node2 = helper.make_node(
        "MultiThreshold",
        ["t1", "T2"],
        ["t2"],
        domain="qonnx.custom_op.general",
        out_dtype="UINT4",
        data_layout="NHWC",
    )

    # Layer 3: Mul
    scale_data = np.array([2.0], dtype=np.float32)
    scale = helper.make_tensor("scale", TensorProto.FLOAT, [1], scale_data.tolist())
    node3 = helper.make_node("Mul", ["t2", "scale"], ["t3"])

    # Layer 4: Softmax
    node4 = helper.make_node("Softmax", ["t3"], ["out"], axis=-1)

    graph = helper.make_graph(
        [node0, node1, node2, node3, node4],
        "test_validation",
        [inp],
        [out],
        initializer=[W1, T2, scale],
    )

    model = qonnx_make_model(graph)
    model = ModelWrapper(model)

    # Set INT4 datatypes
    model.set_tensor_datatype("inp", DataType["INT4"])
    model.set_tensor_datatype("W1", DataType["INT4"])
    model.set_tensor_datatype("T2", DataType["INT16"])

    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    return model


@pytest.mark.fpgadataflow
def test_validate_dataflow_conversion_scenarios():
    """Test validation through progressive conversion scenarios.

    Test plan:
    0. No conversions - should fail
    1. Convert layer 2 (MultiThreshold) → [non, non, fpga, non, non] - should pass
    2. Convert layer 4 (Softmax) → [non, non, fpga, non, fpga] - should FAIL (non-contiguous)
    3. Convert layer 3 and 1 (Mul and MatMul) → [non, fpga, fpga, fpga, fpga] - should pass
    """

    # Scenario 0: No fpgadataflow layers - should fail
    print("\n--- Scenario 0: No fpgadataflow layers ---")
    model = make_test_model()
    result = model.analysis(validate_dataflow_conversion)
    print(f"Valid: {result['valid']}")
    print(f"Message: {result['message']}")

    assert result["valid"] is False, "Expected validation to fail with no fpgadataflow layers"
    assert "No fpgadataflow layers found" in result["message"]
    assert len(result["unconverted_layers"]) == 5

    # Scenario 1: Convert layer 2 (MultiThreshold) → [non, non, fpga, non, non]
    print("\n--- Scenario 1: Convert layer 2 (MultiThreshold) ---")
    model = model.transform(InferThresholdingLayer())
    result = model.analysis(validate_dataflow_conversion)
    print(f"Valid: {result['valid']}")
    print(f"Message: {result['message']}")

    assert result["valid"] is True
    assert "contiguous block" in result["message"].lower()
    assert result["dataflow_block"] == (2, 2)

    # Scenario 2: Convert layer 4 (Softmax) → [non, non, fpga, non, fpga] - should FAIL
    print("\n--- Scenario 2: Convert layer 4 (Softmax) - EXPECT FAILURE ---")
    model = model.transform(InferHWSoftmax())
    result = model.analysis(validate_dataflow_conversion)
    print(f"Valid: {result['valid']}")
    print(f"Message: {result['message']}")

    assert (
        result["valid"] is False
    ), "Expected validation to fail with non-contiguous dataflow block"
    assert "Non-contiguous dataflow block detected" in result["message"]

    # Scenario 3: Convert layer 3 (Mul) and layer 1 (MatMul) → [non, fpga, fpga, fpga, fpga]
    print("\n--- Scenario 3: Convert layers 3 (Mul) and 1 (MatMul) ---")
    model = model.transform(InferElementwiseBinaryOperation())
    model = model.transform(InferQuantizedMatrixVectorActivation())
    result = model.analysis(validate_dataflow_conversion)
    print(f"Valid: {result['valid']}")
    print(f"Message: {result['message']}")

    assert result["valid"] is True
    assert "contiguous block" in result["message"].lower()
    assert result["dataflow_block"] == (1, 4)

    # Final verification
    print("\n--- Final verification ---")
    nodes = model.graph.node
    fpgadataflow_count = sum(1 for node in nodes if is_fpgadataflow_node(node))
    print(f"Fpgadataflow layers: {fpgadataflow_count} / {len(nodes)}")
    print(f"Total nodes: {len(nodes)}")

    # 4 out of 5 layers should be fpgadataflow (all except Transpose)
    assert fpgadataflow_count == 4
    assert result["valid"] is True
    assert len(result["unconverted_layers"]) == 1  # Only Transpose unconverted
