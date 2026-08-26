############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright is held by AMD and is provided under BSD-3-Clause license.

import pytest

import numpy as np
import onnx.helper as oh
from onnx import TensorProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model

import finn.core.onnx_exec as ox
from finn.transformation.streamline.extract_multithreshold_scale_bias import (
    ExtractMultiThresholdScaleBias,
)


@pytest.mark.streamline
@pytest.mark.parametrize("extract_scale", [True, False])
@pytest.mark.parametrize("extract_bias", [True, False])
def test_extract_multithreshold_scale_bias(extract_scale, extract_bias):
    # Set scale and bias values based on parameters
    out_scale = 2.5 if extract_scale else 1.0
    out_bias = 1.0 if extract_bias else 0.0

    # Create simple MultiThreshold model
    inp = oh.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 64])
    thres = oh.make_tensor_value_info("thres", TensorProto.FLOAT, [64, 15])
    outp = oh.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 64])

    mt_node = oh.make_node(
        "MultiThreshold",
        ["inp", "thres"],
        ["outp"],
        domain="qonnx.custom_op.general",
        out_dtype="UINT4",
        out_scale=out_scale,
        out_bias=out_bias,
    )

    modelproto = qonnx_make_model(
        oh.make_graph(
            name="test",
            inputs=[inp],
            outputs=[outp],
            value_info=[thres],
            nodes=[mt_node],
        )
    )
    model = ModelWrapper(modelproto)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    # Set threshold values (15 thresholds per channel for UINT4)
    thres_values = np.random.randn(64, 15).astype(np.float32)
    model.set_initializer("thres", thres_values)

    # Apply transformation
    new_model = model.transform(ExtractMultiThresholdScaleBias())

    # Verify numerical correctness
    inp_dict = {"inp": np.random.randn(1, 64).astype(np.float32)}
    assert ox.compare_execution(model, new_model, inp_dict)

    # Verify graph structure changes
    expected_node_count = 1  # Start with MultiThreshold
    if extract_scale:
        expected_node_count += 1  # Add Mul node
    if extract_bias:
        expected_node_count += 1  # Add Add node

    assert len(new_model.graph.node) == expected_node_count

    # Verify MultiThreshold attributes were reset
    mt_node_transformed = new_model.graph.node[0]
    assert mt_node_transformed.op_type == "MultiThreshold"
    mt_inst = getCustomOp(mt_node_transformed)
    assert mt_inst.get_nodeattr("out_scale") == 1.0
    assert mt_inst.get_nodeattr("out_bias") == 0.0

    # Verify node types in correct order
    if extract_scale and extract_bias:
        assert new_model.graph.node[0].op_type == "MultiThreshold"
        assert new_model.graph.node[1].op_type == "Mul"
        assert new_model.graph.node[2].op_type == "Add"
    elif extract_scale:
        assert new_model.graph.node[0].op_type == "MultiThreshold"
        assert new_model.graph.node[1].op_type == "Mul"
    elif extract_bias:
        assert new_model.graph.node[0].op_type == "MultiThreshold"
        assert new_model.graph.node[1].op_type == "Add"
