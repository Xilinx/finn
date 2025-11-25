############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

import pytest

import onnx
import os
from onnxscript import BOOL, FLOAT
from onnxscript import opset13 as op
from onnxscript import script
from onnxscript.values import Opset
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from typing import Any, Dict

from finn.transformation.general import ApplyConfig
from finn.util.config import extract_model_config, extract_model_config_to_json

# this is a pretend opset so that we can create
# qonnx custom ops with onnxscript
qops = Opset("qonnx.custom_op.general", 1)


@script(default_opset=op)
def main_graph_fn(
    main_inp: FLOAT[1, 28, 28, 1], condition: BOOL, nested_condition: BOOL
) -> FLOAT[1, 4, 4, 144]:
    """Main graph with nested if statement in else branch."""
    im2col_0 = qops.Im2Col(
        main_inp,
        stride=[1, 1],
        kernel_size=[3, 3],
        pad_amount=[1, 1, 1, 1],
        input_shape=[1, 28, 28, 1],
    )

    # Python if statement → ONNX If node with subgraph
    # settings for Im2Col are meant to validate the extraction/application of attributes
    # and are not necessarily realistic or correct
    if condition:
        # Then branch: simple subgraph (2 levels)
        main_out = qops.Im2Col(
            im2col_0,
            stride=[2, 1],
            kernel_size=[5, 5],
            pad_amount=[2, 2, 2, 2],
            input_shape=[1, 14, 14, 144],
        )
    else:
        im2col_1 = qops.Im2Col(
            im2col_0,
            stride=[2, 1],
            kernel_size=[6, 6],
            pad_amount=[3, 3, 3, 3],
            input_shape=[1, 14, 14, 145],
        )
        # Else branch: nested if statement (3 levels)
        if nested_condition:
            main_out = qops.Im2Col(
                im2col_1,
                stride=[3, 1],
                kernel_size=[7, 7],
                pad_amount=[4, 4, 4, 4],
                input_shape=[1, 4, 4, 146],
            )
        else:
            main_out = qops.Im2Col(
                im2col_1,
                stride=[3, 2],
                kernel_size=[8, 8],
                pad_amount=[5, 5, 5, 5],
                input_shape=[1, 4, 4, 147],
            )

    return main_out


def build_expected_config_from_node(node: onnx.NodeProto, prefix="") -> Dict[str, Any]:
    """Build expected config dictionary from a given ONNX node."""
    custom_op = getCustomOp(node)
    attrs = {}
    for attr in node.attribute:
        attrs[attr.name] = custom_op.get_nodeattr(attr.name)
    return {prefix + node.name: attrs}


def make_im2col_test_model():
    """Create a simple ONNX model with a single Im2Col node."""

    model_proto = main_graph_fn.to_model_proto()

    im2col_node = model_proto.graph.node[0]
    if_im2col_then_node = model_proto.graph.node[1].attribute[0].g.node[0]
    if_im2col_else_node = model_proto.graph.node[1].attribute[1].g.node[0]
    nested_if_im2col_then_node = (
        model_proto.graph.node[1].attribute[1].g.node[1].attribute[0].g.node[0]
    )
    nested_if_im2col_else_node = (
        model_proto.graph.node[1].attribute[1].g.node[1].attribute[1].g.node[0]
    )

    # this test assumes that all Im2Col nodes have the same name
    # to verify that node aliasing is handled correctly between nodes on
    # the same and different levels of the hierarchy
    assert im2col_node.name == if_im2col_then_node.name
    assert im2col_node.name == if_im2col_else_node.name
    assert im2col_node.name == nested_if_im2col_then_node.name
    assert im2col_node.name == nested_if_im2col_else_node.name

    expected_config = {}
    expected_config["Defaults"] = {}
    expected_config.update(build_expected_config_from_node(im2col_node))
    expected_config.update(
        build_expected_config_from_node(if_im2col_then_node, prefix="n1_then_branch_")
    )
    expected_config.update(
        build_expected_config_from_node(if_im2col_else_node, prefix="n1_else_branch_")
    )
    expected_config.update(
        build_expected_config_from_node(
            nested_if_im2col_then_node, prefix="n1_else_branch_n1_then_branch_"
        )
    )
    expected_config.update(
        build_expected_config_from_node(
            nested_if_im2col_else_node, prefix="n1_else_branch_n1_else_branch_"
        )
    )

    return ModelWrapper(model_proto), expected_config


@pytest.mark.util
def test_extract_model_config():
    """Test extraction of model config from models with and without subgraphs."""

    model, expected_config = make_im2col_test_model()

    attrs_to_extract = ["kernel_size", "stride", "pad_amount", "input_shape"]

    extracted_config = extract_model_config(
        model, subgraph_hier=None, attr_names_to_extract=attrs_to_extract
    )
    assert extracted_config == expected_config, "Extracted config does not match expected config"


@pytest.mark.util
def test_roundtrip_export_import():
    """Test config extraction and re-application preserves node attributes."""
    model, expected_config = make_im2col_test_model()
    attrs_to_extract = ["kernel_size", "stride", "pad_amount", "input_shape"]

    # Extract config from model
    original_config = extract_model_config(
        model, subgraph_hier=None, attr_names_to_extract=attrs_to_extract
    )
    # Save in json
    config_json_file = os.environ["FINN_BUILD_DIR"] + "/original_config.json"
    extract_model_config_to_json(model, config_json_file, attrs_to_extract)

    # Modify all Im2Col nodes to different values (recursively through subgraphs)
    def modify_all_im2col_nodes(graph_proto):
        for node in graph_proto.node:
            if node.op_type == "Im2Col":
                inst = getCustomOp(node)
                inst.set_nodeattr("kernel_size", [11, 11])
                inst.set_nodeattr("stride", [5, 5])
                inst.set_nodeattr("pad_amount", [7, 7, 7, 7])
                inst.set_nodeattr("input_shape", "")  # input_shape is a string attribute
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    modify_all_im2col_nodes(attr.g)

    modify_all_im2col_nodes(model.graph)
    # test that modification happened
    new_config = extract_model_config(
        model, subgraph_hier=None, attr_names_to_extract=attrs_to_extract
    )
    assert new_config != original_config, "Config of nodes wasn't properly changed"

    model = model.transform(ApplyConfig(config_json_file))

    # Re-extract config and verify it matches original
    restored_config = extract_model_config(
        model, subgraph_hier=None, attr_names_to_extract=attrs_to_extract
    )
    assert restored_config == original_config, "Config not properly restored after roundtrip"

    if os.path.exists(config_json_file):
        os.remove(config_json_file)
