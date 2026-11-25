############################################################################
# Copyright (C) 2020-2022, Xilinx, Inc.
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

# Note: This files is migrated and extended from qonnx.util.config
# For more information on the git history of the file see here:
# https://github.com/fastmachinelearning/qonnx/blob/
# abb9eb12e0248014a805f505aacfaeb14d42409a/src/qonnx/util/config.py


import json
import onnx
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp, is_custom_op
from typing import Any


# update this code to handle export configs from subgraphs
# where the subgraph is found in a node's attribute as a graph type
def extract_model_config(
    model: ModelWrapper,
    subgraph_hier: str | None,
    attr_names_to_extract: list[str],
    apply_to_subgraphs: bool = False,
) -> dict[str, dict[str, Any]]:
    """Create a dictionary with layer name -> attribute mappings extracted from the
    model. The created dictionary can be later applied on a model with
    finn.transformation.general.ApplyConfig.

    When apply_to_subgraphs is True, nodes in subgraphs are also extracted and
    prefixed with their parent hierarchy using '_' as separator. For example, a
    node 'Conv_0' inside a subgraph of node 'IfNode_0' will be exported as
    'IfNode_0_<attr>_Conv_0' in the config. This matches the hierarchical key
    convention consumed by ApplyConfig(config, apply_to_subgraphs=True), so the
    two form a symmetric export/import pair. Defaults to False (top-level only)."""

    cfg: dict[str, dict[str, Any]] = dict()
    cfg["Defaults"] = dict()
    for n in model.graph.node:
        new_hier = n.name if subgraph_hier is None else str(subgraph_hier) + "_" + n.name

        # Check if this is a custom op and prepare to extract attributes
        is_custom = is_custom_op(n.domain, n.op_type)
        if is_custom:
            oi = getCustomOp(n)
            layer_dict: dict[str, Any] = dict()
            for attr in attr_names_to_extract:
                try:
                    layer_dict[attr] = oi.get_nodeattr(attr)
                except AttributeError:
                    pass

        # Process node attributes - handle both subgraphs and extractable attributes
        if apply_to_subgraphs:
            for attr in n.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    # If the attribute is a graph, extract configs from the subgraph
                    # recursively. Include the subgraph attribute name in the hierarchy
                    subgraph_hier_with_attr = new_hier + "_" + attr.name
                    cfg.update(
                        extract_model_config(
                            model.make_subgraph_modelwrapper(attr.g),
                            subgraph_hier_with_attr,
                            attr_names_to_extract,
                            apply_to_subgraphs=apply_to_subgraphs,
                        )
                    )

        # Add the node's config if we extracted any attributes
        if is_custom and len(layer_dict) > 0:
            cfg[new_hier] = layer_dict

    return cfg


def extract_model_config_to_json(
    model: ModelWrapper,
    json_filename: str,
    attr_names_to_extract: list[str],
    apply_to_subgraphs: bool = False,
) -> None:
    """Create a json file with layer name -> attribute mappings extracted from the
    model. The created json file can be later applied on a model with
    finn.transformation.general.ApplyConfig."""

    with open(json_filename, "w") as f:
        json.dump(
            extract_model_config(
                model,
                subgraph_hier=None,
                attr_names_to_extract=attr_names_to_extract,
                apply_to_subgraphs=apply_to_subgraphs,
            ),
            f,
            indent=2,
        )


def _hier_name(subgraph_hier, name):
    """Config key of a node, as built by extract_model_config."""
    return name if subgraph_hier is None else str(subgraph_hier) + "_" + name


def _nodes_with_hier(model, subgraph_hier, apply_to_subgraphs):
    """Yield (node, subgraph_hier) for all nodes, descending into graph
    attributes like extract_model_config when apply_to_subgraphs is set."""
    for n in model.graph.node:
        yield n, subgraph_hier
        if apply_to_subgraphs:
            for attr in n.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    yield from _nodes_with_hier(
                        model.make_subgraph_modelwrapper(attr.g),
                        _hier_name(subgraph_hier, n.name) + "_" + attr.name,
                        apply_to_subgraphs,
                    )


def extract_model_config_consolidate_shuffles(
    model: ModelWrapper,
    output_file: str,
    hw_attrs: list[str],
    apply_to_subgraphs: bool = False,
) -> None:
    """Export flow that takes into consideration how Shuffle operations have been decomposed"""
    extract_model_config_to_json(
        model, output_file, hw_attrs, apply_to_subgraphs=apply_to_subgraphs
    )

    with open(output_file, "r") as f:
        config: dict[str, Any] = json.load(f)

    shuffle_configs: dict[str, dict[str, Any]] = {}
    nodes_to_remove: list[str] = []

    for node, hier in _nodes_with_hier(model, None, apply_to_subgraphs):
        if node.op_type in ["InnerShuffle_rtl", "OuterShuffle_hls"]:
            inst = getCustomOp(node)
            original_name = inst.get_nodeattr("original_node_name")
            original_simd = inst.get_nodeattr("original_simd")
            # keys inside subgraphs (e.g. FINNLoop bodies) carry the hierarchy
            # prefix of extract_model_config; the original Shuffle sat in the
            # same graph, so it shares the prefix
            key = _hier_name(hier, node.name)
            original_key = _hier_name(hier, original_name)

            if original_name and key in config:
                if original_key not in shuffle_configs:
                    consolidated_config = config[key].copy()
                    if original_simd is not None:
                        consolidated_config["SIMD"] = original_simd
                    shuffle_configs[original_key] = consolidated_config
                nodes_to_remove.append(key)

    for node_name in nodes_to_remove:
        del config[node_name]

    config.update(shuffle_configs)

    with open(output_file, "w") as f:
        json.dump(config, f, indent=2)
