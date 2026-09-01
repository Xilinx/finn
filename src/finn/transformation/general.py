############################################################################
# Copyright (C) 2020-2022, Xilinx, Inc.
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

# Note: This transformation is migrated and extended from qonnx.transformation.general
# For more information on the git history of the file see here:
# https://github.com/fastmachinelearning/qonnx/blob/
# abb9eb12e0248014a805f505aacfaeb14d42409a/src/qonnx/transformation/general.py

import json
import warnings

# Protobuf onnx graph node type
from onnx import AttributeProto, NodeProto, mapping  # noqa
from qonnx.custom_op.registry import getCustomOp, is_custom_op
from qonnx.transformation.base import Transformation


class ApplyConfig(Transformation):
    """Applies node properties (attributes) from either a config dict or its JSON
    representation given as a filename.
    The JSON file can specify default values for particular op_types, as well
    as values for nodes with particular names. Example dict::

        {
        # set kernel_size = 3 for all nodes with op_type=Im2Col
        "Defaults" : {"kernel_size" : [3, ["Im2Col"]]},
        # set kernel_size = 7 for the particular node with name Im2Col_0
        "Im2Col_0" : {"kernel_size" : 7}
        }

    Note: ApplyConfig traverses subgraphs itself (using hierarchical config keys
    of the form ``<parent_hier>_<attr_name>_<node_name>``, matching
    finn.util.config.extract_model_config). This descent is controlled by the
    ``apply_to_subgraphs`` constructor argument and defaults to False (top-level
    only). Do NOT pass ``apply_to_subgraphs=True`` to ``ModelWrapper.transform``
    for this transform - the generic mechanism re-enters each subgraph as a
    standalone top-level model, which loses the hierarchical key context and
    would look up nodes with the wrong (flat) keys.
    """

    # Marker: this transform manages its own subgraph traversal (via the
    # apply_to_subgraphs constructor argument). A guard in ModelWrapper.transform
    # can check this marker to reject the generic apply_to_subgraphs=True path.
    handles_subgraphs_internally = True

    def __init__(self, config, node_filter=lambda x: True, apply_to_subgraphs=False):
        super().__init__()
        self.config = config
        self.node_filter = node_filter
        self.apply_to_subgraphs = apply_to_subgraphs
        self.used_configurations = ["Defaults"]
        self.missing_configurations = []
        self.ignored_non_custom_configurations = []

    def configure_network(self, graph_proto, model_config, subgraph_hier):
        # Configure network - graph_proto can be a GraphProto or ModelWrapper
        # If it's a ModelWrapper, get the graph
        if hasattr(graph_proto, "graph"):
            graph = graph_proto.graph
        else:
            graph = graph_proto

        for node in graph.node:
            if not self.node_filter(node):
                continue

            # Build the config key by prepending hierarchy
            config_key = (
                node.name if subgraph_hier is None else str(subgraph_hier) + "_" + node.name
            )

            try:
                node_config = model_config[config_key].copy()
            except KeyError:
                self.missing_configurations += [node.name]
                node_config = {}

            if is_custom_op(node.domain, node.op_type):
                inst = getCustomOp(node)

                if "Defaults" in model_config.keys():
                    # set specified defaults
                    default_values = []
                    for key, value in model_config["Defaults"].items():
                        assert len(value) % 2 == 0
                        if key not in model_config:
                            for val, op in zip(value[::2], value[1::2]):
                                default_values.append((key, val, op))
                                assert not (op == "all" and len(value) > 2)
                    default_configs = {
                        key: val
                        for key, val, op in default_values
                        if op == "all" or node.op_type in op
                    }
                    for attr_name, value in default_configs.items():
                        inst.set_nodeattr(attr_name, value)

                # set node attributes from specified configuration
                for attr_name, value in node_config.items():
                    if attr_name == "impl_style" and node.op_type.startswith("StreamingFIFO"):
                        # impl_style was retired for FIFOs (fifo.sv is the only backend).
                        # Tolerate it in old configs for now, but flag it as a no-op.
                        warnings.warn(
                            "Setting 'impl_style' on %s is meaningless and will raise an "
                            "error in a future release; remove it from your config." % node.name,
                            DeprecationWarning,
                        )
                        continue
                    inst.set_nodeattr(attr_name, value)

                if node_config:
                    self.used_configurations += [config_key]
            elif node_config:
                self.ignored_non_custom_configurations += [(config_key, node.op_type)]

            # Recursively handle nested subgraphs
            if self.apply_to_subgraphs:
                for attr in node.attribute:
                    if attr.type == AttributeProto.GRAPH:
                        # Build the subgraph hierarchy including the attribute name
                        if subgraph_hier is None:
                            new_hier = node.name
                        else:
                            new_hier = str(subgraph_hier) + "_" + node.name
                        # Include the subgraph attribute name in the hierarchy
                        new_hier = new_hier + "_" + attr.name
                        self.configure_network(attr.g, model_config, subgraph_hier=new_hier)

    def apply(self, model):
        if isinstance(self.config, dict):
            model_config = self.config
        else:
            with open(self.config, "r") as f:
                model_config = json.load(f)

        # apply configuration on upper level
        self.configure_network(model.model.graph, model_config, subgraph_hier=None)

        # Configuration verification
        # Remove duplicates from missing_configurations
        # (can happen with shared subgraphs in If nodes)
        unique_missing = list(dict.fromkeys(self.missing_configurations))
        if len(unique_missing) > 0:
            warnings.warn("\nNo HW configuration for nodes: " + ", ".join(unique_missing))

        # Check for matched configs that couldn't be applied because they were
        # specified for standard ONNX nodes instead of custom ops.
        unique_non_custom = list(dict.fromkeys(self.ignored_non_custom_configurations))
        if len(unique_non_custom) > 0:
            formatted_non_custom = [
                "{} ({})".format(config_key, op_type) for config_key, op_type in unique_non_custom
            ]
            warnings.warn(
                "\nHW configurations for non-custom nodes were ignored: "
                + ", ".join(formatted_non_custom)
                + ". Configs can only be applied to custom ops."
            )

        # Check for unused configs (top-level configs that weren't applied)
        ignored_configurations = [
            config_key for config_key, _ in self.ignored_non_custom_configurations
        ]
        unused_configs = [
            x
            for x in model_config
            if x not in self.used_configurations
            and x not in ignored_configurations
            and x != "Defaults"
        ]
        if len(unused_configs) > 0:
            warnings.warn("\nUnused HW configurations: " + ", ".join(unused_configs))

        # one iteration is enough
        return (model, False)
