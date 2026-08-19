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
import numpy as np
import warnings

# Protobuf onnx graph node type
from onnx import AttributeProto
from qonnx.custom_op.registry import getCustomOp, is_custom_op
from qonnx.transformation.base import Transformation
from qonnx.util.basic import get_by_name


def maxpool_ceil_mode_output_dims(ifm_dim, k, stride, pad_begin, pad_end):
    """Return the ``(naive_ceil, drop_rule)`` output sizes for a single spatial
    dimension of a ``ceil_mode=1`` pooling op.

    ``naive_ceil`` is the ONNX-spec formula, also used by
    ``onnx.shape_inference`` and QONNX's ``compute_pool_output_dim``.
    ``drop_rule`` additionally discards any pooling window that would start
    entirely inside the trailing padding - the behaviour of PyTorch/cuDNN
    (always) and onnxruntime (>=1.21). The two disagree only for degenerate
    geometry where the last window begins in the end padding; when they agree
    the ``ceil_mode=1`` op is equivalent to ``ceil_mode=0`` in both shape and
    values."""
    naive = int(np.ceil((ifm_dim + pad_begin + pad_end - k) / stride)) + 1
    drop = naive
    while drop > 1 and (drop - 1) * stride >= ifm_dim + pad_begin:
        drop -= 1
    return naive, drop


class AssertNoAmbiguousMaxPoolCeilMode(Transformation):
    """Reject MaxPool/MaxPoolNHWC nodes with ``ceil_mode=1`` whose output size
    is ambiguous, i.e. where the ONNX-spec naive-ceil formula and the
    drop-window rule (PyTorch/cuDNN, onnxruntime>=1.21) disagree for the actual
    input dimensions.

    Such models produce different output shapes/values depending on the runtime
    used, so FINN cannot compile them safely. ``ceil_mode=1`` nodes whose
    naive-ceil and drop-rule outputs match are left untouched, since they are
    provably equivalent to ``ceil_mode=0`` for the given dimensions."""

    def apply(self, model):
        for node in model.graph.node:
            if node.op_type not in ("MaxPool", "MaxPoolNHWC"):
                continue
            ceil_mode = get_by_name(node.attribute, "ceil_mode")
            if ceil_mode is None or ceil_mode.i != 1:
                continue
            kernel_shape = get_by_name(node.attribute, "kernel_shape")
            strides = get_by_name(node.attribute, "strides")
            if kernel_shape is None or strides is None:
                continue
            kernel_shape = list(kernel_shape.ints)
            strides = list(strides.ints)
            pads = get_by_name(node.attribute, "pads")
            pads = list(pads.ints) if pads is not None else [0] * (2 * len(kernel_shape))
            ishape = model.get_tensor_shape(node.input[0])
            if ishape is None or len(ishape) != 4:
                continue
            # MaxPool is NCHW, MaxPoolNHWC is NHWC - pick the spatial dims
            spatial = list(ishape[1:3]) if node.op_type == "MaxPoolNHWC" else list(ishape[2:4])
            n_sp = len(spatial)
            for ax, ifm_dim in enumerate(spatial):
                naive, drop = maxpool_ceil_mode_output_dims(
                    ifm_dim, kernel_shape[ax], strides[ax], pads[ax], pads[ax + n_sp]
                )
                if naive != drop:
                    raise RuntimeError(
                        "MaxPool node '%s' uses ceil_mode=1 with an ambiguous output "
                        "size along spatial axis %d: the ONNX-spec naive-ceil formula "
                        "yields %d while the drop-window rule (PyTorch/cuDNN, "
                        "onnxruntime>=1.21) yields %d. FINN cannot compile this model "
                        "because different runtimes produce different output shapes. "
                        "Re-export with ceil_mode=0 or adjust padding/kernel/stride so "
                        "the last pooling window does not start inside the trailing "
                        "padding." % (node.name, ax, naive, drop)
                    )
        return (model, False)


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

    """

    def __init__(self, config, node_filter=lambda x: True):
        super().__init__()
        self.config = config
        self.node_filter = node_filter
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
                    inst.set_nodeattr(attr_name, value)

                if node_config:
                    self.used_configurations += [config_key]
            elif node_config:
                self.ignored_non_custom_configurations += [(config_key, node.op_type)]

            # Recursively handle nested subgraphs
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
