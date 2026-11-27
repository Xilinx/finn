# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Guard the in-tree folding and specialize configs against stale node names.

ApplyConfig only warns about config keys that match no node: a key left behind
by an op_type rename silently leaves the node at its default attributes.
"""

import pytest

import json
import numpy as np
import os
import re
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import qonnx_make_model

import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.custom_op.fpgadataflow import custom_op as hw_ops
from finn.custom_op.fpgadataflow.hls import custom_op as hls_ops
from finn.custom_op.fpgadataflow.rtl import custom_op as rtl_ops
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.general import ApplyConfig

pytestmark = pytest.mark.util

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIRS = ["src/finn/qnn-data", "tests/benchmark", "notebooks", "tutorials"]
REGISTERED_OP_TYPES = set(hw_ops) | set(hls_ops) | set(rtl_ops)


def find_node_configs():
    """Return the repo-relative paths of all ApplyConfig JSON files, i.e. the
    JSON files whose top level is an object with a "Defaults" entry."""
    found = []
    for config_dir in CONFIG_DIRS:
        for root, _, files in os.walk(os.path.join(REPO_ROOT, config_dir)):
            for name in sorted(files):
                if not name.endswith(".json"):
                    continue
                path = os.path.join(root, name)
                with open(path) as f:
                    content = json.load(f)
                if isinstance(content, dict) and "Defaults" in content:
                    found.append(os.path.relpath(path, REPO_ROOT))
    return sorted(found)


NODE_CONFIGS = find_node_configs()


def load_config(rel_path):
    with open(os.path.join(REPO_ROOT, rel_path)) as f:
        return json.load(f)


def key_op_type(key):
    """Return the registered op_type that a node-name key was derived from by
    GiveUniqueNodeNames (<op_type>_<index>), or None. Hierarchical subgraph keys
    (<parent>_<attr>_<op_type>_<index>) are matched by their suffix."""
    m = re.fullmatch(r"(.+)_\d+", key)
    if m is None:
        return None
    stem = m.group(1)
    for op_type in REGISTERED_OP_TYPES:
        if stem == op_type or stem.endswith("_" + op_type):
            return op_type
    return None


def test_node_configs_found():
    # guard the discovery itself: an empty parametrization would pass vacuously
    assert len(NODE_CONFIGS) >= 30


@pytest.mark.parametrize("config_file", NODE_CONFIGS)
def test_config_keys_name_registered_op_types(config_file):
    config = load_config(config_file)
    stale_keys = [k for k in config if k != "Defaults" and key_op_type(k) is None]
    assert not stale_keys, "%s: keys match no registered op_type: %s" % (config_file, stale_keys)
    stale_defaults = [
        op_type
        for value in config["Defaults"].values()
        for op_types in value[1::2]
        if op_types != "all"
        for op_type in op_types
        if op_type not in REGISTERED_OP_TYPES
    ]
    assert not stale_defaults, "%s: Defaults name unregistered op_types: %s" % (
        config_file,
        stale_defaults,
    )


def make_topk_model(labels, k):
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, labels])
    values = helper.make_tensor_value_info("values", TensorProto.FLOAT, [1, k])
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [1, k])
    k_init = helper.make_tensor("k", TensorProto.INT64, [1], [k])
    topk_node = helper.make_node("TopK", ["inp", "k"], ["values", "indices"], axis=-1)
    # the values output stays unconsumed, as InferTopKLayer requires
    graph = helper.make_graph(
        [topk_node], "topk_graph", [inp], [indices], value_info=[values], initializer=[k_init]
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="topk-model"))
    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_initializer("k", np.asarray([k], dtype=np.int64))
    return model


TOPK_CONFIGS = [
    c
    for c in NODE_CONFIGS
    if any(key_op_type(k) in ("HWTopK", "HWTopK_hls") for k in load_config(c))
]


@pytest.mark.parametrize("config_file", TOPK_CONFIGS)
def test_topk_config_keys_match_default_node_names(config_file):
    """The HWTopK entries of each config are applied to the node that a TopK
    becomes under the default naming: HWTopK_0 after InferTopKLayer, HWTopK_hls_0
    after SpecializeLayers."""
    config = load_config(config_file)
    model = make_topk_model(labels=1000, k=5)
    model = model.transform(to_hw.InferTopKLayer())
    model = model.transform(GiveUniqueNodeNames())
    for stage_op_type in ["HWTopK", "HWTopK_hls"]:
        if stage_op_type == "HWTopK_hls":
            model = model.transform(SpecializeLayers("xc7z020clg400-1"))
            model = model.transform(GiveUniqueNodeNames())
        stage_config = {k: v for k, v in config.items() if key_op_type(k) == stage_op_type}
        if not stage_config:
            continue
        apply_config = ApplyConfig({"Defaults": {}, **stage_config})
        model = model.transform(apply_config)
        assert set(stage_config) <= set(
            apply_config.used_configurations
        ), "%s: %s match no node, nodes are %s" % (
            config_file,
            sorted(stage_config),
            [n.name for n in model.graph.node],
        )
