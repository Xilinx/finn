# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import json
from finn_ci import sharding

pytestmark = pytest.mark.util


def write_json(path, payload):
    path.write_text(json.dumps(payload))


def test_load_group_weights_ignores_missing_and_corrupt(tmp_path):
    missing = tmp_path / "missing.json"
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{")

    assert sharding.load_group_weights(str(missing)) == {}
    assert sharding.load_group_weights(str(corrupt)) == {}


def test_load_group_weights_returns_max_of_samples(tmp_path):
    master = tmp_path / "master.json"
    write_json(master, {"groups": {"g": {"samples": [10.0, 20.0, 30.0]}}})
    weights = sharding.load_group_weights(str(master))
    assert weights["g"] == 30.0


def test_assign_groups_round_robin_without_timing_signal():
    assignment, source, shard_load, fallback = sharding.assign_groups_to_shards(
        ["b", "a", "c"], 2, weights={}
    )

    assert assignment == {"a": 0, "b": 1, "c": 0}
    assert source == {"a": "round_robin", "b": "round_robin", "c": "round_robin"}
    assert shard_load == [2.0, 1.0]
    assert fallback == 1.0


def test_assign_groups_pins_override_timing_weights():
    assignment, source, shard_load, fallback = sharding.assign_groups_to_shards(
        ["slow", "pinned"], 2, weights={"slow": 100.0, "pinned": 1.0}, pins={"pinned": 0}
    )

    assert assignment["pinned"] == 0
    assert assignment["slow"] == 1
    assert source["pinned"] == "pinned"
    assert source["slow"] == "known"
    assert fallback == 100.0


def test_assign_groups_unknown_pins_use_fallback_weight():
    assignment, source, shard_load, fallback = sharding.assign_groups_to_shards(
        ["known", "pinned"], 2, weights={"known": 9.0}, pins={"pinned": 0}
    )

    assert assignment["pinned"] == 0
    assert source["pinned"] == "pinned"
    assert fallback == 9.0
    assert shard_load[0] == 9.0


def test_assign_groups_rejects_zero_or_negative_num_shards():
    # Defence-in-depth: a future direct caller (off the plugin path that
    # validates separately) must get a loud error rather than an IndexError
    # when shard_load[shard] is written.
    with pytest.raises(ValueError, match="num_shards must be >= 1"):
        sharding.assign_groups_to_shards(["a"], 0)
    with pytest.raises(ValueError, match="num_shards must be >= 1"):
        sharding.assign_groups_to_shards(["a"], -1)


def test_assign_groups_rejects_out_of_range_pin():
    with pytest.raises(ValueError, match=r"pin for 'k' out of range"):
        sharding.assign_groups_to_shards(["k"], 2, pins={"k": 2})
    with pytest.raises(ValueError, match=r"pin for 'k' out of range"):
        sharding.assign_groups_to_shards(["k"], 2, pins={"k": -1})


def test_assign_groups_rejects_non_int_pin():
    with pytest.raises(ValueError, match="must be an int"):
        sharding.assign_groups_to_shards(["k"], 2, pins={"k": "zero"})


def test_canonical_key_strips_xdist_group_suffix():
    # xdist's loadgroup forwards a nodeid with an "@<group>" suffix; canonical_key
    # recovers the bare group name so its members sum together, not as singletons.
    assert sharding.canonical_key("tests/x.py::test_a@grpA") == "grpA"


def test_canonical_key_ignores_param_at_sign():
    # an "@" inside a parametrize id (e.g. an email-like value) must not be
    # mistaken for an xdist loadgroup "@<group>" suffix, which would collapse
    # unrelated tests into one group and one shared timing weight.
    nodeid = "tests/x.py::test_a[user@host]"
    assert sharding.canonical_key(nodeid) == nodeid


def test_canonical_key_reduces_notebook_path_to_basename():
    nodeid = "tests/notebooks/test_x.py::test_run[/ws/abs/foo.ipynb]"
    assert sharding.canonical_key(nodeid) == "tests/notebooks/test_x.py::test_run[foo.ipynb]"
