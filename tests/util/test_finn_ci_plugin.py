# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import json
import os
import time
from finn_ci import plugin, sharding

pytestmark = pytest.mark.util
pytest_plugins = ["pytester"]


class _StubItem:
    # minimal stand-in for a pytest Item: _group_key only needs nodeid and an
    # (empty) xdist_group marker iterator.
    def __init__(self, nodeid):
        self.nodeid = nodeid

    def iter_markers(self, name):
        return iter(())


def test_group_key_canonicalises_ungrouped_notebook_nodeid():
    # the timing master stores canonical group names (notebook paths reduced to
    # basename), so the sharding read path must canonicalise an ungrouped nodeid
    # the same way or a recorded notebook weight could never be matched.
    nodeid = "tests/notebooks/test_x.py::test_run[/ws/abs/foo.ipynb]"
    group = plugin._group_key(_StubItem(nodeid))
    assert group == "tests/notebooks/test_x.py::test_run[foo.ipynb]"
    assert group == sharding.canonical_key(nodeid)
    # a plain nodeid is returned unchanged (canonical_key is a no-op there)
    plain = "tests/util/test_x.py::test_y[1-2]"
    assert plugin._group_key(_StubItem(plain)) == plain


def _install_finn_ci_plugin(pytester):
    """Wire the real plugin into a pytester sandbox via pytest_plugins.

    The plugin is a normal importable module (finn_ci.plugin on sys.path), so the
    sandbox conftest only has to name it. Worker detection in the plugin keys
    off the inner config's workerinput attribute rather than the inherited
    PYTEST_XDIST_WORKER env var, so an inline sub-run is correctly seen as a
    non-worker and no env scrubbing is needed.
    """
    pytester.makeconftest('pytest_plugins = ["finn_ci.plugin"]')


def test_pytest_plugin_writes_timings_for_successful_sharded_run(pytester):
    _install_finn_ci_plugin(pytester)
    pytester.makepyfile(
        test_sample="""
def test_ok():
    assert True
"""
    )

    result = pytester.runpytest(
        "--num-shards=1",
        "--shard-id=0",
        "--junitxml=stage.xml",
        "-q",
    )

    result.assert_outcomes(passed=1)
    data = json.loads((pytester.path / "stage.timings.json").read_text())
    assert data["stash"] == "stage"
    assert data["shard"] == {"num": 1, "id": 0}
    assert data["groups"][0]["name"].endswith("test_sample.py::test_ok")


def test_pytest_plugin_writes_empty_shard_sidecar_when_slice_collected_zero(pytester):
    # Two single-test groups round-robin onto shards 0 and 1 (sorted by
    # nodeid). Shard 1 therefore gets one item, shard 0 gets the other.
    # If we then ask for shard 0 with a marker that only matches the
    # second test, the slice is empty and the plugin must:
    #  - remap exit 5 to 0 so the build does not fail spuriously
    #  - drop a <stash>.empty-shard sidecar so the aggregator can tell
    #    "shard had no work" apart from "shard crashed".
    _install_finn_ci_plugin(pytester)
    pytester.makepyfile(
        test_sample="""
import pytest

@pytest.mark.fpgadataflow
def test_a():
    assert True

@pytest.mark.end2end
def test_b():
    assert True
"""
    )

    result = pytester.runpytest(
        "-m",
        "fpgadataflow",
        "--num-shards=2",
        "--shard-id=1",
        "--junitxml=stage.xml",
        "-q",
    )

    assert result.ret == 0
    sidecar = pytester.path / "stage.empty-shard"
    assert sidecar.exists()
    assert "0 items" in sidecar.read_text()


def test_pytest_plugin_rejects_conflicting_shard_pins_within_xdist_group(pytester):
    # Two tests share an xdist_group and disagree on @pytest.mark.shard(N).
    # _assignment_details must surface this as a UsageError rather than
    # silently splitting a chained checkpoint sequence across shards.
    _install_finn_ci_plugin(pytester)
    pytester.makepyfile(
        test_sample="""
import pytest

@pytest.mark.xdist_group(name="chain")
@pytest.mark.shard(0)
def test_first():
    assert True

@pytest.mark.xdist_group(name="chain")
@pytest.mark.shard(1)
def test_second():
    assert True
"""
    )

    result = pytester.runpytest(
        "--num-shards=2",
        "--shard-id=0",
        "--junitxml=stage.xml",
        "-q",
    )

    assert result.ret != 0
    combined = "\n".join(result.outlines + result.errlines)
    assert "conflicting" in combined
    assert "chain" in combined


def test_pytest_plugin_dry_run_prints_per_shard_table_and_exits_zero(pytester):
    # --dry-run-shards must print the header row and exit 0 without running
    # any test. We deselect everything by design so exit 5 is benign.
    _install_finn_ci_plugin(pytester)
    pytester.makepyfile(
        test_sample="""
def test_a():
    assert True

def test_b():
    assert True
"""
    )

    result = pytester.runpytest(
        "--num-shards=2",
        "--shard-id=0",
        "--dry-run-shards",
        "--junitxml=stage.xml",
        "-q",
    )

    assert result.ret == 0
    out = "\n".join(result.outlines)
    assert "dry-run-shards" in out
    assert "shard" in out and "items" in out and "groups" in out
    # No tests should have actually run.
    result.assert_outcomes()


def test_pytest_plugin_no_double_count_under_xdist(pytester):
    # Each CI shard runs under xdist, so the controller must aggregate every
    # worker's forwarded reports into one timings.json while workers keep
    # self.timings=None and never write a partial one. Under --dist loadgroup
    # the forwarded nodeids carry an @<group> suffix, so the plugin folds them
    # back onto the xdist_group via canonical_key: each group must appear once,
    # its members counted once each AND their seconds summed (not maxed), which
    # is the weight the timing master keys on downstream.
    pytest.importorskip("xdist")
    # runpytest_subprocess is a fresh interpreter, so the ci/ dir that the repo
    # conftest puts on sys.path at runtime is not visible to it. Make the
    # sandbox conftest add ci/ itself before naming the plugin.
    ci_dir = os.path.dirname(os.path.dirname(os.path.abspath(sharding.__file__)))
    pytester.makeconftest(
        "import sys\n" "sys.path.insert(0, %r)\n" "pytest_plugins = ['finn_ci.plugin']\n" % ci_dir
    )
    # grpA's members sleep unequally (0.30 + 0.05) so the summed group weight
    # sits well above the 0.30 slowest single member: a max-per-test regression
    # would record ~0.30 and fail the seconds assertion below.
    pytester.makepyfile(
        test_sample="""
import time
import pytest

@pytest.mark.xdist_group(name="grpA")
def test_a():
    time.sleep(0.30)

@pytest.mark.xdist_group(name="grpA")
def test_b():
    time.sleep(0.05)

@pytest.mark.xdist_group(name="grpB")
def test_c():
    time.sleep(0.05)

@pytest.mark.xdist_group(name="grpB")
def test_d():
    time.sleep(0.05)
"""
    )

    result = pytester.runpytest_subprocess(
        "-n",
        "2",
        "--dist",
        "loadgroup",
        "--num-shards=1",
        "--shard-id=0",
        "--junitxml=stage.xml",
        "-q",
    )

    result.assert_outcomes(passed=4)
    data = json.loads((pytester.path / "stage.timings.json").read_text())
    by_group = {}
    for g in data["groups"]:
        key = sharding.canonical_key(g["name"])
        entry = by_group.setdefault(key, {"count": 0, "seconds": 0.0})
        entry["count"] += g["count"]
        entry["seconds"] += g["seconds"]
    assert {k: v["count"] for k, v in by_group.items()} == {"grpA": 2, "grpB": 2}
    assert by_group["grpA"]["seconds"] > 0.33


class _FakeConfig:
    def __init__(self, options):
        self._options = options

    def getoption(self, name):
        return self._options.get(name)


class _FakeSession:
    def __init__(self, config, exitstatus):
        self.config = config
        self.exitstatus = exitstatus


def _drive_sessionfinish(tmp_path, exitstatus):
    # exercise the per-shard trust guard via a stub session, without spinning up
    # a full pytest run just to reach pytest_sessionfinish.
    instance = plugin._FinnCiShardingPlugin()
    instance.timings = {
        "per_test_seconds": {"tests/foo.py::test_a": 1.0},
        "nodeid_to_group": {},
        "start_monotonic": time.monotonic(),
    }
    config = _FakeConfig(
        {
            "--num-shards": 1,
            "--shard-id": 0,
            "--dry-run-shards": False,
            "--junitxml": str(tmp_path / "stage.xml"),
        }
    )
    instance.pytest_sessionfinish(_FakeSession(config, exitstatus), exitstatus)
    return tmp_path / "stage.timings.json"


def test_plugin_skips_timings_sidecar_on_interrupted_session(tmp_path):
    # an aborted/interrupted shard has truncated timings and must not feed the
    # master, so no sidecar is written.
    assert not _drive_sessionfinish(tmp_path, pytest.ExitCode.INTERRUPTED).exists()


def test_pytest_plugin_writes_timings_sidecar_on_test_failure(pytester):
    # a plain test failure (exit 1) still has real durations, so the sidecar is
    # written and can feed the master. only interrupted/aborted runs skip it.
    _install_finn_ci_plugin(pytester)
    pytester.makepyfile(
        test_sample="""
def test_fails():
    assert False
"""
    )

    result = pytester.runpytest(
        "--num-shards=1",
        "--shard-id=0",
        "--junitxml=stage.xml",
        "-q",
    )

    result.assert_outcomes(failed=1)
    data = json.loads((pytester.path / "stage.timings.json").read_text())
    assert data["groups"][0]["name"].endswith("test_sample.py::test_fails")
