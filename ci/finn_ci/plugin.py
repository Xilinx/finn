# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pytest plugin for Jenkins shard selection and timing capture.

Each Jenkins stage runs pytest with -m <marker> --num-shards N --shard-id I.
Group assignment is deterministic and weighted by the timings in
FINN_CI_TIMINGS_FILE. If that file is absent or corrupt,
assignment falls back to deterministic round-robin over the sorted group
keys. Tests sharing an @pytest.mark.xdist_group always land in the same
shard so chained load_test_checkpoint_or_skip steps stay together. Tests
without an explicit group are grouped by nodeid.

@pytest.mark.shard(N) pins a group to shard N for flaky-test isolation.

The timing state lives on the registered plugin instance rather than a module
global, so the xdist controller, each worker, and any inline pytester sub-run
keep their own copy. That also lets pytest_runtest_logreport, which receives
no config argument, reach the state via self.
"""
import pytest

import json
import os
import time
from finn_ci import sharding

SHARD_MARKER_NAME = "shard"


def pytest_addoption(parser):
    group = parser.getgroup("finn-ci-sharding")
    group.addoption(
        "--num-shards",
        type=int,
        default=0,
        help="Split the collected test set into N deterministic shards.",
    )
    group.addoption(
        "--shard-id",
        type=int,
        default=0,
        help="Which shard (0-indexed) to run. Requires --num-shards.",
    )
    # CI-maintainer tool for tuning a STAGES row's shard count: prints the
    # predicted shard balance and exits without running anything. the weight_s
    # column is only meaningful when FINN_CI_TIMINGS_FILE points at a populated
    # timing master. With no timing data it degrades to a count-balanced view.
    group.addoption(
        "--dry-run-shards",
        action="store_true",
        default=False,
        help="CI-maintainer aid: print the predicted shard balance and exit. "
        "weight_s is meaningful only with a populated FINN_CI_TIMINGS_FILE.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "shard(N): pin this test to shard N (use sparingly for flaky-test isolation).",
    )
    config.pluginmanager.register(_FinnCiShardingPlugin(), "_finn_ci_sharding")


def _is_xdist_worker(config):
    """True on an xdist worker process.

    Checks for the workerinput attribute that xdist sets on the worker config,
    rather than the PYTEST_XDIST_WORKER env var. An inline pytester.runpytest
    sub-run inherits that env var from its parent worker and would be
    misreported as a worker, whereas the workerinput attribute does not leak
    into the sub-run.
    """
    return hasattr(config, "workerinput")


def _group_key(item):
    """Return item's xdist_group name, or its canonical nodeid as a singleton group."""
    for mark in item.iter_markers(name="xdist_group"):
        if mark.args:
            return str(mark.args[0])
        name = mark.kwargs.get("name")
        if name is not None:
            return str(name)
    return sharding.canonical_key(item.nodeid)


def _pinned_shard(item, num_shards):
    """Return the @pytest.mark.shard(N) pin for a test item, or None.

    Pins are enforced per xdist_group: every member of a group must agree on
    a single pin (or none), otherwise chained load_test_checkpoint_or_skip
    tests would split across shards and silently break the BNN end2end chains.
    The check that a group's members agree lives in _assignment_details below.
    """
    for mark in item.iter_markers(name=SHARD_MARKER_NAME):
        if not mark.args:
            continue
        pinned = mark.args[0]
        if not isinstance(pinned, int):
            raise pytest.UsageError(
                "@pytest.mark.shard expects an int arg, got %r on %s" % (pinned, item.nodeid)
            )
        if not 0 <= pinned < num_shards:
            raise pytest.UsageError(
                "@pytest.mark.shard(%d) out of range for --num-shards=%d on %s"
                % (pinned, num_shards, item.nodeid)
            )
        return pinned
    return None


def _load_group_weights():
    return sharding.load_group_weights(os.environ.get("FINN_CI_TIMINGS_FILE"))


def _weights_with_fallback():
    # weight applied to groups with no recorded timing yet: the median of the
    # recorded weights, or 1.0 when none are recorded.
    weights_table = _load_group_weights()
    fallback = sharding.fallback_weight(weights_table)
    return weights_table, fallback


def _assignment_details(groups, num_shards):
    """Resolve shard pins, then delegate to sharding.assign_groups_to_shards.

    Every xdist_group must agree on a single @pytest.mark.shard(N) value.
    Otherwise chained checkpoint tests would split across shards.
    """
    pins = {}
    for key, members in groups.items():
        group_pins = {_pinned_shard(it, num_shards) for it in members}
        group_pins.discard(None)
        if len(group_pins) > 1:
            raise pytest.UsageError(
                "conflicting @pytest.mark.shard pins within xdist_group %r: %r"
                % (key, sorted(group_pins))
            )
        if group_pins:
            pins[key] = group_pins.pop()
    weights_table, fallback = _weights_with_fallback()
    assignment, source, shard_load, fallback = sharding.assign_groups_to_shards(
        groups.keys(), num_shards, weights_table, pins
    )
    return assignment, source, shard_load, weights_table, fallback


def _junit_output_info(config):
    junitxml = config.getoption("--junitxml") or ""
    if not junitxml:
        return None, None, None
    out_dir = os.path.dirname(junitxml) or "."
    stash = os.path.splitext(os.path.basename(junitxml))[0]
    return out_dir, stash, junitxml


def _stage_name_from_env():
    return os.environ.get("FINN_CI_STAGE", "")


def _write_shard_map(config, kept, groups, assignment, source, weights_table, fallback):
    # Under xdist the controller and every worker run collection and would each
    # write the identical shardmap to the same path, so let only the controller
    # write it and avoid the concurrent writes.
    if _is_xdist_worker(config):
        return
    out_dir, stash, _ = _junit_output_info(config)
    if not out_dir or not stash:
        return
    num_shards = int(config.getoption("--num-shards"))
    shard_id = int(config.getoption("--shard-id"))
    rows = []
    lines = []
    for item in sorted(kept, key=lambda it: it.nodeid):
        group = _group_key(item)
        weight = weights_table.get(group, fallback)
        row = {
            "nodeid": item.nodeid,
            "stage": _stage_name_from_env(),
            "stash": stash,
            "shard_id": shard_id,
            "num_shards": num_shards,
            "group": group,
            "weight_s": round(float(weight), 3),
            "source": source.get(group, "unknown"),
        }
        rows.append(row)
        lines.append(
            "nodeid={nodeid} stage={stage} shard={shard_num}/{shard_count} "
            "stash={stash} group={group} weight_s={weight_s:.3f} source={source}".format(
                nodeid=row["nodeid"],
                stage=row["stage"],
                shard_num=shard_id + 1,
                shard_count=num_shards,
                stash=stash,
                group=group,
                weight_s=row["weight_s"],
                source=row["source"],
            )
        )
    try:
        with open(os.path.join(out_dir, stash + ".shardmap.json"), "w") as f:
            json.dump(rows, f, indent=2, sort_keys=True)
            f.write("\n")
        with open(os.path.join(out_dir, stash + ".shardmap.txt"), "w") as f:
            for line in lines:
                f.write(line + "\n")
    except OSError:
        pass


class _FinnCiShardingPlugin:
    """Stateful pytest hooks, one instance per process (xdist controller or worker).

    The timing dict is left as None on xdist workers so the per-test reports
    they forward to the controller are counted once (on the controller) and
    not double-counted.
    """

    def __init__(self):
        self.timings = None

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, config, items):
        # trylast lets pytest's own -m deselection run before we shard.
        num_shards = config.getoption("--num-shards")
        dry_run = config.getoption("--dry-run-shards")
        if num_shards <= 0:
            return
        shard_id = config.getoption("--shard-id")
        if not 0 <= shard_id < num_shards:
            raise pytest.UsageError(
                "--shard-id=%d out of range for --num-shards=%d" % (shard_id, num_shards)
            )
        if not items:
            # fail loudly on a stale-marker typo so the stage cannot pass
            # without running any tests.
            raise pytest.UsageError(
                "no tests collected for this marker. CI shard configuration is "
                "out of sync with the test markers"
            )

        groups = {}
        for item in items:
            groups.setdefault(_group_key(item), []).append(item)
        assignment, source, _shard_load, weights_table, fallback = _assignment_details(
            groups, num_shards
        )

        if dry_run:
            per_shard = {i: {"items": 0, "groups": [], "seconds": 0.0} for i in range(num_shards)}
            for key, members in groups.items():
                s = assignment[key]
                per_shard[s]["items"] += len(members)
                per_shard[s]["groups"].append(key)
                per_shard[s]["seconds"] += weights_table.get(key, fallback)
            print(
                "\n--- dry-run-shards (num_shards=%d, groups=%d, items=%d) ---"
                % (num_shards, len(groups), len(items))
            )
            print(
                "%-8s %-8s %-8s %-10s %s" % ("shard", "items", "groups", "weight_s", "sample_group")
            )
            for i in range(num_shards):
                gs = sorted(per_shard[i]["groups"])
                sample = gs[0] if gs else "(empty)"
                print(
                    "%-8d %-8d %-8d %-10.1f %s"
                    % (i, per_shard[i]["items"], len(gs), per_shard[i]["seconds"], sample)
                )
            config.hook.pytest_deselected(items=list(items))
            items[:] = []
            return

        kept = [it for it in items if assignment[_group_key(it)] == shard_id]
        _write_shard_map(config, kept, groups, assignment, source, weights_table, fallback)
        my_groups = sorted(k for k, s in assignment.items() if s == shard_id)
        print(
            "[finn-sharding] shard %d/%d: %d item(s) across %d group(s)"
            % (shard_id, num_shards, len(kept), len(my_groups))
        )
        if my_groups:
            sample = my_groups[:5]
            ellipsis = " ..." if len(my_groups) > 5 else ""
            print("[finn-sharding] groups: %s%s" % (sample, ellipsis))
        items[:] = kept

    def pytest_sessionstart(self, session):
        self.timings = None
        if session.config.getoption("--num-shards") <= 0:
            return
        if _is_xdist_worker(session.config):
            return
        self.timings = {
            "per_test_seconds": {},
            "nodeid_to_group": {},
            "start_monotonic": time.monotonic(),
        }

    def pytest_collection_finish(self, session):
        if self.timings is None:
            return
        for item in session.items:
            for mark in item.iter_markers(name="xdist_group"):
                if mark.args:
                    self.timings["nodeid_to_group"][item.nodeid] = str(mark.args[0])
                    break
                name = mark.kwargs.get("name")
                if name is not None:
                    self.timings["nodeid_to_group"][item.nodeid] = str(name)
                    break

    def pytest_runtest_logreport(self, report):
        if self.timings is None:
            return
        # sum setup+call+teardown per nodeid. xdist forwards worker reports here.
        duration = float(getattr(report, "duration", 0.0) or 0.0)
        self.timings["per_test_seconds"][report.nodeid] = (
            self.timings["per_test_seconds"].get(report.nodeid, 0.0) + duration
        )

    def pytest_sessionfinish(self, session, exitstatus):
        # --dry-run-shards deselects everything by design, so exit 5 ("no tests
        # collected") is benign. For a real sharded run, exit 5 can also happen
        # when this shard's slice is empty after group assignment. Remap it to
        # 0, but print a warning so an unbalanced split is visible.
        num_shards = session.config.getoption("--num-shards")
        if exitstatus == 5:
            if session.config.getoption("--dry-run-shards"):
                session.exitstatus = 0
            elif num_shards > 0:
                shard_id = session.config.getoption("--shard-id")
                # Sidecar lets a build-level aggregator detect the empty shard
                # without grepping stdout. Best-effort: an OSError here must not
                # change exit status.
                out_dir, stash, _ = _junit_output_info(session.config)
                # only the controller writes the sidecar: workers share out_dir,
                # and a worker that drew 0 of the shard's groups is not the shard
                # itself being empty.
                if out_dir and stash and not _is_xdist_worker(session.config):
                    try:
                        with open(os.path.join(out_dir, stash + ".empty-shard"), "w") as f:
                            f.write("shard %d/%d collected 0 items\n" % (shard_id, num_shards))
                    except OSError:
                        pass
                print(
                    "[finn-sharding] WARNING: shard %d/%d collected 0 items after assignment"
                    % (shard_id, num_shards)
                )
                session.exitstatus = 0
        if self.timings is None:
            return
        # an interrupted/aborted shard has truncated timings, so skip the
        # sidecar: a killed or timed-out run must not feed the persistent
        # master. a plain test failure (exit 1) still has real durations.
        if session.exitstatus in (
            pytest.ExitCode.INTERRUPTED,
            pytest.ExitCode.INTERNAL_ERROR,
            pytest.ExitCode.USAGE_ERROR,
        ):
            return
        out_dir, stash, junitxml = _junit_output_info(session.config)
        if not junitxml:
            return
        out_path = os.path.join(out_dir, stash + ".timings.json")
        groups = {}
        for nodeid, duration in self.timings["per_test_seconds"].items():
            # under --dist loadgroup the forwarded report nodeid gains an
            # "@<group>" suffix, so it misses the plain-nodeid keys in
            # nodeid_to_group. canonical_key recovers the group from the suffix
            # so a group's members sum together per shard, not as singletons.
            group = self.timings["nodeid_to_group"].get(nodeid) or sharding.canonical_key(nodeid)
            entry = groups.setdefault(group, {"name": group, "count": 0, "seconds": 0.0})
            entry["count"] += 1
            entry["seconds"] += duration
        payload = {
            "stash": stash,
            "shard": {
                "num": session.config.getoption("--num-shards"),
                "id": session.config.getoption("--shard-id"),
            },
            "metadata": {
                "job": os.environ.get("FINN_CI_JOB_NAME"),
                "build": os.environ.get("FINN_CI_BUILD_NUMBER"),
                "stage": os.environ.get("FINN_CI_STAGE"),
                "timings_file": os.environ.get("FINN_CI_TIMINGS_FILE"),
            },
            "wall_seconds": time.monotonic() - self.timings["start_monotonic"],
            "groups": sorted(groups.values(), key=lambda g: -g["seconds"]),
        }
        try:
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)
        except OSError:
            pass
