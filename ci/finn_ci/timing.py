# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Self-maintaining per-group timing master and per-shard report summaries.

The persistent master at
${FINN_CI_NFS_ROOT}/_ci_state/<jobKey>/ci_timings_master.json holds the last
MAX_SAMPLES observations per group. update_master appends a build's
observations. Persisting them to the master is opt-in (the caller persists for
builds that ran to a normal end). The bin packer (finn_ci.sharding) weights
each group by the window max. This module also merges the per-shard shard-map
sidecars and prints the per-shard wall-clock summary used for triage.
"""

import collections
import glob
import os
import re
import sys
import time
from finn_ci import jsonio, sharding

# Per-group rolling window for the timing master. The bin packer reads the
# max of the window, meaning each group is weighted by its slowest recent run.
MAX_SAMPLES = 5

# summarize-timings flags shards exceeding this multiple of the family median.
SLOW_FACTOR = 1.5


# =============================================================================
# Reports I/O (merge maps, per-shard summary)
# =============================================================================


def load_map_rows(path):
    data = jsonio.read_json(path, default=[])
    if isinstance(data, list):
        return data
    return []


def merge_maps(reports_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(reports_dir, "*.shardmap.json"))):
        rows.extend(load_map_rows(path))
    rows.sort(
        key=lambda r: (
            str(r.get("stage", "")),
            int(r.get("shard_id", 0)),
            str(r.get("nodeid", "")),
        )
    )
    json_path = os.path.join(reports_dir, "shard_map.json")
    txt_path = os.path.join(reports_dir, "shard_map.txt")
    jsonio.write_json_atomic(json_path, rows)
    with open(txt_path, "w") as f:
        for row in rows:
            f.write(
                "nodeid={nodeid} stage={stage} shard={shard_num}/{shard_count} "
                "stash={stash} group={group} weight_s={weight_s:.3f} source={source}\n".format(
                    nodeid=row.get("nodeid", ""),
                    stage=row.get("stage", ""),
                    shard_num=int(row.get("shard_id", 0)) + 1,
                    shard_count=int(row.get("num_shards", 1)),
                    stash=row.get("stash", ""),
                    group=row.get("group", ""),
                    weight_s=float(row.get("weight_s", 0.0) or 0.0),
                    source=row.get("source", ""),
                )
            )
    print("ci_sharding merge-maps: wrote %d row(s)" % len(rows))
    return 0


def timing_rows(reports_dir):
    rows = []
    pattern = os.path.join(reports_dir, "*.timings.json")
    for path in sorted(glob.glob(pattern)):
        data = jsonio.read_json(path, default={})
        if not isinstance(data, dict):
            print("ci_sharding summarize: could not parse %s" % path, file=sys.stderr)
            continue
        stash = data.get("stash") or os.path.basename(path).split(".")[0]
        groups = data.get("groups") or []
        top = groups[0] if groups else {"name": "(none)", "seconds": 0.0}
        rows.append(
            (
                stash,
                int(data.get("shard", {}).get("id", 0)),
                float(data.get("wall_seconds", 0.0) or 0.0),
                float(top.get("seconds", 0.0) or 0.0),
                str(top.get("name", "")),
            )
        )
    return rows


def family(stash):
    return re.sub(r"_\d+$", "", stash)


def summarize_timings(reports_dir):
    rows = timing_rows(reports_dir)
    if not rows:
        print("ci_sharding summarize: no parseable timings.json files in %s" % reports_dir)
        return 0
    by_family = collections.defaultdict(list)
    for row in rows:
        by_family[family(row[0])].append(row)
    print()
    print("=== per-shard wall-clock ===")
    print("%-36s %3s %10s %12s  %s" % ("stash", "id", "wall_s", "max_group_s", "max_group"))
    print("-" * 100)
    slow_found = False
    for fam in sorted(by_family):
        fam_rows = sorted(by_family[fam], key=lambda r: r[1])
        walls = sorted(r[2] for r in fam_rows)
        median = walls[len(walls) // 2] if walls else 0.0
        for stash, sid, wall, mx_sec, mx_name in fam_rows:
            flag = ""
            if median > 0.0 and wall > SLOW_FACTOR * median:
                flag = "  <<< SLOW SHARD (%.1fx median)" % (wall / median)
                slow_found = True
            print("%-36s %3d %10.1f %12.1f  %s%s" % (stash, sid, wall, mx_sec, mx_name, flag))
        print()
    if slow_found:
        print(
            "ci_sharding summarize: one or more shards exceeded %.1fx family median. "
            "A trusted full build refreshes the timing master from these observations."
            % SLOW_FACTOR
        )
    return 0


# =============================================================================
# Timing master state machine
# =============================================================================
# Schema v1:
#
#   {"schema_version": 1, "updated_at": str, "last_update": {...},
#    "groups": {<name>: {"samples": [s1, ..., sMAX_SAMPLES]}}}
#
# Bump SCHEMA_VERSION when the master layout changes incompatibly: an
# unrecognised version is discarded and the timings cold-start, which one
# non-aborted build repopulates.

SCHEMA_VERSION = 1


def normalise_master(data):
    """Coerce arbitrary input to the master schema (drops unknown top-level keys)."""
    if not isinstance(data, dict):
        data = {}
    schema_version = data.get("schema_version")
    if schema_version is not None and schema_version != SCHEMA_VERSION:
        print(
            "ci_sharding normalise_master: unrecognised schema_version %r, "
            "treating as empty (expected %d)" % (schema_version, SCHEMA_VERSION),
            file=sys.stderr,
        )
        data = {}
    groups = data.get("groups")
    if not isinstance(groups, dict):
        groups = {}
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at": data.get("updated_at"),
        "groups": dict(groups),
    }


def observed_groups_from_reports(reports_dir):
    """Return {group_name: max_seconds} over this build's timing sidecars."""
    observed = {}
    for path in sorted(glob.glob(os.path.join(reports_dir, "*.timings.json"))):
        data = jsonio.read_json(path, default={})
        if not isinstance(data, dict):
            continue
        for entry in data.get("groups") or []:
            name = sharding.canonical_key(str(entry.get("name", "")))
            if not name:
                continue
            try:
                seconds = float(entry.get("seconds", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if seconds > observed.get(name, 0.0):
                observed[name] = round(seconds, 3)
    return observed


def _apply_per_group_update(observed_seconds, current_entry):
    """Append the observation to current_entry's window, trimmed to MAX_SAMPLES."""
    prior_samples = sharding._samples_from_entry(current_entry)
    new_samples = (prior_samples + [round(float(observed_seconds), 3)])[-MAX_SAMPLES:]
    return {"samples": new_samples}


def update_master(reports_dir, master_path, out_path, update_persistent=False, metadata=None):
    """Merge observed timings into a per-build preview and optionally the master.

    Every call writes out_path. Updating the persistent master is opt-in via
    update_persistent, which the caller passes for any non-aborted build.
    Preview mode (update_persistent off) leaves the on-disk master untouched.
    Either way, every observation in this build is appended to its group's
    samples and the window is trimmed to MAX_SAMPLES.
    """
    observed_seconds = observed_groups_from_reports(reports_dir)
    metadata = metadata or {}
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def apply(current, persist=False):
        master = normalise_master(current)
        master["updated_at"] = now_iso
        for name, seconds in observed_seconds.items():
            master["groups"][name] = _apply_per_group_update(seconds, master["groups"].get(name))
        master["last_update"] = {
            "job": metadata.get("job"),
            "build": metadata.get("build"),
            "persistent_update": bool(persist),
            "observed_groups": len(observed_seconds),
        }
        return master

    persistent_updated = False
    if master_path and update_persistent:
        # No cross-agent lock. write_json_atomic renames into place, so a
        # reader never sees a half-written file and overlapping writers settle
        # to last-writer-wins. A sample dropped by a concurrent
        # read-modify-write is re-added by the next build that observes the
        # group, and since the bin packer only reads the window max, an
        # occasional missing sample makes no practical difference.
        master = apply(jsonio.read_json(master_path, default={}), persist=True)
        jsonio.write_json_atomic(master_path, master)
        persistent_updated = True
    elif master_path:
        master = apply(jsonio.read_json(master_path, default={}), persist=False)
    else:
        master = apply({}, persist=False)
    if out_path:
        jsonio.write_json_atomic(out_path, master)
    print(
        "ci_sharding update: %d observed, %d in master, persistent_update=%s"
        % (len(observed_seconds), len(master.get("groups", {})), persistent_updated)
    )
    return 0


def prepare_timing_snapshot(master_path, snapshot_path):
    """Copy the persistent master to a per-build snapshot for shard consumption.

    Cold start writes an empty snapshot so sharding falls back to
    deterministic round-robin until the first build populates the master.
    """
    master = jsonio.read_json(master_path, default=None)
    master = normalise_master(master)
    jsonio.write_json_atomic(snapshot_path, master)
    print(
        "ci_sharding prepare: wrote %s with %d group(s)"
        % (snapshot_path, len(master.get("groups", {})))
    )
    return 0
