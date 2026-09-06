# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic weight-balanced group-to-shard assignment.

Reads per-group weights from a timing state file (written by the timing
master) and packs groups into shards by longest-processing-time-first greedy
bin packing, degrading to round-robin when there is no timing signal.
"""

import os
import re
from finn_ci import jsonio

# Internal regexes for canonical_key. The group suffix is the "@<group>" that
# xdist's loadgroup appends to a forwarded nodeid. Anchor on a token that stops
# at a "]" or whitespace so a parametrised id such as test[user@host] is not
# mistaken for a group suffix.
GROUP_SUFFIX_RE = re.compile(r"@([^\]\s]+)$")
NOTEBOOK_PARAM_RE = re.compile(r"\[(/[^\]]+\.ipynb)\]")


def canonical_key(name):
    """Normalise timing group names across xdist and workspace-specific paths."""
    m = GROUP_SUFFIX_RE.search(name)
    if m:
        return m.group(1)
    m2 = NOTEBOOK_PARAM_RE.search(name)
    if m2:
        path = m2.group(1)
        return name.replace(m2.group(0), "[%s]" % os.path.basename(path))
    return name


def _samples_from_entry(entry):
    """Return the samples list from a master entry (empty if absent or malformed)."""
    if not isinstance(entry, dict):
        return []
    samples = entry.get("samples")
    if not isinstance(samples, list):
        return []
    return [float(s) for s in samples if s is not None]


def _window_weight(values):
    """Max of the positive values: the conservative per-group weight.

    Returns 0.0 on an empty input.
    """
    return max((v for v in values if v > 0.0), default=0.0)


def _entry_weight_seconds(entry):
    """Extract the weight (window max) from a master/snapshot group entry."""
    return _window_weight(_samples_from_entry(entry))


def load_group_weights(path):
    """Return {group_name: seconds} from a timing state file."""
    data = jsonio.read_json(path, default={})
    groups = data.get("groups") if isinstance(data, dict) else None
    if not isinstance(groups, dict):
        return {}
    weights = {}
    for key, val in groups.items():
        seconds = _entry_weight_seconds(val)
        if seconds > 0.0:
            weights[str(key)] = seconds
    return weights


def fallback_weight(weights):
    # weight used for groups with no timing data: the median of the known
    # group weights, or 1.0 at cold-start so a brand-new master still
    # produces a reproducible round-robin assignment.
    known = sorted(w for w in weights.values() if w > 0.0)
    return known[len(known) // 2] if known else 1.0


def assign_groups_to_shards(group_keys, num_shards, weights=None, pins=None):
    """Assign group keys to shard ids.

    Pinned groups are placed first. Unpinned groups are packed by
    longest-processing-time-first greedy bin packing when timing data is
    available, otherwise by round-robin over the sorted group keys so the
    fallback placement is reproducible and balanced by group count.

    Raises ValueError for num_shards < 1 or any pin outside [0, num_shards).
    """
    num_shards = int(num_shards)
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1, got %r" % (num_shards,))
    pins = pins or {}
    for key, shard in pins.items():
        try:
            shard_int = int(shard)
        except (TypeError, ValueError):
            raise ValueError("pin for %r must be an int, got %r" % (key, shard))
        if not 0 <= shard_int < num_shards:
            raise ValueError(
                "pin for %r out of range: %d not in [0, %d)" % (key, shard_int, num_shards)
            )
    group_keys = sorted(str(k) for k in group_keys)
    weights = weights or {}
    assignment = {}
    source = {}
    shard_load = [0.0] * num_shards
    fallback = fallback_weight(weights)

    for key in group_keys:
        if key in pins:
            shard = int(pins[key])
            assignment[key] = shard
            source[key] = "pinned"
            shard_load[shard] += weights.get(key, fallback)

    if num_shards <= 1:
        for key in group_keys:
            assignment.setdefault(key, 0)
            source.setdefault(key, "single")
        return assignment, source, shard_load, fallback

    unpinned = [k for k in group_keys if k not in assignment]
    has_signal = any(weights.get(k, 0.0) > 0.0 for k in unpinned)

    if not has_signal:
        # No timing signal: balance by group count, but seed from any pinned
        # load so pins are respected. Placing each group on the least-loaded
        # shard (ties broken by index) stays deterministic and matches plain
        # round-robin when there are no pins.
        for key in unpinned:
            shard = min(range(num_shards), key=lambda s: (shard_load[s], s))
            assignment[key] = shard
            source[key] = "round_robin"
            shard_load[shard] += 1.0
        return assignment, source, shard_load, fallback

    unpinned.sort(key=lambda k: (-weights.get(k, fallback), k))
    for key in unpinned:
        weight = weights.get(key, fallback)
        shard = min(range(num_shards), key=lambda s: (shard_load[s], s))
        assignment[key] = shard
        source[key] = "known" if key in weights else "fallback"
        shard_load[shard] += weight
    return assignment, source, shard_load, fallback
