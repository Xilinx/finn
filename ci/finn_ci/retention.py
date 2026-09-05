# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Retention and pruning for the shared CI image, artifact and snapshot trees.

One numbered-tree rotation core (_prune_numbered) backs the image, artifact and
snapshot trees. They differ only in the file lister and the remover. RETENTION
holds the per-tree window so a caller never restates the policy. Deletions
tolerate concurrent removal on a shared NFS parent.
"""

import errno
import os
import re
import shutil
import time

# Per-tree retention for the "prune --kind {image,artifact,snapshot}" CLI
# subcommand. Artifacts are the per-board fallback used when a board's most
# recent build regresses, so they are kept deep. Snapshots are the small
# per-build timing inputs, so a shallow window is enough.
RETENTION = {
    "image": {"retain": 3, "ageDays": 14},
    "artifact": {"retain": 30, "ageDays": 30},
    "snapshot": {"retain": 3, "ageDays": 2},
}


def robust_rmtree(path, retries=6, initial_delay=0.1, backoff=2.0):
    """remove a directory tree with retries for transient NFS cleanup races.

    mirror of finn.util.basic.robust_rmtree so ci/ stays importable on
    bare agents that have no finn package.
    """
    if not path or not os.path.exists(path):
        return
    delay = initial_delay
    for attempt in range(retries):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            if exc.errno not in (errno.ENOTEMPTY, errno.EBUSY) or attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= backoff


SNAPSHOT_FILE_RE = re.compile(r"^build_(\d+)_timings_input\.json$")


def _coerce_current_build(value, tag):
    try:
        return int(str(value))
    except (TypeError, ValueError):
        raise ValueError(
            "ci_sharding %s: current_build must be an integer-like string, got %r" % (tag, value)
        )


def _list_numeric_dirs(parent):
    """Return (build, name) for digit-named subdirectories of parent."""
    return [(int(d), d) for d in os.listdir(parent) if d.isdigit()]


def _list_snapshot_files(parent):
    """Return (build, name) for build_<N>_timings_input.json files."""
    out = []
    for name in os.listdir(parent):
        m = SNAPSHOT_FILE_RE.match(name)
        if m:
            out.append((int(m.group(1)), name))
    return out


def _prune_numbered(
    parent,
    current_build,
    retain_n,
    max_age_days,
    dry_run,
    *,
    tag,
    list_entries=_list_numeric_dirs,
    remove=robust_rmtree,
):
    """Delete build-numbered entries of parent outside the newest retain_n.

    list_entries maps the parent to (build, name) pairs and remove deletes
    one path. Both default to the numeric build-dir tree (rmtree), and
    prune_snapshots overrides them for the snapshot files (unlink).

    The newest retain_n builds and current_build are always kept. An older
    entry is removed only once it is past max_age_days. Concurrent deletion on
    a shared NFS parent is tolerated at both probe sites: an entry that
    vanishes during the age check or the remove is treated as already-pruned.
    Returns the number matched.
    """
    retain_n = int(retain_n)
    max_age_days = int(max_age_days)
    if retain_n < 1:
        raise ValueError("retain_n must be >= 1")
    if max_age_days < 0:
        raise ValueError("max_age_days must be >= 0")
    current_build_int = _coerce_current_build(current_build, tag)
    if not os.path.isdir(parent):
        print("ci_sharding %s: %s not present, skipping" % (tag, parent))
        return 0
    cutoff = time.time() - (max_age_days * 24 * 60 * 60)
    # Compare by int so an on-disk "0123" matches a BUILD_NUMBER of "123".
    entries = sorted(list_entries(parent))
    keep = {build for build, _ in entries[-retain_n:]}
    keep.add(current_build_int)
    matched = 0
    for build, name in entries:
        if build in keep:
            continue
        path = os.path.join(parent, name)
        if max_age_days > 0:
            try:
                if os.path.getmtime(path) >= cutoff:
                    continue
            except FileNotFoundError:
                continue
        matched += 1
        if dry_run:
            print("ci_sharding %s: would delete %s" % (tag, path))
        else:
            print("ci_sharding %s: deleting %s" % (tag, path))
            try:
                remove(path)
            except FileNotFoundError:
                pass
    print(
        "ci_sharding %s: done (parent=%s current=%s retain_n=%s "
        "max_age_days=%s dry_run=%s matched=%d)"
        % (tag, parent, current_build_int, retain_n, max_age_days, int(dry_run), matched)
    )
    return matched


def prune_images(shared_dir, job_key, current_build, retain_n, max_age_days, dry_run=False):
    parent = os.path.join(shared_dir, job_key)
    return _prune_numbered(
        parent, current_build, retain_n, max_age_days, dry_run, tag="prune-images"
    )


def prune_artifacts(artifact_dir, job_key, current_build, retain_n, max_age_days, dry_run=False):
    """Rotate ${FINN_CI_NFS_ROOT}/artifacts/ci_runs/<job_key>/ for this build job.

    Keeps the newest retain_n builds plus any younger than max_age_days. HW
    resolves per board to the newest READY zip, so pruning is safe as long as
    a board's most recent READY still falls inside the retained window. The
    window is kept deep for that reason (see RETENTION["artifact"]).
    """
    parent = os.path.join(artifact_dir, "ci_runs", job_key)
    return _prune_numbered(
        parent, current_build, retain_n, max_age_days, dry_run, tag="prune-artifacts"
    )


def prune_snapshots(state_root, job_key, current_build, retain_n, max_age_days, dry_run=False):
    """Rotate per-build timing snapshot files under _ci_state/<job_key>/.

    The snapshots are named build_<N>_timings_input.json and live alongside
    the persistent ci_timings_master.json, which is left untouched. Only the
    build-numbered files are eligible. This shares the numbered-rotation core
    with the image and artifact trees, differing only in the file lister and
    the os.unlink remover.
    """
    parent = os.path.join(state_root, job_key)
    return _prune_numbered(
        parent,
        current_build,
        retain_n,
        max_age_days,
        dry_run,
        tag="prune-snapshots",
        list_entries=_list_snapshot_files,
        remove=os.unlink,
    )


def prune_pip_cache(root, keep, max_age_days, dry_run=False):
    """Delete cache-key subdirs of root older than max_age_days.

    The directory pointed at by keep is always retained. A subdir's own mtime
    is the age key, so an actively reused cache dir is bumped on write and
    survives. Returns the number matched.
    """
    max_age_days = int(max_age_days)
    if max_age_days < 0:
        raise ValueError("max_age_days must be >= 0")
    if not os.path.isdir(root):
        return 0
    keep_abs = os.path.abspath(keep) if keep else None
    cutoff = time.time() - (max_age_days * 24 * 60 * 60)
    matched = 0
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        if keep_abs and os.path.abspath(path) == keep_abs:
            continue
        if max_age_days > 0:
            try:
                if os.path.getmtime(path) >= cutoff:
                    continue
            except FileNotFoundError:
                continue
        matched += 1
        if dry_run:
            print("ci_sharding prune-pip-cache: would delete %s" % path)
        else:
            print("ci_sharding prune-pip-cache: deleting %s" % path)
            robust_rmtree(path)
    return matched
