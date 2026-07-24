# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import os
from finn_ci import __main__ as cli
from finn_ci import retention

pytestmark = pytest.mark.util


@pytest.mark.parametrize(
    "prune_fn, rel_parent",
    [
        (retention.prune_images, "job"),
        (retention.prune_artifacts, "ci_runs/job"),
    ],
)
def test_prune_keeps_current_build_and_newest(tmp_path, prune_fn, rel_parent):
    # prune_images and prune_artifacts share the numbered-rotation core, and
    # only the parent path differs (<root>/job vs <root>/ci_runs/job).
    parent = tmp_path.joinpath(*rel_parent.split("/"))
    for build in ("1", "2", "3", "4"):
        path = parent / build
        path.mkdir(parents=True)
        os.utime(str(path), (1, 1))

    prune_fn(str(tmp_path), "job", "5", retain_n=2, max_age_days=0)

    assert not (parent / "1").exists()
    assert not (parent / "2").exists()
    assert (parent / "3").exists()
    assert (parent / "4").exists()


def test_prune_images_skips_when_parent_missing(tmp_path, capsys):
    rc = retention.prune_images(str(tmp_path / "absent"), "job", "1", 1, 0)
    captured = capsys.readouterr()
    assert rc == 0
    assert "not present, skipping" in captured.out


def test_prune_numbered_dry_run_matches_real_run_count(tmp_path):
    parent = tmp_path / "p"
    parent.mkdir()
    for build in ("1", "2", "3", "4"):
        (parent / build).mkdir()
        os.utime(str(parent / build), (1, 1))

    dry = retention._prune_numbered(
        str(parent),
        current_build="5",
        retain_n=1,
        max_age_days=0,
        dry_run=True,
        tag="t",
    )
    assert dry == 3
    assert sorted(p.name for p in parent.iterdir()) == ["1", "2", "3", "4"]

    real = retention._prune_numbered(
        str(parent),
        current_build="5",
        retain_n=1,
        max_age_days=0,
        dry_run=False,
        tag="t",
    )
    assert real == 3
    assert sorted(p.name for p in parent.iterdir()) == ["4"]


def test_prune_numbered_rejects_non_numeric_current(tmp_path):
    # off-Jenkins CLI invocations or a broken BUILD_NUMBER env must not
    # silently degrade retention to "newest N" by passing a string the
    # numeric-only sibling filter can never match.
    parent = tmp_path / "p"
    parent.mkdir()
    for build in ("1", "2"):
        (parent / build).mkdir()
    with pytest.raises(ValueError, match="current_build must be an integer-like string"):
        retention._prune_numbered(
            str(parent),
            current_build="not-a-number",
            retain_n=1,
            max_age_days=0,
            dry_run=True,
            tag="t",
        )
    with pytest.raises(ValueError, match="current_build must be an integer-like string"):
        retention._prune_numbered(
            str(parent),
            current_build=None,
            retain_n=1,
            max_age_days=0,
            dry_run=True,
            tag="t",
        )


def test_prune_numbered_canonicalises_leading_zeros(tmp_path):
    # On-disk dir name "0123" and a BUILD_NUMBER value of "123" refer to the
    # same build for retention purposes. Without canonicalisation the keep set
    # contained the BUILD_NUMBER as-is and the on-disk leading-zero variant
    # would be eligible for pruning even though it is the current build.
    parent = tmp_path / "p"
    parent.mkdir()
    for build in ("0123", "0124", "0125"):
        (parent / build).mkdir()
        os.utime(str(parent / build), (1, 1))

    matched = retention._prune_numbered(
        str(parent),
        current_build="123",
        retain_n=1,
        max_age_days=0,
        dry_run=False,
        tag="t",
    )
    # newest ("0125") kept by retain_n, current build ("0123" via int(123))
    # kept by the current-build guard. "0124" is the only one pruned.
    assert matched == 1
    surviving = sorted(p.name for p in parent.iterdir())
    assert surviving == ["0123", "0125"]


def test_prune_numbered_tolerates_concurrent_delete(tmp_path):
    parent = tmp_path / "p"
    parent.mkdir()
    for build in ("1", "2", "3"):
        (parent / build).mkdir()
        os.utime(str(parent / build), (1, 1))

    real_rmtree = retention.robust_rmtree
    state = {"first": True}

    def flaky_rmtree(path, *args, **kwargs):
        # simulate another CI run pruning '1' between our listdir and rmtree
        if state["first"]:
            state["first"] = False
            raise FileNotFoundError(path)
        return real_rmtree(path, *args, **kwargs)

    matched = retention._prune_numbered(
        str(parent),
        current_build="5",
        retain_n=1,
        max_age_days=0,
        dry_run=False,
        tag="t",
        remove=flaky_rmtree,
    )
    assert matched == 2
    # build '3' is kept (retain_n=1, newest), '2' got rmtreed for real,
    # '1' was the simulated race victim and we tolerated it
    surviving = sorted(p.name for p in parent.iterdir())
    assert "3" in surviving
    assert "2" not in surviving


def test_prune_numbered_tolerates_concurrent_delete_in_age_check(tmp_path, monkeypatch):
    parent = tmp_path / "p"
    parent.mkdir()
    # retain_n=1 keeps the newest ('3'). '1' and '2' are both deletion
    # candidates. age cutoff is in the past so both qualify on mtime.
    for build in ("1", "2", "3"):
        (parent / build).mkdir()
        os.utime(str(parent / build), (1, 1))

    real_getmtime = retention.os.path.getmtime

    def flaky_getmtime(path):
        if path.endswith("/1"):
            raise FileNotFoundError(path)
        return real_getmtime(path)

    monkeypatch.setattr(retention.os.path, "getmtime", flaky_getmtime)
    matched = retention._prune_numbered(
        str(parent),
        current_build="9",
        retain_n=1,
        max_age_days=7,
        dry_run=False,
        tag="t",
    )
    # '1' raised FileNotFoundError during the age probe so the loop treats it
    # as already-pruned and does not count it. '2' was processed normally and
    # deleted. The point is that the FileNotFoundError on '1' did not abort
    # the loop and leave '2' behind.
    assert matched == 1
    assert "2" not in [p.name for p in parent.iterdir()]
    assert "3" in [p.name for p in parent.iterdir()]


def test_prune_snapshots_keeps_current_build_and_newest(tmp_path):
    state_root = tmp_path / "state"
    job = "finn"
    (state_root / job).mkdir(parents=True)
    for n in (1, 2, 3, 4, 5):
        (state_root / job / ("build_%d_timings_input.json" % n)).write_text("{}")
    # Non-numbered files (the master itself, corrupt backups) must be left
    # alone even when they sort lexicographically alongside snapshots.
    (state_root / job / "ci_timings_master.json").write_text("{}")
    (state_root / job / "ci_timings_master.json.corrupt-1").write_text("{}")
    retention.prune_snapshots(str(state_root), job, current_build="3", retain_n=2, max_age_days=0)
    remaining = sorted(p.name for p in (state_root / job).iterdir())
    assert "build_3_timings_input.json" in remaining
    assert "build_4_timings_input.json" in remaining
    assert "build_5_timings_input.json" in remaining
    assert "ci_timings_master.json" in remaining
    assert "ci_timings_master.json.corrupt-1" in remaining
    assert "build_1_timings_input.json" not in remaining
    assert "build_2_timings_input.json" not in remaining


def test_prune_snapshots_skips_when_parent_missing(tmp_path, capsys):
    rc = retention.prune_snapshots(
        str(tmp_path / "nope"), "finn", current_build="1", retain_n=2, max_age_days=0
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "not present, skipping" in captured.out


def test_prune_snapshots_rejects_non_numeric_current(tmp_path):
    with pytest.raises(ValueError, match="prune-snapshots: current_build must be"):
        retention.prune_snapshots(
            str(tmp_path), "finn", current_build="x", retain_n=1, max_age_days=0
        )


def test_prune_snapshots_honours_age_gating(tmp_path):
    state_root = tmp_path / "state"
    job = "finn"
    parent = state_root / job
    parent.mkdir(parents=True)
    old = parent / "build_1_timings_input.json"
    fresh = parent / "build_2_timings_input.json"
    old.write_text("{}")
    fresh.write_text("{}")
    os.utime(str(old), (1, 1))

    retention.prune_snapshots(str(state_root), job, current_build="3", retain_n=1, max_age_days=1)

    assert not old.exists()
    assert fresh.exists()


def test_prune_snapshots_tolerates_concurrent_delete_on_unlink(tmp_path, monkeypatch):
    state_root = tmp_path / "state"
    job = "finn"
    parent = state_root / job
    parent.mkdir(parents=True)
    for n in (1, 2, 3):
        path = parent / ("build_%d_timings_input.json" % n)
        path.write_text("{}")
        os.utime(str(path), (1, 1))
    real_unlink = retention.os.unlink
    state = {"first": True}

    def flaky_unlink(path):
        if state["first"]:
            state["first"] = False
            raise FileNotFoundError(path)
        return real_unlink(path)

    monkeypatch.setattr(retention.os, "unlink", flaky_unlink)
    retention.prune_snapshots(str(state_root), job, current_build="9", retain_n=1, max_age_days=0)

    assert not (parent / "build_2_timings_input.json").exists()
    assert (parent / "build_3_timings_input.json").exists()


def test_prune_snapshots_dry_run_does_not_delete(tmp_path):
    state_root = tmp_path / "state"
    job = "finn"
    (state_root / job).mkdir(parents=True)
    for n in (1, 2, 3):
        (state_root / job / ("build_%d_timings_input.json" % n)).write_text("{}")
    retention.prune_snapshots(
        str(state_root), job, current_build="3", retain_n=1, max_age_days=0, dry_run=True
    )
    remaining = sorted(p.name for p in (state_root / job).iterdir())
    assert remaining == [
        "build_1_timings_input.json",
        "build_2_timings_input.json",
        "build_3_timings_input.json",
    ]


def test_prune_snapshots_cli_smoke(tmp_path):
    state_root = tmp_path / "state"
    job = "finn"
    (state_root / job).mkdir(parents=True)
    (state_root / job / "build_1_timings_input.json").write_text("{}")
    rc = cli.main(["prune", "--kind", "snapshot", str(state_root), job, "1", "--dry-run"])
    assert rc == 0


def test_prune_cli_kind_image_reads_retention_window(tmp_path):
    # the prune CLI looks up retain_n/max_age_days from RETENTION[kind] so the
    # caller never passes a window that disagrees with the documented policy.
    parent = tmp_path / "job"
    for build in ("1", "2", "3", "4", "5"):
        path = parent / build
        path.mkdir(parents=True)
        os.utime(str(path), (1, 1))  # ancient, so the age gate never holds them
    rc = cli.main(["prune", "--kind", "image", str(tmp_path), "job", "9"])
    assert rc == 0
    # RETENTION["image"]["retain"] == 3 keeps the newest three (3, 4, 5).
    # current build 9 is absent on disk, so 1 and 2 are the only ones pruned.
    assert sorted(p.name for p in parent.iterdir()) == ["3", "4", "5"]


def test_prune_cli_rejects_unknown_kind(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["prune", "--kind", "bogus", str(tmp_path), "job", "1"])


def test_prune_cli_sanitises_job_key(tmp_path):
    # the prune root is os.path.join(root, job_key(JOB_NAME)). a JOB_NAME of
    # ".." must collapse to "job" so the destructive rmtree cannot climb out of
    # root. without sanitisation the parent would resolve to root's parent.
    root = tmp_path / "images"
    job_parent = root / "job"
    for build in ("1", "2", "3", "4", "5"):
        path = job_parent / build
        path.mkdir(parents=True)
        os.utime(str(path), (1, 1))
    sibling = root / "sibling"
    sibling.mkdir()
    rc = cli.main(["prune", "--kind", "image", str(root), "..", "9"])
    assert rc == 0
    # RETENTION["image"]["retain"] == 3 keeps the newest three under root/job,
    # and the ".." never escaped to touch the sibling tree.
    assert sorted(p.name for p in job_parent.iterdir()) == ["3", "4", "5"]
    assert sibling.exists()


def test_prune_pip_cache_keeps_current_and_fresh(tmp_path):
    root = tmp_path / "pip"
    keep = root / "keepme"
    old = root / "stale"
    fresh = root / "recent"
    for d in (keep, old, fresh):
        d.mkdir(parents=True)
    os.utime(str(old), (1, 1))
    matched = retention.prune_pip_cache(str(root), str(keep), max_age_days=14)
    # 'old' (ancient mtime) is pruned, 'keep' is excluded, 'recent' is fresh
    assert matched == 1
    assert not old.exists()
    assert keep.exists()
    assert fresh.exists()


def test_prune_pip_cache_dry_run_deletes_nothing(tmp_path):
    root = tmp_path / "pip"
    old = root / "stale"
    old.mkdir(parents=True)
    os.utime(str(old), (1, 1))
    matched = retention.prune_pip_cache(str(root), "", max_age_days=14, dry_run=True)
    assert matched == 1
    assert old.exists()


def test_prune_pip_cache_missing_root_is_noop(tmp_path):
    assert retention.prune_pip_cache(str(tmp_path / "absent"), "", 14) == 0


def test_prune_numbered_rejects_negative_max_age_days(tmp_path):
    # A negative window must fail loudly rather than silently disabling the age
    # gate (which would delete every eligible entry regardless of age).
    parent = tmp_path / "p"
    parent.mkdir()
    (parent / "1").mkdir()
    with pytest.raises(ValueError, match="max_age_days must be >= 0"):
        retention._prune_numbered(
            str(parent), current_build="9", retain_n=1, max_age_days=-1, dry_run=True, tag="t"
        )


def test_prune_pip_cache_rejects_negative_max_age_days(tmp_path):
    root = tmp_path / "pip"
    root.mkdir()
    with pytest.raises(ValueError, match="max_age_days must be >= 0"):
        retention.prune_pip_cache(str(root), "", max_age_days=-1)
