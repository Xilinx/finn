# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import json
from finn_ci import jsonio, sharding, timing

pytestmark = pytest.mark.util


def write_json(path, payload):
    path.write_text(json.dumps(payload))


def _seed_master_with_group(path, name, samples, **extra):
    write_json(
        path,
        {
            "schema_version": timing.SCHEMA_VERSION,
            "groups": {name: {"samples": list(samples), **extra}},
        },
    )


def _write_observation(reports_dir, stash, name, seconds):
    write_json(
        reports_dir / ("%s.timings.json" % stash),
        {
            "stash": stash,
            "metadata": {"job": "j", "build": "1", "stage": stash},
            "groups": [{"name": name, "seconds": seconds, "count": 1}],
        },
    )


def test_update_master_preserves_unseen_and_appends_seen(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    master = tmp_path / "master.json"
    out = reports / "ci_timings_master.json"
    write_json(
        master,
        {
            "schema_version": timing.SCHEMA_VERSION,
            "groups": {
                "seen": {"samples": [1.0], "count": 1},
                "unseen": {"samples": [7.0], "count": 2},
            },
        },
    )
    write_json(
        reports / "stage.timings.json",
        {
            "stash": "stage",
            "metadata": {"job": "job", "build": "12", "stage": "Stage"},
            "groups": [{"name": "seen", "seconds": 3.5, "count": 4}],
        },
    )

    timing.update_master(str(reports), str(master), str(out), update_persistent=True)

    persisted = json.loads(master.read_text())
    merged = json.loads(out.read_text())
    # observed groups are always appended, and unseen groups are left untouched.
    assert persisted["groups"]["seen"]["samples"] == [1.0, 3.5]
    assert persisted["groups"]["unseen"]["samples"] == [7.0]
    assert merged["groups"]["seen"]["samples"] == [1.0, 3.5]
    assert merged["last_update"]["observed_groups"] == 1
    assert merged["last_update"]["persistent_update"] is True


def test_merge_maps_writes_searchable_text(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    write_json(
        reports / "stage.shardmap.json",
        [
            {
                "nodeid": "tests/foo.py::test_bar",
                "stage": "Stage",
                "stash": "stage",
                "shard_id": 0,
                "num_shards": 2,
                "group": "grp",
                "weight_s": 1.25,
                "source": "known",
            }
        ],
    )

    timing.merge_maps(str(reports))

    text = (reports / "shard_map.txt").read_text()
    assert "nodeid=tests/foo.py::test_bar" in text
    assert "stage=Stage" in text
    assert "shard=1/2" in text
    assert "source=known" in text


def test_prepare_timing_snapshot_empty_when_master_missing(tmp_path):
    snapshot = tmp_path / "snapshot.json"
    timing.prepare_timing_snapshot(str(tmp_path / "missing-master.json"), str(snapshot))
    data = json.loads(snapshot.read_text())
    assert data["groups"] == {}
    assert data["schema_version"] == timing.SCHEMA_VERSION


def test_prepare_timing_snapshot_copies_master_when_present(tmp_path):
    master = tmp_path / "master.json"
    snapshot = tmp_path / "snapshot.json"
    write_json(
        master,
        {
            "schema_version": timing.SCHEMA_VERSION,
            "groups": {"slow": {"samples": [12.0]}},
        },
    )

    timing.prepare_timing_snapshot(str(master), str(snapshot))

    data = json.loads(snapshot.read_text())
    assert data["groups"]["slow"]["samples"] == [12.0]


def test_update_master_raises_on_persistent_write_failure(tmp_path, monkeypatch):
    # Persistent write failure propagates so the calling pipeline can mark
    # the build UNSTABLE instead of silently leaving a stale master behind.
    reports = tmp_path / "reports"
    reports.mkdir()
    master = tmp_path / "master.json"
    out = reports / "ci_timings_master.json"
    write_json(master, {"schema_version": timing.SCHEMA_VERSION, "groups": {}})
    write_json(
        reports / "stage.timings.json",
        {
            "stash": "stage",
            "metadata": {"job": "j", "build": "1", "stage": "Stage"},
            "groups": [{"name": "seen", "seconds": 1.0, "count": 1}],
        },
    )

    def boom(*_a, **_k):
        raise IOError("simulated NFS write failure")

    monkeypatch.setattr(jsonio, "write_json_atomic", boom)

    with pytest.raises(IOError, match="simulated NFS write failure"):
        timing.update_master(
            str(reports),
            str(master),
            str(out),
            update_persistent=True,
            metadata={"job": "j", "build": "1"},
        )


def test_update_master_no_master_path_writes_preview(tmp_path):
    # Local-fallback mode (no NFS): the per-build preview is still written,
    # the master simply has nowhere to live.
    reports = tmp_path / "reports"
    reports.mkdir()
    out = reports / "ci_timings_master.json"
    write_json(
        reports / "stage.timings.json",
        {
            "stash": "stage",
            "metadata": {"job": "j", "build": "1", "stage": "Stage"},
            "groups": [{"name": "seen", "seconds": 1.0, "count": 1}],
        },
    )

    rc = timing.update_master(
        str(reports),
        master_path="",
        out_path=str(out),
        metadata={"job": "j", "build": "1"},
    )

    preview = json.loads(out.read_text())
    assert rc == 0
    assert preview["groups"]["seen"]["samples"] == [1.0]
    assert preview["last_update"]["observed_groups"] == 1
    assert preview["last_update"]["persistent_update"] is False


def test_observed_groups_uses_max_seconds_for_duplicate_group(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    _write_observation(reports, "fast", "same", 1.0)
    _write_observation(reports, "slow", "same", 9.0)
    observed = timing.observed_groups_from_reports(str(reports))
    assert observed["same"] == 9.0


def test_update_master_cold_start_accepts_first_observation(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    master = tmp_path / "master.json"
    out = reports / "ci_timings_master.json"
    write_json(master, {"schema_version": timing.SCHEMA_VERSION, "groups": {}})
    _write_observation(reports, "stage", "newgroup", 42.0)
    timing.update_master(str(reports), str(master), str(out), update_persistent=True)
    persisted = json.loads(master.read_text())
    assert persisted["groups"]["newgroup"]["samples"] == [42.0]
    assert persisted["last_update"]["observed_groups"] == 1


def test_update_master_grows_samples_to_max_then_trims(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    master = tmp_path / "master.json"
    out = reports / "ci_timings_master.json"
    _seed_master_with_group(master, "g", [10.0, 10.0, 10.0, 10.0])
    _write_observation(reports, "stage", "g", 11.0)
    timing.update_master(str(reports), str(master), str(out), update_persistent=True)
    persisted = json.loads(master.read_text())
    # 5th sample appended, window full but not yet trimmed.
    assert persisted["groups"]["g"]["samples"] == [10.0, 10.0, 10.0, 10.0, 11.0]
    # Next observation evicts the oldest sample (FIFO ring).
    _write_observation(reports, "stage", "g", 12.0)
    timing.update_master(str(reports), str(master), str(out), update_persistent=True)
    persisted = json.loads(master.read_text())
    assert persisted["groups"]["g"]["samples"] == [10.0, 10.0, 10.0, 11.0, 12.0]


def test_update_master_uses_max_so_slowest_recent_run_sets_weight(tmp_path):
    # max window: [10, 10, 10, 10, 35] weighs the group at its slowest recent
    # run (35), the conservative estimate. The sample ages out of the window
    # after MAX_SAMPLES newer runs, so a one-off spike does not pin it forever.
    reports = tmp_path / "reports"
    reports.mkdir()
    master = tmp_path / "master.json"
    out = reports / "ci_timings_master.json"
    _seed_master_with_group(master, "g", [10.0, 10.0, 10.0, 10.0])
    _write_observation(reports, "stage", "g", 35.0)
    timing.update_master(str(reports), str(master), str(out), update_persistent=True)
    persisted = json.loads(master.read_text())
    assert persisted["groups"]["g"]["samples"] == [10.0, 10.0, 10.0, 10.0, 35.0]
    weights = sharding.load_group_weights(str(master))
    assert weights["g"] == 35.0


def test_update_master_preview_leaves_persistent_master_untouched(tmp_path):
    # Non-persist mode must write the per-build preview to out_path but
    # never touch the on-disk master.
    reports = tmp_path / "reports"
    reports.mkdir()
    master = tmp_path / "master.json"
    out = reports / "ci_timings_master.json"
    write_json(
        master,
        {
            "schema_version": timing.SCHEMA_VERSION,
            "groups": {"g": {"samples": [10.0]}},
        },
    )
    _write_observation(reports, "stage", "g", 25.0)
    timing.update_master(str(reports), str(master), str(out))
    persisted = json.loads(master.read_text())
    preview = json.loads(out.read_text())
    assert persisted["groups"]["g"]["samples"] == [10.0]
    assert preview["groups"]["g"]["samples"] == [10.0, 25.0]
    assert preview["last_update"]["persistent_update"] is False


def test_normalise_master_drops_unknown_schema_version(capsys):
    out = timing.normalise_master({"schema_version": 99, "groups": {"g": {"samples": [1.0]}}})
    assert out["schema_version"] == timing.SCHEMA_VERSION
    assert out["groups"] == {}
    assert "unrecognised schema_version" in capsys.readouterr().err
