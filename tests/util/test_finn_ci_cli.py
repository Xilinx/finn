# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import json
from finn_ci import __main__ as cli
from finn_ci import config

pytestmark = pytest.mark.util


def test_stage_choices_json_cli(capsys):
    rc = cli.main(["stage-choices-json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out) == config.jenkins_stage_choices()


def test_validate_config_single_invocation_returns_full_payload(capsys):
    # The Jenkinsfile collapses the Validate-time config into this one call.
    # The contract is keys present, well-formed and readJSON-ready, asserted
    # as a subset so adding a future key does not break this test.
    rc = cli.main(["validate-config", "--choice", "sanity", "--job-name", "finn.dev"])
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert {"enabled_params", "job_key", "shard_plan"} <= set(payload)
    assert payload["enabled_params"] == ["sanity"]
    # job-key sanitiser is shared with the standalone subcommand.
    assert payload["job_key"] == "finn.dev"
    # the shard plan is the build pipeline's single source for the branch list
    plan = payload["shard_plan"]
    assert {"shards", "candidates", "zipArtifacts"} <= set(plan)
    assert plan["shards"], "sanity choice must produce at least one shard"


def test_validate_config_rejects_orphan_zipartifact_board(monkeypatch, capsys):
    # validate_config() runs inside the subcommand so a STAGES row with
    # an orphan board fails Validate loudly, not three stages later when
    # the HW pipeline tries to look it up.
    bad_stages = list(config.STAGES) + [
        {
            "param": "sanity",
            "stage": "Bad",
            "marker": "sanity_bnn",
            "shards": 1,
            "workers": 1,
            "zipArtifacts": {"hwTestType": "bnn_build_sanity", "boards": ["NotABoard"]},
        }
    ]
    monkeypatch.setattr(config, "STAGES", bad_stages)
    rc = cli.main(["validate-config", "--choice", "full", "--job-name", "j"])
    assert rc == 2
    assert "NotABoard" in capsys.readouterr().err


def test_validate_config_runs_validate_stage_row_for_every_entry(monkeypatch, capsys):
    # CLI form: main() catches ValueError, prints a one-line ci_sharding:
    # message to stderr, and exits 2 instead of leaking a Python traceback
    # into the Jenkins Validate console.
    bad_stages = [
        {"param": "p", "stage": "Bad", "marker": "a and b", "shards": 1, "workers": 1},
    ]
    monkeypatch.setattr(config, "STAGES", bad_stages)
    rc = cli.main(["validate-config", "--choice", "p", "--job-name", "j"])
    assert rc == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("ci_sharding: ")
    assert "unsafe marker" in captured.err
    assert "Traceback" not in captured.err
