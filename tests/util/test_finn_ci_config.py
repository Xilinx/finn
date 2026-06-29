# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from finn_ci import config

pytestmark = pytest.mark.util


def test_enabled_params_sanity_runs_only_sanity_rows():
    assert config.enabled_params_for_choice("sanity") == ["sanity"]


def test_enabled_params_bare_name_returns_that_one():
    assert config.enabled_params_for_choice("fpgadataflow") == ["fpgadataflow"]
    assert config.enabled_params_for_choice("end2end") == ["end2end"]


# ----------------------------------------------------------------------------
# shard_plan
# ----------------------------------------------------------------------------

_PLAN_STAGES = [
    {"param": "sanity", "stage": "Sanity Build", "marker": "sanity_bnn", "shards": 1, "workers": 1},
    {
        "param": "end2end",
        "stage": "BNN U250",
        "marker": "bnn_u250",
        "shards": 2,
        "workers": 2,
        "coverage": True,
        "zipArtifacts": {"hwTestType": "bnn_build_full", "boards": ["U250", "ZCU104"]},
    },
]


def test_shard_plan_expands_rows_into_per_shard_entries():
    plan = config.shard_plan("full", stages=_PLAN_STAGES)
    stages = [s["stage"] for s in plan["shards"]]
    assert stages == ["Sanity Build", "BNN U250 (1/2)", "BNN U250 (2/2)"]
    # single-shard row keeps the bare display name and stash
    sanity = plan["shards"][0]
    assert sanity["stash"] == "sanity_build"
    assert "coverageFile" not in sanity
    # multi-shard row carries shardId, coverageFile, zipArtifacts
    u250_b = plan["shards"][2]
    assert u250_b["shardId"] == 1
    assert u250_b["numShards"] == 2
    assert u250_b["stash"] == "bnn_u250_2"
    assert u250_b["coverageFile"] == "bnn_u250_2.coverage"
    assert u250_b["zipArtifacts"]["hwTestType"] == "bnn_build_full"


def test_shard_plan_filters_by_substring_but_keeps_candidates():
    plan = config.shard_plan("full", stage_filter="U250", stages=_PLAN_STAGES)
    assert [s["stage"] for s in plan["shards"]] == ["BNN U250 (1/2)", "BNN U250 (2/2)"]
    # candidates is the unfiltered active-row display list for the error path
    assert plan["candidates"] == ["Sanity Build", "BNN U250"]


def test_shard_plan_filter_miss_yields_empty_shards_with_candidates():
    plan = config.shard_plan("full", stage_filter="does-not-match", stages=_PLAN_STAGES)
    assert plan["shards"] == []
    assert plan["candidates"] == ["Sanity Build", "BNN U250"]


def test_shard_plan_zip_artifacts_flatten_per_row_per_board_not_per_shard():
    plan = config.shard_plan("full", stages=_PLAN_STAGES)
    # one entry per (active row, board), so the 2-shard row contributes once
    # per board, not once per shard
    assert plan["zipArtifacts"] == [
        {"stage": "BNN U250", "hwTestType": "bnn_build_full", "board": "U250"},
        {"stage": "BNN U250", "hwTestType": "bnn_build_full", "board": "ZCU104"},
    ]


def test_shard_plan_only_includes_active_rows():
    plan = config.shard_plan("sanity", stages=_PLAN_STAGES)
    assert [s["stage"] for s in plan["shards"]] == ["Sanity Build"]


def test_shard_plan_matches_live_stages_stash_names():
    # shard_plan stash names must equal stash_name() so the conftest plugin
    # sidecars and the aggregate-time unstash agree.
    plan = config.shard_plan("full")
    for entry in plan["shards"]:
        expected = config.stash_name(
            entry["stage"].split(" (")[0], entry["numShards"], entry["shardId"]
        )
        assert entry["stash"] == expected


def test_marker_safe_pattern_accepts_live_markers_and_rejects_foot_guns():
    # MARKER_SAFE_PATTERN has one home (finn_ci.config) and marker safety is
    # enforced in validate_config during Validate, so the Jenkinsfile carries no
    # duplicate regex. The marker is interpolated into the CI pipeline's shell
    # -m argument, so the pattern must forbid and/not and the double-space shape
    # (empty token) that would be unsafe or ambiguous there.
    pattern = config.MARKER_SAFE_PATTERN
    for row in config.STAGES:
        assert pattern.match(row["marker"]), (
            "MARKER_SAFE_PATTERN rejects STAGES marker %r" % row["marker"]
        )
    for bad in ("foo and bar", "not slow", "foo  or bar", "foo or", "or foo"):
        assert not pattern.match(bad), "MARKER_SAFE_PATTERN should reject %r" % bad


def test_enabled_params_rejects_unknown_choice_loudly():
    with pytest.raises(ValueError, match="unknown STAGES choice"):
        config.enabled_params_for_choice("notarealchoice")


def test_enabled_params_full_on_synthetic_stages_picks_up_new_params():
    # A new CI param (no edits to enabled_params_for_choice required) flows
    # through the "full" choice automatically because the function reads
    # ci_param_names.
    custom_stages = [
        {"param": "sanity", "stage": "S", "marker": "s", "shards": 1, "workers": 1},
        {"param": "newthing", "stage": "N", "marker": "n", "shards": 1, "workers": 1},
    ]
    assert config.enabled_params_for_choice("full", stages=custom_stages) == [
        "sanity",
        "newthing",
    ]
    assert config.enabled_params_for_choice("newthing", stages=custom_stages) == ["newthing"]
    assert config.jenkins_stage_choices(stages=custom_stages) == [
        "sanity",
        "full",
        "newthing",
    ]


def test_jenkins_stage_choices_omits_sanity_head_when_param_absent():
    # If a future STAGES has no sanity rows at all, jenkins_stage_choices()
    # must not synthesise a "sanity" choice the dropdown cannot deliver.
    no_sanity = [
        {"param": "fpgadataflow", "stage": "F", "marker": "f", "shards": 1, "workers": 1},
        {"param": "end2end", "stage": "E", "marker": "e", "shards": 1, "workers": 1},
    ]
    assert config.jenkins_stage_choices(stages=no_sanity) == [
        "full",
        "fpgadataflow",
        "end2end",
    ]


@pytest.mark.parametrize(
    "board,row,match",
    [
        ("B", {}, "missing required"),
        (
            "B",
            {
                "agentLabel": "",
                "credentialsId": None,
                "restartPrep": False,
                "setupScript": "pynq",
                "marker": "B",
                "bnnMarker": "bnn_b",
            },
            "invalid agentLabel",
        ),
        (
            "B",
            {
                "agentLabel": "finn-b",
                "credentialsId": "",
                "restartPrep": False,
                "setupScript": "pynq",
                "marker": "B",
                "bnnMarker": "bnn_b",
            },
            "invalid credentialsId",
        ),
        (
            "B",
            {
                "agentLabel": "finn-b",
                "credentialsId": None,
                "restartPrep": True,
                "setupScript": "pynq",
                "marker": "B",
                "bnnMarker": "bnn_b",
            },
            "restartPrep=true but no credentialsId",
        ),
        (
            "B",
            {
                "agentLabel": "finn-b",
                "credentialsId": None,
                "restartPrep": False,
                "setupScript": "bogus",
                "marker": "B",
                "bnnMarker": "bnn_b",
            },
            "invalid setupScript",
        ),
        (
            "B",
            {
                "agentLabel": "finn-b",
                "credentialsId": None,
                "restartPrep": False,
                "setupScript": "pynq",
                "marker": "B",
                "bnnMarker": "",
            },
            "invalid bnnMarker",
        ),
    ],
)
def test_validate_board_row_rejects_bad_input(board, row, match):
    with pytest.raises(ValueError, match=match):
        config.validate_board_row(board, row)


def test_validate_config_rejects_orphan_zipartifact_board_via_helper():
    custom_stages = [
        {
            "param": "p",
            "stage": "X",
            "marker": "x",
            "shards": 1,
            "workers": 1,
            "zipArtifacts": {"hwTestType": "t", "boards": ["FakeBoard"]},
        },
    ]
    with pytest.raises(ValueError, match="FakeBoard"):
        config.validate_config(stages=custom_stages, boards={})


@pytest.mark.parametrize(
    "zip_art,match",
    [
        ({}, "non-empty hwTestType"),
        ({"hwTestType": "", "boards": ["U250"]}, "non-empty hwTestType"),
        ({"hwTestType": "t"}, "non-empty boards list"),
        ({"hwTestType": "t", "boards": []}, "non-empty boards list"),
        ({"hwTestType": "t", "boards": [""]}, "non-empty boards list"),
    ],
)
def test_validate_stage_row_rejects_bad_zip_artifacts(zip_art, match):
    row = {
        "param": "p",
        "stage": "X",
        "marker": "x",
        "shards": 1,
        "workers": 1,
        "zipArtifacts": zip_art,
    }
    with pytest.raises(ValueError, match=match):
        config.validate_stage_row(row)


def test_job_key_strips_dot_traversal():
    # the per-build prune paths are os.path.join(parent, job_key(...)/...) so
    # JOB_NAME=".." or all-dot inputs must not survive the sanitiser
    assert config.job_key("..") == "job"
    assert config.job_key(".") == "job"
    assert config.job_key("...") == "job"
    assert config.job_key("") == "job"
    assert config.job_key(None) == "job"
    # leading-/trailing-dot inputs lose just the offending dots
    assert config.job_key(".foo") == "foo"
    assert config.job_key("foo.") == "foo"
    # legitimate names with internal dots survive
    assert config.job_key("finn.dev") == "finn.dev"
    # the path-separator class still gets replaced with '_' as before
    assert config.job_key("../etc") == "_etc"


def test_validate_config_rejects_hw_test_type_without_label(monkeypatch):
    bad_stages = list(config.STAGES) + [
        {
            "param": "sanity",
            "stage": "Bad",
            "marker": "sanity_bnn",
            "shards": 1,
            "workers": 1,
            "zipArtifacts": {"hwTestType": "bnn_build_unlabelled", "boards": ["U250"]},
        }
    ]
    monkeypatch.setattr(config, "STAGES", bad_stages)
    with pytest.raises(ValueError, match="bnn_build_unlabelled"):
        config.validate_config()


def test_validate_stage_row_accepts_each_live_row():
    for row in config.STAGES:
        config.validate_stage_row(row)


@pytest.mark.parametrize(
    "bad,match",
    [
        (
            {"stage": "X", "marker": "a", "shards": 1, "workers": 1},
            "missing param",
        ),
        (
            {"stage": "X", "param": "", "marker": "a", "shards": 1, "workers": 1},
            "missing param",
        ),
        (
            {"stage": "X", "param": 7, "marker": "a", "shards": 1, "workers": 1},
            "missing param",
        ),
        (
            {"stage": "X", "param": "p", "marker": "a and b", "shards": 1, "workers": 1},
            "unsafe marker",
        ),
        (
            {"stage": "X", "param": "p", "marker": "a", "shards": 0, "workers": 1},
            "invalid shards",
        ),
        (
            {"stage": "X", "param": "p", "marker": "a", "shards": 1, "workers": 0},
            "invalid workers",
        ),
        (
            {
                "stage": "X",
                "param": "p",
                "marker": "a",
                "shards": 1,
                "workers": 1,
                "coverage": "yes",
            },
            "invalid coverage",
        ),
    ],
)
def test_validate_stage_row_rejects_bad_input(bad, match):
    with pytest.raises(ValueError, match=match):
        config.validate_stage_row(bad)


def test_validate_stage_row_accepts_explicit_coverage_bool():
    row = {
        "param": "p",
        "stage": "X",
        "marker": "a",
        "shards": 1,
        "workers": 1,
        "coverage": True,
    }
    config.validate_stage_row(row)
    row["coverage"] = False
    config.validate_stage_row(row)
