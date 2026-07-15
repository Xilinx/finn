# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CI board and stage tables, plus helpers.

The single source of truth for the CI board metadata (BOARDS), the per-row
build matrix (STAGES), and the derivations the build pipeline needs: row
validation, the Jenkins stage-choice list, and the per-shard plan. Consumed by
the pytest plugin (stash names, STAGES) and by the end2end board
parametrisation (BOARDS/TEST_BOARDS).
"""

import re

# STAGES row markers are interpolated into the CI pipeline's shell -m
# argument, so only the "a or b or c" disjunction shape is accepted. This
# keeps and/not and other shell-unsafe forms out of STAGES.
MARKER_SAFE_PATTERN = re.compile(r"^[A-Za-z0-9_]+( or [A-Za-z0-9_]+)*$")

# Board setup scripts implemented by the CI HW pipeline's on-board test step.
VALID_BOARD_SETUP_SCRIPTS = ("alveo", "pynq")


# =============================================================================
# Tables (BOARDS, STAGES)
# =============================================================================
# Single source of truth for the HW pipeline's per-board metadata and the
# build pipeline's per-row matrix. The Jenkinsfile reads these via the
# hw-config-json and validate-config commands.

# Per-board HW-pipeline metadata. Fields:
#   agentLabel:    Jenkins label of the board agent
#   credentialsId: Jenkins credential binding for sudo on the board
#   restartPrep:   pre-run reboot procedure for Zynq boards
#   setupScript:   alveo (XRT) or pynq (on-board pynq venv)
#   marker:        the -m expression the on-board test file runs against
#   bnnMarker:     the build-side -m token the BNN end2end test file
#                  parametrises this board under (bnn_<board>)
#
# TEST_BOARDS derives from BOARDS and is what tests/end2end
# parametrises against. Reordering the keys reorders the test matrix.
BOARDS = {
    "Pynq-Z1": {
        "agentLabel": "finn-pynq",
        "credentialsId": "pynq-z1-credentials",
        "restartPrep": True,
        "setupScript": "pynq",
        "marker": "Pynq",
        "bnnMarker": "bnn_pynq",
    },
    "KV260_SOM": {
        "agentLabel": "finn-kv260",
        "credentialsId": "user-ubuntu-credentials",
        "restartPrep": True,
        "setupScript": "pynq",
        "marker": "KV260_SOM",
        "bnnMarker": "bnn_kv260",
    },
    "ZCU104": {
        "agentLabel": "finn-zcu104",
        "credentialsId": "pynq-z1-credentials",
        "restartPrep": True,
        "setupScript": "pynq",
        "marker": "ZCU104",
        "bnnMarker": "bnn_zcu104",
    },
    "U250": {
        "agentLabel": "finn-u250",
        "credentialsId": None,
        "restartPrep": False,
        "setupScript": "alveo",
        "marker": "U250",
        "bnnMarker": "bnn_u250",
    },
}

# Board iteration order for test parametrisation. Single source of
# truth: tests/end2end/test_end2end_bnn_pynq.py consumes this tuple.
TEST_BOARDS = tuple(BOARDS)


# Per-test-type human-readable label, surfaced in the Jenkins HW pipeline
# stage names. Add a row when introducing a new STAGES.zipArtifacts.hwTestType.
# validate_config() rejects an hwTestType without a label.
HW_TEST_TYPE_LABELS = {
    "bnn_build_sanity": "Sanity",
    "bnn_build_full": "end2end",
}


# Per-row CI matrix. Fields:
#   param:        Jenkins STAGES choice that activates this row
#   stage:        human-readable display name
#   marker:       pytest -m expression (must match MARKER_SAFE_PATTERN in
#                 the CI pipeline, so only "a or b ...")
#   shards:       how many parallel shards to split the row into
#   workers:      pytest-xdist worker count per shard. >1 implies
#                 --dist loadgroup so xdist_group chains (e.g. tests linked
#                 by load_test_checkpoint_or_skip) stay on one worker
#   coverage:     optional, request coverage report
#   zipArtifacts: optional, {"hwTestType": ..., "boards": [...]} declaring the
#                 build-to-HW handoff zips this row publishes. hwTestType must
#                 have a HW_TEST_TYPE_LABELS entry
STAGES = [
    {
        "param": "sanity",
        "stage": "Sanity - Build Hardware",
        "marker": "sanity_bnn",
        "shards": 4,
        "workers": 1,
        "zipArtifacts": {
            "hwTestType": "bnn_build_sanity",
            "boards": list(TEST_BOARDS),
        },
    },
    {
        "param": "sanity",
        "stage": "Sanity - Unit Tests",
        "marker": "util or brevitas_export or streamline or transform or notebooks",
        "shards": 1,
        "workers": 8,
        "coverage": True,
    },
    {
        "param": "fpgadataflow",
        "stage": "fpgadataflow",
        "marker": "fpgadataflow",
        "shards": 2,
        "workers": 8,
        "coverage": True,
    },
    {
        "param": "end2end",
        "stage": "End2end",
        "marker": "end2end",
        "shards": 3,
        "workers": 6,
    },
    {
        "param": "end2end",
        "stage": "BNN U250",
        "marker": "bnn_u250",
        "shards": 2,
        "workers": 2,
        "zipArtifacts": {"hwTestType": "bnn_build_full", "boards": ["U250"]},
    },
    {
        "param": "end2end",
        "stage": "BNN Pynq-Z1",
        "marker": "bnn_pynq",
        "shards": 3,
        "workers": 2,
        "zipArtifacts": {"hwTestType": "bnn_build_full", "boards": ["Pynq-Z1"]},
    },
    {
        "param": "end2end",
        "stage": "BNN ZCU104",
        "marker": "bnn_zcu104",
        "shards": 2,
        "workers": 4,
        "zipArtifacts": {"hwTestType": "bnn_build_full", "boards": ["ZCU104"]},
    },
    {
        "param": "end2end",
        "stage": "BNN KV260",
        "marker": "bnn_kv260",
        "shards": 2,
        "workers": 2,
        "zipArtifacts": {"hwTestType": "bnn_build_full", "boards": ["KV260_SOM"]},
    },
]


def validate_stage_row(row):
    """Sanity-check a single STAGES row in isolation.

    Catches a typo in param / marker / shards / workers / coverage at
    config-load time so a malformed row cannot silently survive into Jenkins
    or the pytest plugin. A missing or empty param matters most: such a row
    is skipped by every helper that gates on rowIsActive, with nothing
    logging why.
    """
    stage = row.get("stage", "<unnamed>")
    param = row.get("param")
    if not isinstance(param, str) or not param:
        raise ValueError("STAGES row %r has invalid or missing param=%r" % (stage, param))
    marker = row.get("marker", "")
    if not MARKER_SAFE_PATTERN.match(str(marker)):
        raise ValueError(
            "STAGES row %r has unsafe marker %r "
            "(only 'a or b or c ...' is allowed)" % (stage, marker)
        )
    shards = row.get("shards")
    if not isinstance(shards, int) or shards < 1:
        raise ValueError("STAGES row %r has invalid shards=%r" % (stage, shards))
    workers = row.get("workers")
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("STAGES row %r has invalid workers=%r" % (stage, workers))
    if "coverage" in row and not isinstance(row["coverage"], bool):
        raise ValueError(
            "STAGES row %r has invalid coverage=%r (must be bool)" % (stage, row["coverage"])
        )
    zip_art = row.get("zipArtifacts")
    if zip_art is None:
        return
    _validate_zip_artifact(stage, zip_art)


def _validate_zip_artifact(stage, zip_art):
    """Sanity-check the optional "zipArtifacts" block of one STAGES row."""
    if not isinstance(zip_art, dict):
        raise ValueError("STAGES row %r has invalid zipArtifacts=%r" % (stage, zip_art))
    hw_test_type = zip_art.get("hwTestType")
    if not isinstance(hw_test_type, str) or not hw_test_type:
        raise ValueError("STAGES row %r has zipArtifacts without a non-empty hwTestType" % stage)
    boards = zip_art.get("boards")
    if (
        not isinstance(boards, list)
        or not boards
        or not all(isinstance(b, str) and b for b in boards)
    ):
        raise ValueError("STAGES row %r has zipArtifacts without a non-empty boards list" % stage)


def validate_board_row(board, row):
    """Sanity-check one BOARDS row consumed by the CI HW pipeline."""
    if not isinstance(row, dict):
        raise ValueError("BOARDS row %r must be a dict, got %r" % (board, row))
    required = (
        "agentLabel",
        "credentialsId",
        "restartPrep",
        "setupScript",
        "marker",
        "bnnMarker",
    )
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError("BOARDS row %r is missing required field(s): %r" % (board, missing))
    if not isinstance(row["agentLabel"], str) or not row["agentLabel"]:
        raise ValueError("BOARDS row %r has invalid agentLabel=%r" % (board, row["agentLabel"]))
    credentials = row["credentialsId"]
    if credentials is not None and (not isinstance(credentials, str) or not credentials):
        raise ValueError("BOARDS row %r has invalid credentialsId=%r" % (board, credentials))
    if not isinstance(row["restartPrep"], bool):
        raise ValueError("BOARDS row %r has invalid restartPrep=%r" % (board, row["restartPrep"]))
    if row["restartPrep"] and not credentials:
        raise ValueError("BOARDS row %r has restartPrep=true but no credentialsId" % board)
    if row["setupScript"] not in VALID_BOARD_SETUP_SCRIPTS:
        raise ValueError(
            "BOARDS row %r has invalid setupScript=%r (allowed: %r)"
            % (board, row["setupScript"], list(VALID_BOARD_SETUP_SCRIPTS))
        )
    if not isinstance(row["marker"], str) or not row["marker"]:
        raise ValueError("BOARDS row %r has invalid marker=%r" % (board, row["marker"]))
    if not isinstance(row["bnnMarker"], str) or not row["bnnMarker"]:
        raise ValueError("BOARDS row %r has invalid bnnMarker=%r" % (board, row["bnnMarker"]))


def validate_config(stages=None, boards=None, hw_test_type_labels=None):
    """Validate every STAGES row and the global BOARDS cross-references.

    Raises ValueError if a STAGES row references a board not in BOARDS, if a
    row's hwTestType has no HW_TEST_TYPE_LABELS entry, or if any single
    row is malformed. Called by every CLI subcommand that consumes BOARDS,
    STAGES, or HW_TEST_TYPE_LABELS (validate-config, hw-config-json)
    so a bad table fails fast wherever it is first read.
    """
    stages = stages if stages is not None else STAGES
    boards = boards if boards is not None else BOARDS
    labels = hw_test_type_labels if hw_test_type_labels is not None else HW_TEST_TYPE_LABELS
    for board, row in boards.items():
        validate_board_row(board, row)
    for row in stages:
        validate_stage_row(row)
    referenced_boards = set()
    referenced_test_types = set()
    for row in stages:
        zip_art = row.get("zipArtifacts") or {}
        for b in zip_art.get("boards", []) or []:
            referenced_boards.add(str(b))
        hw_test_type = zip_art.get("hwTestType")
        if hw_test_type:
            referenced_test_types.add(str(hw_test_type))
    missing_boards = sorted(referenced_boards - set(boards))
    if missing_boards:
        raise ValueError("STAGES references board(s) not declared in BOARDS: %r" % missing_boards)
    missing_labels = sorted(referenced_test_types - set(labels))
    if missing_labels:
        raise ValueError(
            "STAGES references hwTestType(s) without a HW_TEST_TYPE_LABELS entry: %r"
            % missing_labels
        )


# =============================================================================
# Stage choice / job-key helpers
# =============================================================================


def stash_name(stage, shards, shard_id):
    """Mirror the shard stash naming the CI pipeline relies on."""
    base = re.sub(r"[^a-z0-9]+", "_", stage.lower()).strip("_")
    return base if shards <= 1 else "%s_%d" % (base, shard_id + 1)


def job_key(job_name):
    """Sanitise a Jenkins JOB_NAME into a filesystem-safe ci_runs/<key> path component."""
    # Strip leading/trailing dots so JOB_NAME=".." cannot survive into a
    # parent-directory traversal under ci_runs/<jobKey>/.
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", job_name or "job").strip(".")
    return out or "job"


def ci_param_names(stages=None):
    """Return the distinct STAGES.param values in declaration order."""
    stages = stages if stages is not None else STAGES
    seen = []
    for row in stages:
        name = row.get("param")
        if name and name not in seen:
            seen.append(name)
    return seen


def enabled_params_for_choice(choice, stages=None):
    """Map the Jenkins STAGES choice to the set of CI params it enables.

    "full" enables every distinct param. A bare param name enables just
    that one. Unknown choices fail loudly.
    """
    stages = stages if stages is not None else STAGES
    all_params = ci_param_names(stages)
    if choice == "full":
        return list(all_params)
    if choice in all_params:
        return [choice]
    raise ValueError(
        "unknown STAGES choice %r (recognised: full, %s)" % (choice, ", ".join(all_params))
    )


def jenkins_stage_choices(stages=None):
    """Return the Jenkins UI choices for STAGES in display order.

    "sanity" leads when it exists, then "full"
    for the nightly matrix, then every other distinct param.
    """
    names = list(ci_param_names(stages))
    head = ["sanity"] if "sanity" in names else []
    rest = [n for n in names if n != "sanity"]
    return head + ["full"] + rest


# =============================================================================
# Build-pipeline shard plan
# =============================================================================
# The CI pipeline consumes one shard_plan payload instead of re-deriving
# the active-row/shard/stash matrix in the Jenkinsfile.


def shard_stage_name(stage, shards, shard_id):
    """Mirror the per-shard Jenkins stage display name."""
    return stage if shards <= 1 else "%s (%d/%d)" % (stage, shard_id + 1, shards)


def _shard_entry(row):
    """Expand one active STAGES row into its per-shard plan entries."""
    num_shards = int(row["shards"])
    entries = []
    for shard_id in range(num_shards):
        stash = stash_name(row["stage"], num_shards, shard_id)
        entry = {
            "stage": shard_stage_name(row["stage"], num_shards, shard_id),
            "stash": stash,
            "marker": row["marker"],
            "workers": int(row["workers"]),
            "numShards": num_shards,
            "shardId": shard_id,
            "coverage": bool(row.get("coverage", False)),
        }
        if entry["coverage"]:
            entry["coverageFile"] = "%s.coverage" % stash
        zip_art = row.get("zipArtifacts")
        if zip_art:
            entry["zipArtifacts"] = zip_art
        entries.append(entry)
    return entries


def active_stage_rows(choice, stages=None):
    """Return the STAGES rows enabled by the given Jenkins STAGES choice."""
    stages = stages if stages is not None else STAGES
    enabled = set(enabled_params_for_choice(choice, stages))
    return [row for row in stages if row.get("param") in enabled]


def shard_plan(choice, stage_filter="", stages=None):
    """Build the per-shard plan the build pipeline runs for a STAGES choice.

    Returns a dict with three keys:

      - "shards": the shard branches to run, one entry per shard of each
        active row. When stage_filter is set, only shards whose display name
        contains it survive (a substring match).
      - "candidates": the display name of every active row, unfiltered. Lets
        the caller report a stage_filter that matched nothing alongside the
        stage names it could have matched.
      - "zipArtifacts": one entry per (active row, board) build-to-HW handoff,
        read straight from the active rows. Not split per shard and ignores
        stage_filter, so it mirrors the aggregate-time walk.
    """
    rows = active_stage_rows(choice, stages)
    all_shards = []
    for row in rows:
        all_shards.extend(_shard_entry(row))
    if stage_filter:
        shards = [s for s in all_shards if stage_filter in s["stage"]]
    else:
        shards = all_shards
    zip_artifacts = []
    for row in rows:
        zip_art = row.get("zipArtifacts")
        if not zip_art:
            continue
        for board in zip_art.get("boards", []):
            zip_artifacts.append(
                {
                    "stage": row["stage"],
                    "hwTestType": zip_art["hwTestType"],
                    "board": board,
                }
            )
    return {
        "shards": shards,
        "candidates": [row["stage"] for row in rows],
        "zipArtifacts": zip_artifacts,
    }
