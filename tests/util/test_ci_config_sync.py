# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import os
import re
from finn_ci import config, hw

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytestmark = pytest.mark.util


def test_jenkinsfile_stage_choices_match_python_source():
    # Anchor on the STAGES choice block so a future ``choice(name: 'XYZ', ...)``
    # cannot match instead. Accept both single- and double-quoted Groovy strings.
    jenkinsfile = os.path.join(REPO_ROOT, "ci", "Jenkinsfile")
    text = open(jenkinsfile).read()
    match = re.search(
        r"""choice\(\s*name:\s*['"]STAGES['"],\s*choices:\s*\[([^\]]+)\]""",
        text,
    )
    assert match is not None, "could not locate STAGES choice block in Jenkinsfile"
    choices = re.findall(r"""['"]([^'"]+)['"]""", match.group(1))
    expected = config.jenkins_stage_choices()
    assert (
        choices == expected
    ), "Jenkinsfile STAGES choices %r drifted from finn_ci.config.jenkins_stage_choices() %r" % (
        choices,
        expected,
    )


def test_readme_stages_table_matches_python_source():
    readme = os.path.join(REPO_ROOT, "ci", "README.md")
    text = open(readme).read()
    # Parse the values column of the "| STAGES value | ... |" table.
    table_rows = re.findall(r"^\|\s*`([a-z0-9_]+)`(?:\s*\(default\))?\s*\|", text, re.MULTILINE)
    expected = config.jenkins_stage_choices()
    assert (
        table_rows == expected
    ), "README STAGES table %r drifted from finn_ci.config.jenkins_stage_choices() %r" % (
        table_rows,
        expected,
    )


def test_board_harness_packaging_skip_prefix_matches_python_source():
    # The board harness ships on its own and cannot import finn_ci, so the prefix
    # report aggregation keys on is written out in both places. Read from source
    # rather than imported, because the harness pulls in numpy and scipy that
    # only a board needs.
    harness = os.path.join(REPO_ROOT, "ci", "test_bnn_hw_pytest.py")
    text = open(harness).read()
    match = re.search(r"""^packaging_skip_prefix\s*=\s*['"]([^'"]+)['"]""", text, re.MULTILINE)
    assert match is not None, "could not locate packaging_skip_prefix in ci/test_bnn_hw_pytest.py"
    assert (
        match.group(1) == hw.PACKAGING_SKIP_PREFIX
    ), "harness packaging_skip_prefix %r drifted from finn_ci.hw.PACKAGING_SKIP_PREFIX %r" % (
        match.group(1),
        hw.PACKAGING_SKIP_PREFIX,
    )
