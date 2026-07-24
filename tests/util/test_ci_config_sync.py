# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import os
import re
from finn_ci import config

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
