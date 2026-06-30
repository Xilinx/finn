# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(REPO_ROOT, "ci", "scripts", "print_pytest_failures.py")


JUNIT_WITH_FAILURES = """\
<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="suite" tests="3" failures="1" errors="1" skipped="0">
    <testcase classname="pkg.mod" name="test_passes" time="0.01"/>
    <testcase classname="pkg.mod" name="test_fails" time="0.02">
      <failure message="assert 1 == 2">stack line 1
stack line 2
stack line 3</failure>
    </testcase>
    <testcase classname="pkg.mod" name="test_errors" time="0.03">
      <error message="boom">trace line 1
trace line 2</error>
    </testcase>
  </testsuite>
</testsuites>
"""


def _run(xml_path, stash, lines_per, max_fails):
    return subprocess.run(
        [sys.executable, SCRIPT, str(xml_path), stash, str(lines_per), str(max_fails)],
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.mark.util
def test_print_pytest_failures_emits_per_failure_blocks(tmp_path):
    xml = tmp_path / "stage.xml"
    xml.write_text(JUNIT_WITH_FAILURES)

    result = _run(xml, "stage", lines_per=10, max_fails=10)

    out = result.stdout
    assert "[pytest-failures stage] 2 test failure(s)" in out
    assert "FAILURE: pkg.mod::test_fails" in out
    assert "assert 1 == 2" in out
    assert "stack line 3" in out
    assert "ERROR: pkg.mod::test_errors" in out
    assert "trace line 2" in out


@pytest.mark.util
def test_print_pytest_failures_truncates_long_bodies(tmp_path):
    body_lines = "\n".join("line %02d" % i for i in range(50))
    xml = tmp_path / "stage.xml"
    xml.write_text(
        "<?xml version='1.0'?>\n"
        "<testsuites><testsuite name='s' tests='1' failures='1'>\n"
        "<testcase classname='c' name='t'>\n"
        "<failure message='m'>%s</failure>\n"
        "</testcase></testsuite></testsuites>\n" % body_lines
    )

    result = _run(xml, "stage", lines_per=5, max_fails=10)

    assert "earlier lines elided" in result.stdout
    assert "line 49" in result.stdout
    assert "line 04" not in result.stdout


@pytest.mark.util
def test_print_pytest_failures_caps_to_max_failures(tmp_path):
    cases = "\n".join(
        "<testcase classname='c' name='t%d'><failure message='m'>x</failure></testcase>" % i
        for i in range(5)
    )
    xml = tmp_path / "stage.xml"
    xml.write_text(
        "<?xml version='1.0'?>\n"
        "<testsuites><testsuite name='s' tests='5' failures='5'>\n"
        "%s\n</testsuite></testsuites>\n" % cases
    )

    result = _run(xml, "stage", lines_per=10, max_fails=2)

    assert "5 test failure(s)" in result.stdout
    assert "and 3 more failure(s) elided" in result.stdout


@pytest.mark.util
def test_print_pytest_failures_handles_no_failures(tmp_path):
    xml = tmp_path / "stage.xml"
    xml.write_text(
        "<?xml version='1.0'?>\n"
        "<testsuites><testsuite name='s' tests='1' failures='0'>\n"
        "<testcase classname='c' name='t'/></testsuite></testsuites>\n"
    )

    result = _run(xml, "stage", lines_per=10, max_fails=10)

    assert "no test failures recorded" in result.stdout


@pytest.mark.util
def test_print_pytest_failures_handles_unparseable_xml(tmp_path):
    xml = tmp_path / "stage.xml"
    xml.write_text("not actually xml")

    result = _run(xml, "stage", lines_per=10, max_fails=10)

    assert "failed to parse" in result.stdout
