# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from finn_ci import __main__ as cli
from finn_ci import failures

pytestmark = pytest.mark.util


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


def test_print_failures_emits_per_failure_blocks(tmp_path, capsys):
    xml = tmp_path / "stage.xml"
    xml.write_text(JUNIT_WITH_FAILURES)

    rc = failures.print_failures(str(xml), "stage", 10, 10)

    out = capsys.readouterr().out
    assert rc == 0
    assert "[pytest-failures stage] 2 test failure(s)" in out
    assert "FAILURE: pkg.mod::test_fails" in out
    assert "assert 1 == 2" in out
    assert "stack line 3" in out
    assert "ERROR: pkg.mod::test_errors" in out
    assert "trace line 2" in out


def test_print_failures_truncates_long_bodies(tmp_path, capsys):
    body_lines = "\n".join("line %02d" % i for i in range(50))
    xml = tmp_path / "stage.xml"
    xml.write_text(
        "<?xml version='1.0'?>\n"
        "<testsuites><testsuite name='s' tests='1' failures='1'>\n"
        "<testcase classname='c' name='t'>\n"
        "<failure message='m'>%s</failure>\n"
        "</testcase></testsuite></testsuites>\n" % body_lines
    )

    failures.print_failures(str(xml), "stage", 5, 10)

    out = capsys.readouterr().out
    assert "earlier lines elided" in out
    assert "line 49" in out
    assert "line 04" not in out


def test_print_failures_caps_to_max_failures(tmp_path, capsys):
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

    failures.print_failures(str(xml), "stage", 10, 2)

    out = capsys.readouterr().out
    assert "5 test failure(s)" in out
    assert "and 3 more failure(s) elided" in out


def test_print_failures_handles_no_failures(tmp_path, capsys):
    xml = tmp_path / "stage.xml"
    xml.write_text(
        "<?xml version='1.0'?>\n"
        "<testsuites><testsuite name='s' tests='1' failures='0'>\n"
        "<testcase classname='c' name='t'/></testsuite></testsuites>\n"
    )

    failures.print_failures(str(xml), "stage", 10, 10)

    assert "no test failures recorded" in capsys.readouterr().out


def test_print_failures_handles_unparseable_xml(tmp_path, capsys):
    xml = tmp_path / "stage.xml"
    xml.write_text("not actually xml")

    failures.print_failures(str(xml), "stage", 10, 10)

    assert "failed to parse" in capsys.readouterr().out


def test_print_failures_cli_smoke(tmp_path, capsys):
    # exercise the print-failures subcommand wiring in finn_ci.__main__.
    xml = tmp_path / "stage.xml"
    xml.write_text(JUNIT_WITH_FAILURES)
    rc = cli.main(["print-failures", str(xml), "stage", "10", "10"])
    assert rc == 0
    assert "[pytest-failures stage] 2 test failure(s)" in capsys.readouterr().out
