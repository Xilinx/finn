# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build-to-HW handoff resolver and the HW pipeline's config projection.

The build pipeline publishes one zip per (hwTestType, board) with a .READY
sibling, and resolve_build_zips decides which build each board is tested
from. The hw_* helpers flatten the shared BOARDS/STAGES tables into the shape
the Groovy HW pipeline reads via hw-config-json, so the HW side holds no copy
of the config. The report helpers run over the per-board JUnit XML before
Jenkins publishes it, so neither a harness error nor a model the build failed
to package is reported as a hardware test result.
"""

import os
import xml.etree.ElementTree as ET
from finn_ci import config


def resolve_build_zips(artifact_dir, job_key, test_types, boards, build_dir=""):
    """Resolve (testType, board) -> the build zip that pair is tested from.

    Walks ${artifact_dir}/ci_runs/<job_key>/ newest-first and picks, per pair,
    the highest-numbered build whose zips/<testType>/<board>.zip.READY sibling
    is present. A resolved pair carries zip, buildDir, build, latestBuild and
    fallback, where latestBuild is the newest build that published a READY zip
    for that test type and fallback marks a board served by an older build than
    that one. Boards with no READY come back as {}.

    build_dir pins every pair to that single directory, so it is the newest
    build in scope and nothing falls back. A missing READY there is reported
    per-board but does not abort the call.

    Raises ValueError when there is no tree for job_key at all, which is a wrong
    build_job_name rather than a build that produced nothing.
    """
    out = {tt: {b: {} for b in boards} for tt in test_types}
    if build_dir:
        for tt in test_types:
            for b in boards:
                zip_path = os.path.join(build_dir, "zips", tt, "%s.zip" % b)
                if os.path.isfile(zip_path) and os.path.isfile(zip_path + ".READY"):
                    build = os.path.basename(os.path.normpath(build_dir))
                    out[tt][b] = {
                        "zip": zip_path,
                        "buildDir": build_dir,
                        "build": build,
                        "latestBuild": build,
                        "fallback": False,
                    }
        return out

    job_root = _job_root(artifact_dir, job_key)
    try:
        # ASCII digits only: Jenkins build numbers are, and any other numeric
        # character either sorts as a build it is not or makes int() raise
        candidates = sorted(
            (d for d in os.listdir(job_root) if d.isascii() and d.isdigit()),
            key=int,
            reverse=True,
        )
    except OSError:
        return out

    remaining = {(tt, b) for tt in test_types for b in boards}
    for build in candidates:
        if not remaining:
            break
        candidate_dir = os.path.join(job_root, build)
        for tt, b in list(remaining):
            zip_path = os.path.join(candidate_dir, "zips", tt, "%s.zip" % b)
            if os.path.isfile(zip_path) and os.path.isfile(zip_path + ".READY"):
                out[tt][b] = {"zip": zip_path, "buildDir": candidate_dir, "build": build}
                remaining.discard((tt, b))
    for tt in test_types:
        selected = [entry for entry in out.get(tt, {}).values() if entry.get("build")]
        if not selected:
            continue
        # every build plants its directory in its first stage, so the newest
        # directory present says nothing about what was published for this
        # test type and only the newest publishing build is a fair baseline
        latest_build = max((entry["build"] for entry in selected), key=int)
        for entry in selected:
            entry["latestBuild"] = latest_build
            entry["fallback"] = entry["build"] != latest_build
    return out


def _job_root(artifact_dir, job_key):
    """Return ci_runs/<job_key>, or raise naming the job keys that do exist.

    A mistyped build_job_name otherwise leaves every board with no READY zip,
    which reads as a broken build pipeline rather than a wrong parameter.
    """
    runs_root = os.path.join(artifact_dir, "ci_runs")
    job_root = os.path.join(runs_root, job_key)
    if os.path.isdir(job_root):
        return job_root
    try:
        available = sorted(
            d for d in os.listdir(runs_root) if os.path.isdir(os.path.join(runs_root, d))
        )
    except OSError:
        available = []
    raise ValueError(
        "resolve-build-zips: no build artifacts at %s. %s"
        % (
            job_root,
            (
                "Check build_job_name. Job keys published here: %s" % ", ".join(available)
                if available
                else "Nothing has been published under %s yet." % runs_root
            ),
        )
    )


# A testcase pytest writes for a file it could not import carries no classname
# and the child element says which counter on the enclosing suite it fed. Every
# real test has a classname, because its node id always names the module.
_COUNTER_FOR_CHILD = {"error": "errors", "failure": "failures"}


def strip_collection_errors(reports_dir):
    """Drop collection-error entries from every JUnit XML under reports_dir.

    Returns {filename: [removed names]} for the files that changed, empty when
    there was nothing to do. A pytest run that dies during collection still
    writes a report, and Jenkins counts those entries as failed tests belonging
    to no board, which overstates the hardware failure count. Entries for tests
    that actually ran are left alone, so a run that collected some tests and
    then errored keeps its real results.
    """
    if not os.path.isdir(reports_dir):
        return {}
    dropped = {}
    for name in sorted(os.listdir(reports_dir)):
        if not name.endswith(".xml"):
            continue
        removed = _strip_one_report(os.path.join(reports_dir, name))
        if removed:
            dropped[name] = removed
    return dropped


def _strip_one_report(xml_path):
    """Rewrite one JUnit XML without its collection-error entries.

    A file that cannot be read or rewritten is skipped rather than costing every
    other board its cleanup.
    """
    try:
        tree = ET.parse(xml_path)
    except (ET.ParseError, OSError):
        return []
    removed = []
    # iter() covers both the <testsuites> wrapper and a bare <testsuite> root
    for suite in tree.getroot().iter("testsuite"):
        for case in list(suite.findall("testcase")):
            if case.get("classname"):
                continue
            for child, counter in _COUNTER_FOR_CHILD.items():
                if case.find(child) is None:
                    continue
                suite.remove(case)
                removed.append(case.get("name", "?"))
                _decrement(suite, "tests")
                _decrement(suite, counter)
                break
    if removed:
        try:
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)
        except OSError:
            return []
    return removed


def _decrement(element, attr):
    try:
        value = int(element.get(attr, "0"))
    except ValueError:
        return
    element.set(attr, str(max(value - 1, 0)))


# Leads the skip reason the on-board harness writes for a model directory the
# build packaged without a driver. A JUnit skip carries no field but its
# message, so this prefix is what separates one from an unrelated skip. The
# harness ships to the board alone and cannot import this package, so the
# literal is repeated there and held in step by test_ci_config_sync.
PACKAGING_SKIP_PREFIX = "incomplete deployment package:"


def packaging_skips(reports_dir):
    """Map each JUnit XML under reports_dir to the packaging skips it carries.

    Returns {filename: [distinct reasons]} for the files that carry any, empty
    when there are none. A skip does not fail a Jenkins build, so a board whose
    models were all packaged without a driver would publish a green run that
    tested no hardware at all. Reasons come back whole, each already naming its
    model directory and the files it lacks.
    """
    if not os.path.isdir(reports_dir):
        return {}
    found = {}
    for name in sorted(os.listdir(reports_dir)):
        if not name.endswith(".xml"):
            continue
        reasons = _packaging_skips_in(os.path.join(reports_dir, name))
        if reasons:
            found[name] = reasons
    return found


def _packaging_skips_in(xml_path):
    """Distinct packaging-skip reasons in one report, in document order.

    Both of a model directory's tests skip with the same reason, so they are
    deduplicated to one line per directory.
    """
    try:
        tree = ET.parse(xml_path)
    except (ET.ParseError, OSError):
        return []
    reasons = []
    for skipped in tree.getroot().iter("skipped"):
        message = skipped.get("message", "")
        if not message.startswith(PACKAGING_SKIP_PREFIX):
            continue
        reason = message[len(PACKAGING_SKIP_PREFIX) :].strip()
        if reason and reason not in reasons:
            reasons.append(reason)
    return reasons


def hw_shards(boards=None):
    """Flatten BOARDS into the ordered list of rows the Groovy HW pipeline expects."""
    boards = boards if boards is not None else config.BOARDS
    return [dict(board=name, **fields) for name, fields in boards.items()]


def hw_test_types(stages=None):
    """Return distinct zipArtifacts.hwTestType values in declaration order.

    Single source of truth for the HW pipeline's per-test-type stage loop.
    """
    stages = stages if stages is not None else config.STAGES
    seen = []
    for row in stages:
        zip_art = row.get("zipArtifacts")
        if not zip_art:
            continue
        hw_test_type = zip_art["hwTestType"]
        if hw_test_type not in seen:
            seen.append(hw_test_type)
    return seen


def hw_test_type_labels(stages=None, labels=None):
    """Return {hwTestType: label} for every hwTestType referenced in STAGES.

    Same iteration order as hw_test_types. Adding an HW test type is an entry
    in HW_TEST_TYPE_LABELS, with no Jenkinsfile edit.
    """
    stages = stages if stages is not None else config.STAGES
    labels = labels if labels is not None else config.HW_TEST_TYPE_LABELS
    return {tt: labels[tt] for tt in hw_test_types(stages)}
