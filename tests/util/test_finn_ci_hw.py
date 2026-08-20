# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import importlib.util
import json
import os
import warnings
import xml.etree.ElementTree as ET
from finn_ci import __main__ as cli
from finn_ci import config, hw
from types import SimpleNamespace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytestmark = pytest.mark.util


def _seed_build(artifact_dir, job_key, build, test_type, board, ready=True):
    build_dir = artifact_dir / "ci_runs" / job_key / build
    zip_dir = build_dir / "zips" / test_type
    zip_dir.mkdir(parents=True, exist_ok=True)
    zip_path = zip_dir / ("%s.zip" % board)
    zip_path.write_text("")
    if ready:
        (zip_dir / ("%s.zip.READY" % board)).write_text("")
    return zip_path, build_dir


def test_resolve_build_zips_picks_newest_ready_per_pair(tmp_path):
    art = tmp_path / "artifacts"
    _seed_build(art, "finn", "10", "bnn_build_full", "U250", ready=False)
    _seed_build(art, "finn", "11", "bnn_build_full", "U250", ready=True)
    _seed_build(art, "finn", "12", "bnn_build_full", "Pynq-Z1", ready=True)

    out = hw.resolve_build_zips(str(art), "finn", ["bnn_build_full"], ["U250", "Pynq-Z1", "ZCU104"])

    assert out["bnn_build_full"]["U250"]["zip"].endswith("/11/zips/bnn_build_full/U250.zip")
    assert out["bnn_build_full"]["U250"]["buildDir"].endswith("/finn/11")
    assert out["bnn_build_full"]["Pynq-Z1"]["zip"].endswith("/12/zips/bnn_build_full/Pynq-Z1.zip")
    assert out["bnn_build_full"]["ZCU104"] == {}


def test_resolve_build_zips_falls_back_to_older_build_per_board(tmp_path):
    # A new build that only succeeded for Pynq-Z1 must not strand U250 on the
    # older build it last produced a READY for.
    art = tmp_path / "artifacts"
    _seed_build(art, "finn", "20", "bnn_build_full", "U250", ready=True)
    _seed_build(art, "finn", "21", "bnn_build_full", "Pynq-Z1", ready=True)

    out = hw.resolve_build_zips(str(art), "finn", ["bnn_build_full"], ["U250", "Pynq-Z1"])

    assert out["bnn_build_full"]["U250"]["buildDir"].endswith("/finn/20")
    assert out["bnn_build_full"]["Pynq-Z1"]["buildDir"].endswith("/finn/21")
    assert out["bnn_build_full"]["U250"]["fallback"] is True
    assert out["bnn_build_full"]["Pynq-Z1"]["fallback"] is False
    assert out["bnn_build_full"]["U250"]["latestBuild"] == "21"


def test_resolve_build_zips_ignores_a_build_that_published_nothing(tmp_path):
    art = tmp_path / "artifacts"
    _seed_build(art, "finn", "20", "bnn_build_full", "U250", ready=True)
    # Every build plants its directory in its first stage, so build 21 exists
    # having published no READY zip at all. It is not a build U250 is behind.
    (art / "ci_runs" / "finn" / "21").mkdir(parents=True)

    out = hw.resolve_build_zips(str(art), "finn", ["bnn_build_full"], ["U250"])

    assert out["bnn_build_full"]["U250"]["buildDir"].endswith("/finn/20")
    assert out["bnn_build_full"]["U250"]["latestBuild"] == "20"
    assert out["bnn_build_full"]["U250"]["fallback"] is False


def test_resolve_build_zips_measures_fallback_per_test_type(tmp_path):
    # The ordinary cadence: a full build, then a sanity-only build, then a
    # build still running. Nothing here is stale.
    art = tmp_path / "artifacts"
    boards = ["U250", "Pynq-Z1", "ZCU104", "KV260_SOM"]
    for board in boards:
        _seed_build(art, "finn", "100", "bnn_build_full", board, ready=True)
        _seed_build(art, "finn", "100", "bnn_build_sanity", board, ready=True)
        _seed_build(art, "finn", "101", "bnn_build_sanity", board, ready=True)
    (art / "ci_runs" / "finn" / "102").mkdir(parents=True)

    out = hw.resolve_build_zips(str(art), "finn", ["bnn_build_full", "bnn_build_sanity"], boards)

    assert [out["bnn_build_full"][b]["latestBuild"] for b in boards] == ["100"] * 4
    assert [out["bnn_build_sanity"][b]["latestBuild"] for b in boards] == ["101"] * 4
    assert not any(out[tt][b]["fallback"] for tt in out for b in boards)


def test_resolve_build_zips_flags_only_the_board_the_newest_build_missed(tmp_path):
    art = tmp_path / "artifacts"
    served = ["U250", "Pynq-Z1", "ZCU104"]
    for board in served + ["KV260_SOM"]:
        _seed_build(art, "finn", "30", "bnn_build_full", board, ready=True)
    for board in served:
        _seed_build(art, "finn", "31", "bnn_build_full", board, ready=True)

    out = hw.resolve_build_zips(str(art), "finn", ["bnn_build_full"], served + ["KV260_SOM"])

    assert not any(out["bnn_build_full"][b]["fallback"] for b in served)
    assert out["bnn_build_full"]["KV260_SOM"]["build"] == "30"
    assert out["bnn_build_full"]["KV260_SOM"]["latestBuild"] == "31"
    assert out["bnn_build_full"]["KV260_SOM"]["fallback"] is True


def test_resolve_build_zips_leaves_a_never_published_test_type_empty(tmp_path):
    art = tmp_path / "artifacts"
    _seed_build(art, "finn", "40", "bnn_build_full", "U250", ready=True)

    out = hw.resolve_build_zips(str(art), "finn", ["bnn_build_full", "bnn_build_sanity"], ["U250"])

    assert out["bnn_build_full"]["U250"]["fallback"] is False
    assert out["bnn_build_sanity"]["U250"] == {}


def test_resolve_build_zips_honours_explicit_build_dir(tmp_path):
    art = tmp_path / "artifacts"
    _seed_build(art, "finn", "5", "bnn_build_full", "U250", ready=True)
    _seed_build(art, "finn", "6", "bnn_build_full", "U250", ready=True)
    explicit = art / "ci_runs" / "finn" / "5"
    out = hw.resolve_build_zips(
        str(art), "finn", ["bnn_build_full"], ["U250"], build_dir=str(explicit)
    )
    assert out["bnn_build_full"]["U250"]["buildDir"] == str(explicit)
    # a pinned directory is the only build in scope, newer ones do not count
    assert out["bnn_build_full"]["U250"]["latestBuild"] == "5"
    assert out["bnn_build_full"]["U250"]["fallback"] is False


def test_resolve_build_zips_returns_empty_when_no_ready_anywhere(tmp_path):
    art = tmp_path / "artifacts"
    _seed_build(art, "finn", "1", "bnn_build_full", "U250", ready=False)

    out = hw.resolve_build_zips(str(art), "finn", ["bnn_build_full"], ["U250"])
    assert out == {"bnn_build_full": {"U250": {}}}


def test_resolve_build_zips_names_the_job_keys_that_exist_when_one_is_mistyped(tmp_path):
    # A mistyped build_job_name otherwise blames the build pipeline for the
    # empty result.
    art = tmp_path / "artifacts"
    _seed_build(art, "finn", "7", "bnn_build_full", "U250", ready=True)

    with pytest.raises(ValueError) as excinfo:
        hw.resolve_build_zips(str(art), "fnin", ["bnn_build_full"], ["U250"])

    assert "build_job_name" in str(excinfo.value)
    assert "finn" in str(excinfo.value)


def test_resolve_build_zips_fails_when_nothing_has_been_published_at_all(tmp_path):
    with pytest.raises(ValueError) as excinfo:
        hw.resolve_build_zips(str(tmp_path / "absent"), "finn", ["bnn_build_full"], ["U250"])

    assert "ci_runs" in str(excinfo.value)


def test_resolve_build_zips_honours_build_dir_without_any_job_tree(tmp_path):
    # The off-Jenkins recovery path pins one directory, so it must not need the
    # per-job tree the auto-discovery path fails without.
    explicit = tmp_path / "somewhere" / "9"
    zip_dir = explicit / "zips" / "bnn_build_full"
    zip_dir.mkdir(parents=True)
    (zip_dir / "U250.zip").write_text("")
    (zip_dir / "U250.zip.READY").write_text("")

    out = hw.resolve_build_zips(
        str(tmp_path / "absent"), "finn", ["bnn_build_full"], ["U250"], build_dir=str(explicit)
    )
    assert out["bnn_build_full"]["U250"]["buildDir"] == str(explicit)


def test_hw_shards_flattens_boards_dict_for_groovy():
    rows = hw.hw_shards()
    boards = [r["board"] for r in rows]
    assert "U250" in boards
    u250 = next(r for r in rows if r["board"] == "U250")
    assert u250["agentLabel"] == "finn-u250"
    assert u250["credentialsId"] is None
    assert u250["restartPrep"] is False
    # ordering matches BOARDS insertion order so HW parallel-branch order
    # is stable across builds
    assert boards == list(config.BOARDS.keys())


def _write_report(reports_dir, name, body, root="testsuites"):
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / name
    inner = '<testsuite name="pytest" errors="%s" failures="%s" tests="%s">%s</testsuite>' % (
        body.count("<error"),
        body.count("<failure"),
        body.count("<testcase"),
        body,
    )
    path.write_text(inner if root == "testsuite" else "<testsuites>%s</testsuites>" % inner)
    return path


# What pytest writes for a file it could not import: no classname, the file's
# dotted path as the name.
_COLLECT_ERROR = (
    '<testcase classname="" name="bnn_w1_a1_tfc.external.xsimd.test.test_wasm">'
    '<error message="collection failure">ModuleNotFoundError</error></testcase>'
)
_REAL_PASS = (
    '<testcase classname="test_bnn_hw_pytest.TestBnn" '
    'name="test_type_execute[Pynq_bnn_w1_a1_tfc_batchSize-1_platform-zynq-iodma]"/>'
)
_REAL_FAIL = (
    '<testcase classname="test_bnn_hw_pytest.TestBnn" '
    'name="test_type_execute[Pynq_bnn_w2_a2_tfc_batchSize-1_platform-zynq-iodma]">'
    "<failure>readback mismatch</failure></testcase>"
)


def test_strip_collection_errors_drops_entries_and_fixes_counters(tmp_path):
    reports = tmp_path / "reports"
    path = _write_report(reports, "bnn_build_full_hw_U250.xml", _COLLECT_ERROR * 2)

    dropped = hw.strip_collection_errors(str(reports))

    assert list(dropped) == ["bnn_build_full_hw_U250.xml"]
    assert len(dropped["bnn_build_full_hw_U250.xml"]) == 2
    suite = ET.parse(str(path)).getroot().find("testsuite")
    assert suite.findall("testcase") == []
    assert suite.get("tests") == "0"
    assert suite.get("errors") == "0"


def test_strip_collection_errors_keeps_the_tests_a_partial_run_did_produce(tmp_path):
    # A run that collected part of its suite and then errored must keep its
    # real results, including the genuine failures.
    reports = tmp_path / "reports"
    path = _write_report(reports, "hw.xml", _REAL_PASS + _COLLECT_ERROR + _REAL_FAIL)

    hw.strip_collection_errors(str(reports))

    suite = ET.parse(str(path)).getroot().find("testsuite")
    names = [tc.get("classname") for tc in suite.findall("testcase")]
    assert names == ["test_bnn_hw_pytest.TestBnn"] * 2
    assert suite.get("tests") == "2"
    assert suite.get("errors") == "0"
    assert suite.get("failures") == "1"
    assert suite.find("testcase/failure") is not None


def test_strip_collection_errors_leaves_a_clean_report_untouched(tmp_path):
    reports = tmp_path / "reports"
    path = _write_report(reports, "hw.xml", _REAL_PASS + _REAL_FAIL)
    before = path.read_text()

    assert hw.strip_collection_errors(str(reports)) == {}
    assert path.read_text() == before


def test_strip_collection_errors_handles_a_bare_testsuite_root(tmp_path):
    reports = tmp_path / "reports"
    path = _write_report(reports, "hw.xml", _COLLECT_ERROR + _REAL_PASS, root="testsuite")

    assert len(hw.strip_collection_errors(str(reports))["hw.xml"]) == 1
    assert len(ET.parse(str(path)).getroot().findall("testcase")) == 1


def test_strip_collection_errors_survives_a_truncated_report(tmp_path):
    # Aggregation must publish whatever else it has rather than die here.
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "truncated.xml").write_text("<testsuites><testsuite>")
    _write_report(reports, "good.xml", _COLLECT_ERROR)

    assert list(hw.strip_collection_errors(str(reports))) == ["good.xml"]


def test_strip_collection_errors_survives_an_unreadable_report(tmp_path):
    # Not a parse failure: a directory that happens to end in .xml, and a report
    # that cannot be opened. Either one would otherwise abort the loop and leave
    # every other board its harness errors.
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "adirectory.xml").mkdir()
    unopenable = reports / "unopenable.xml"
    unopenable.write_text("<testsuites/>")
    unopenable.chmod(0o000)
    _write_report(reports, "good.xml", _COLLECT_ERROR)

    assert list(hw.strip_collection_errors(str(reports))) == ["good.xml"]


def test_strip_collection_errors_survives_a_read_only_report(tmp_path):
    # The rewrite is the other place this can fail. Named to sort first, so an
    # unwritable report would otherwise take the good one down with it.
    reports = tmp_path / "reports"
    read_only = _write_report(reports, "a_readonly.xml", _COLLECT_ERROR)
    read_only.chmod(0o444)
    _write_report(reports, "good.xml", _COLLECT_ERROR)

    assert "good.xml" in hw.strip_collection_errors(str(reports))


def test_strip_collection_errors_tolerates_a_missing_reports_dir(tmp_path):
    assert hw.strip_collection_errors(str(tmp_path / "absent")) == {}


def test_strip_collection_errors_cli_is_silent_when_reports_are_clean(tmp_path, capsys):
    # The HW pipeline treats empty output as "nothing to flag".
    reports = tmp_path / "reports"
    _write_report(reports, "hw.xml", _REAL_PASS)

    assert cli.main(["strip-collection-errors", str(reports)]) == 0
    assert capsys.readouterr().out == ""


def _packaging_skip(test_dir, missing="driver.py", name="test_type_execute"):
    # Built from the constant rather than spelled out, so the fixture cannot
    # drift from the prefix the harness actually writes.
    return (
        '<testcase classname="test_bnn_hw_pytest.TestBnn" name="%s[U250_%s]">'
        '<skipped type="pytest.skip" message="%s %s is missing %s"/></testcase>'
        % (name, test_dir, hw.PACKAGING_SKIP_PREFIX, test_dir, missing)
    )


_UNRELATED_SKIP = (
    '<testcase classname="test_bnn_hw_pytest.TestBnn" '
    'name="test_type_execute[U250_bnn_w1_a1_lfc_batchSize-1_platform-vitis-xrt]">'
    '<skipped type="pytest.skip" message="known Alveo weight-size issue"/></testcase>'
)


def test_packaging_skips_names_each_model_once_however_many_tests_it_skipped(tmp_path):
    # Both of a model's tests carry the same reason.
    reports = tmp_path / "reports"
    _write_report(
        reports,
        "bnn_build_full_hw_U250.xml",
        _packaging_skip("bnn_w1_a1_tfc")
        + _packaging_skip("bnn_w1_a1_tfc", name="test_type_throughput")
        + _packaging_skip("bnn_w2_a2_tfc", missing="driver.py, input.npy"),
    )

    assert hw.packaging_skips(str(reports)) == {
        "bnn_build_full_hw_U250.xml": [
            "bnn_w1_a1_tfc is missing driver.py",
            "bnn_w2_a2_tfc is missing driver.py, input.npy",
        ]
    }


def test_packaging_skips_ignores_a_skip_it_did_not_write(tmp_path):
    # A test skipped for an unrelated reason must not turn the build yellow.
    reports = tmp_path / "reports"
    _write_report(reports, "hw.xml", _UNRELATED_SKIP + _REAL_PASS)

    assert hw.packaging_skips(str(reports)) == {}


def test_packaging_skips_survive_the_collection_error_strip(tmp_path):
    # The strip rewrites the XML in place before this runs, so a packaging skip
    # has to still be there and still be attributed afterwards.
    reports = tmp_path / "reports"
    _write_report(reports, "hw.xml", _COLLECT_ERROR + _packaging_skip("bnn_w2_a2_tfc") + _REAL_FAIL)

    hw.strip_collection_errors(str(reports))

    assert hw.packaging_skips(str(reports)) == {"hw.xml": ["bnn_w2_a2_tfc is missing driver.py"]}


def test_packaging_skips_handles_a_bare_testsuite_root(tmp_path):
    reports = tmp_path / "reports"
    _write_report(reports, "hw.xml", _packaging_skip("bnn_w2_a2_tfc"), root="testsuite")

    assert hw.packaging_skips(str(reports)) == {"hw.xml": ["bnn_w2_a2_tfc is missing driver.py"]}


def test_packaging_skips_survives_a_truncated_report(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "truncated.xml").write_text("<testsuites><testsuite>")
    _write_report(reports, "good.xml", _packaging_skip("bnn_w2_a2_tfc"))

    assert list(hw.packaging_skips(str(reports))) == ["good.xml"]


def test_packaging_skips_tolerates_a_missing_reports_dir(tmp_path):
    assert hw.packaging_skips(str(tmp_path / "absent")) == {}


def test_packaging_skips_cli_is_silent_when_nothing_was_skipped(tmp_path, capsys):
    reports = tmp_path / "reports"
    _write_report(reports, "hw.xml", _REAL_PASS + _UNRELATED_SKIP)

    assert cli.main(["packaging-skips", str(reports)]) == 0
    assert capsys.readouterr().out == ""


def test_packaging_skips_cli_names_the_report_and_the_model(tmp_path, capsys):
    reports = tmp_path / "reports"
    _write_report(reports, "bnn_build_full_hw_U250.xml", _packaging_skip("bnn_w2_a2_tfc"))

    assert cli.main(["packaging-skips", str(reports)]) == 0
    out = capsys.readouterr().out.strip()
    assert out == "bnn_build_full_hw_U250.xml: bnn_w2_a2_tfc is missing driver.py"


def test_strip_collection_errors_cli_names_every_dropped_entry(tmp_path, capsys):
    reports = tmp_path / "reports"
    _write_report(reports, "hw.xml", _COLLECT_ERROR)

    assert cli.main(["strip-collection-errors", str(reports)]) == 0
    out = capsys.readouterr().out.strip()
    assert out == "hw.xml: bnn_w1_a1_tfc.external.xsimd.test.test_wasm"


def test_hw_config_json_cli_bundles_shards_types_and_labels(capsys):
    # The HW pipeline reads this one bundle in Validate and hands it to readJSON.
    rc = cli.main(["hw-config-json"])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert {"shards", "test_types", "labels"} <= set(parsed)
    assert parsed["shards"] == hw.hw_shards()
    assert "bnn_build_sanity" in parsed["test_types"]
    assert "bnn_build_full" in parsed["test_types"]
    assert parsed["labels"] == hw.hw_test_type_labels()


# ---------------------------------------------------------------------------
# On-board harness collection
# ---------------------------------------------------------------------------


def _load_hw_harness():
    # The harness is a pytest file the board runs, not a package module, so it is
    # loaded by path under a name pytest will not collect a second time. Its
    # numeric dependencies are the board's, not this suite's.
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    path = os.path.join(REPO_ROOT, "ci", "test_bnn_hw_pytest.py")
    spec = importlib.util.spec_from_file_location("bnn_hw_harness", path)
    module = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        # the board markers belong to the board's pytest run, so this suite has
        # no reason to register them just to import the file
        warnings.simplefilter("ignore", pytest.PytestUnknownMarkWarning)
        spec.loader.exec_module(module)
    return module


def _board_workspace(tmp_path):
    # A board workspace as unzipping leaves it: one whole model, one the build
    # packaged without a driver, and a cache directory that was never a test.
    whole = tmp_path / "bnn_w1_a1_cnv"
    whole.mkdir()
    (whole / "driver.py").write_text("")
    (whole / "input.npy").write_text("")
    broken = tmp_path / "bnn_w2_a2_tfc"
    broken.mkdir()
    (broken / "input.npy").write_text("")
    (broken / "CMakeLists.txt").write_text("")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "driver.cpython-311.pyc").write_text("")
    return tmp_path


class _FakeMetafunc:
    """Just enough of pytest's Metafunc to drive pytest_generate_tests."""

    def __init__(self, marker):
        self.config = SimpleNamespace(getoption=lambda name: marker)
        self.parametrized = None

    def parametrize(self, argnames, argvalues, ids, scope):
        self.parametrized = (argnames, argvalues, ids)


def test_find_model_dirs_tells_an_incomplete_model_from_a_cache(tmp_path):
    harness = _load_hw_harness()

    assert harness.find_model_dirs(str(_board_workspace(tmp_path))) == {
        "bnn_w1_a1_cnv": [],
        "bnn_w2_a2_tfc": ["driver.py"],
    }


def test_generate_tests_skips_an_incomplete_model_by_name(tmp_path, monkeypatch):
    # A model the build failed to package must stay in the report as a named
    # skip. Dropping it instead only shows up as a test count that fell.
    harness = _load_hw_harness()
    monkeypatch.chdir(_board_workspace(tmp_path))
    metafunc = _FakeMetafunc("ZCU104")

    harness.pytest_generate_tests(metafunc)

    argnames, argvalues, ids = metafunc.parametrized
    assert argnames == ["test_dir", "batch_size", "platform"]
    by_id = dict(zip(ids, argvalues))
    assert sorted(by_id) == [
        "ZCU104_bnn_w1_a1_cnv_batchSize-1_platform-zynq-iodma",
        "ZCU104_bnn_w2_a2_tfc_batchSize-1_platform-zynq-iodma",
    ]
    assert not getattr(by_id["ZCU104_bnn_w1_a1_cnv_batchSize-1_platform-zynq-iodma"], "marks", ())
    marks = by_id["ZCU104_bnn_w2_a2_tfc_batchSize-1_platform-zynq-iodma"].marks
    assert [m.name for m in marks] == ["skip"]
    # the prefix is what report aggregation keys on, so it leads the reason
    assert marks[0].kwargs["reason"] == "%s bnn_w2_a2_tfc is missing driver.py" % (
        hw.PACKAGING_SKIP_PREFIX,
    )
