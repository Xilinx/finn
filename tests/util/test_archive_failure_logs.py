# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""CI bundles retain launcher diagnostics without widening generic file capture."""

import pytest

import os
import subprocess
import tarfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "ci/scripts/archive_failure_logs.sh"


def write_file(root, name):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(name)
    return path


@pytest.mark.util
def test_failure_archive_includes_launcher_transcripts_in_build_families(tmp_path):
    build = tmp_path / "build with spaces"
    included = [
        "code_gen_cppsim_node 1/compile.stderr.log",
        "code_gen_ipgen_node/ipgen.stdout.log",
        "rtlsim_snapshot/compile_xelab.stderr.log",
        "rtlsim_snapshot/rtlsim_xsi.stdout.log",
        "vivado_pynq_proj_test/synth_project.stderr.log",
        "vivado_stitch_proj_test/make_project.stdout.log",
        "vitis_link_proj_test/gen_report_xml.stderr.log",
        "slash_link_proj_test/run_slash_link.stderr.log",
        "cpp_driver_logs_test/git-fetch.stderr.log",
    ]
    expected = [write_file(build, name) for name in included]
    excluded = [
        "unrelated/compile.stderr.log",
        "mycode_gen_test/ipgen.stdout.log",
        "myvitis_link_proj_test/config.txt",
        "cpp_driver_test/source.cpp",
    ]
    for name in excluded:
        write_file(build, name)
    source = tmp_path / "source"
    expected.append(write_file(source, "finn_xsi/compile_xsi.stderr.log"))
    expected.append(write_file(source, "finn_xsi/compile_xsi.sh"))
    write_file(source, "finn_xsi/unrelated.stderr.log")
    bundle = tmp_path / "failure.tar.gz"
    env = {**os.environ, "FINN_ROOT": str(source), "FINN_LSF_NFS_STAGING": ""}
    subprocess.run(["bash", str(SCRIPT), str(build), str(bundle)], env=env, check=True)
    with tarfile.open(bundle) as archive:
        assert set(archive.getnames()) == {str(path).lstrip("/") for path in expected}
        for path in expected:
            assert archive.extractfile(str(path).lstrip("/")).read() == path.read_bytes()


@pytest.mark.util
def test_xsi_source_diagnostics_are_scoped_by_the_shard_marker(tmp_path):
    build = tmp_path / "build"
    build.mkdir()
    source = tmp_path / "source"
    old = write_file(source, "finn_xsi/compile_xsi.stdout.log")
    os.utime(old, (1, 1))
    marker = write_file(tmp_path, "start")
    os.utime(marker, (2, 2))
    recent = write_file(source, "finn_xsi/compile_xsi.stderr.log")
    bundle = tmp_path / "failure.tar.gz"
    env = {**os.environ, "FINN_ROOT": str(source), "FINN_LSF_NFS_STAGING": ""}
    subprocess.run(["bash", str(SCRIPT), str(build), str(bundle), str(marker)], env=env, check=True)
    with tarfile.open(bundle) as archive:
        assert archive.getnames() == [str(recent).lstrip("/")]
