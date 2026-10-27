# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Regression checks for upstream environment/CI review findings."""
import pytest

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from packaging.requirements import Requirement
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docker"))
import config  # noqa: E402
import image as images  # noqa: E402
import launch  # noqa: E402
import runtime_inputs  # noqa: E402

pytestmark = pytest.mark.util


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.startswith(("FINN_", "XILINX_")) or name in (
            "BASH_ENV",
            "ENV",
            "SLASHKIT_DEB_PACKAGE",
            "PLATFORM_REPO_PATHS",
            "HLS_PATH",
            "VIVADO_PATH",
            "VITIS_PATH",
            "LD_PRELOAD",
        ):
            monkeypatch.delenv(name)
    monkeypatch.setenv("FINN_XILINX_ENV", "")
    monkeypatch.setenv("FINN_ROOT", str(ROOT))
    monkeypatch.setenv("FINN_HOST_BUILD_DIR", str(tmp_path / "build"))
    monkeypatch.setenv("FINN_HOME", str(tmp_path / "state"))


def test_missing_optional_platforms_and_installation_are_skipped(clean_env, tmp_path, monkeypatch):
    platform = tmp_path / "real-platform"
    platform.mkdir()
    hls = tmp_path / "HLS"
    hls.mkdir()
    monkeypatch.setenv("FINN_XILINX_PATH", str(tmp_path / "not-installed"))
    monkeypatch.setenv("PLATFORM_REPO_PATHS", str(tmp_path / "absent") + ":" + str(platform))
    monkeypatch.setenv("HLS_PATH", str(hls))
    request = config.resolve_host("build", "mirror")
    assert request["env"]["PLATFORM_REPO_PATHS"] == str(platform)
    assert request["env"]["XILINX_HLS"] == str(hls)
    assert all(Path(m["source"]).is_dir() for m in request["mounts"])


def test_native_hls_only_activation_does_not_fetch_boards(clean_env, tmp_path, monkeypatch):
    root = tmp_path / "tools"
    (root / "Vitis_HLS/2024.2").mkdir(parents=True)
    library = tmp_path / "hlslib"
    library.mkdir()
    monkeypatch.setenv("FINN_XILINX_PATH", str(root))
    monkeypatch.setenv("FINN_XILINX_VERSION", "2024.2")
    monkeypatch.setenv("FINN_RESOURCES_HLSLIB", str(library))
    monkeypatch.setenv("FINN_RESOURCES_OFFLINE", "1")
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source scripts/activate.sh && test -n "$XILINX_HLS" && test -z "${XILINX_VIVADO:-}"',
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not (Path(os.environ["FINN_HOME"]) / "resources").exists()


def test_farm_wrappers_do_not_shadow_bare_tools(clean_env, tmp_path, monkeypatch):
    wrappers = tmp_path / "wrappers"
    wrappers.mkdir()
    for tool in ("make", "vivado"):
        path = wrappers / tool
        path.write_text("#!/bin/sh\nexit 99\n")
        path.chmod(0o755)
    monkeypatch.setenv("FINN_TOOL_DIR_OVERRIDE", str(wrappers))
    result = subprocess.run(
        ["bash", "-c", ". docker/finn-toolchain.sh; command -v make"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() != str(wrappers / "make")
    from finn.util.basic import resolve_xilinx_tool  # noqa: PLC0415

    assert resolve_xilinx_tool("vivado") == str(wrappers / "vivado")


@pytest.mark.parametrize("terminal", [True, False])
def test_explicit_commands_receive_terminal_when_attached(clean_env, monkeypatch, terminal):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: terminal)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: terminal)
    args = argparse.Namespace(
        tier="dev",
        image="image",
        volume=[],
        env=[],
        workdir=None,
        name=None,
        notebook=False,
        command=["--", "ipython"],
    )
    argv = launch.command(args)
    assert ("-i" in argv) is terminal
    assert ("-t" in argv) is terminal
    assert argv[-1] == "ipython"


def test_candidate_failure_cannot_publish_previous_results(clean_env, monkeypatch):
    # Use an isolated checkout-shaped cwd so real reports are not removed.
    root = Path(os.environ["FINN_HOME"]).parent / "candidate"
    root.mkdir()
    for name in ("brevitas_suite.xml", "brevitas_suite.html"):
        (root / name).write_text("old report")
    for name in ("brevitas-provenance", "coverage_brevitas_suite"):
        (root / name).mkdir()
        (root / name / "old.json").write_text("old evidence")
    monkeypatch.setenv("FINN_ROOT", str(root))
    monkeypatch.setenv("BREVITAS_COMMIT", "invalid")
    result = subprocess.run(
        ["bash", str(ROOT / "ci/scripts/brevitas_candidate.sh")], capture_output=True
    )
    assert result.returncode == 2
    assert list(root.iterdir()) == []


@pytest.fixture
def publication(clean_env, tmp_path, monkeypatch):
    monkeypatch.setenv("FINN_IMAGE_REVISION", "img-review")
    directory = tmp_path / "publication"
    info = {
        "schema": 1,
        "image_id": "sha256:" + "a" * 64,
        "environment_revision": "img-review",
        "lock_sha256": images.sha256(ROOT / "uv.lock"),
        "runtimes": [],
        "archive_sha256": hashlib.sha256(b"archive").hexdigest(),
    }
    images.write_metadata(directory, "descriptive-tag", info)
    (directory / images.ARCHIVE).write_bytes(b"archive")
    monkeypatch.setenv("FINN_DOCKER_LOAD_LOCK", str(tmp_path / "load.lock"))
    return directory, info["image_id"]


def test_cached_image_avoids_archive_read_and_load(publication, monkeypatch):
    directory, identity = publication
    monkeypatch.setattr(images, "_loaded", lambda _: True)
    original = images.sha256

    def no_archive_read(path):
        assert Path(path).name != images.ARCHIVE
        return original(path)

    monkeypatch.setattr(images, "sha256", no_archive_read)
    assert images.load(directory) == identity
    (directory / images.ARCHIVE).unlink()
    with pytest.raises(ValueError, match="archive is missing"):
        images.load(directory)


def test_concurrent_misses_load_archive_once(publication, monkeypatch):
    directory, identity = publication
    state = {"loaded": False, "loads": 0}
    monkeypatch.setattr(images, "_loaded", lambda _: state["loaded"])

    def load_once(argv, **kwargs):
        assert argv[:2] == ["docker", "load"]
        state["loads"] += 1
        time.sleep(0.05)
        state["loaded"] = True
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(images.subprocess, "run", load_once)
    with ThreadPoolExecutor(max_workers=6) as executor:
        assert list(executor.map(lambda _: images.load(directory), range(6))) == [identity] * 6
    assert state["loads"] == 1


def test_publication_sigterm_cleans_partial_archive(clean_env, tmp_path, monkeypatch):
    tools = tmp_path / "bin"
    tools.mkdir()
    daemon = tools / "docker"
    image_id = "sha256:" + "a" * 64
    inspection = [
        {
            "Id": image_id,
            "Config": {
                "Labels": {
                    "dev.finn.image-revision": "img-review",
                    "dev.finn.source-mode": "checkout",
                    "dev.finn.runtimes": "",
                }
            },
        }
    ]
    daemon.write_text(
        "#!"
        + sys.executable
        + "\nimport os, pathlib, sys, time\n"
        + "args=sys.argv[1:]\n"
        + 'if args[:2] == ["image", "inspect"]: print('
        + repr(json.dumps(inspection))
        + ")\n"
        + 'elif args[:1] == ["save"]:\n'
        + " sys.stdout.buffer.write(os.urandom(2000000)); sys.stdout.flush(); time.sleep(60)\n"
        + 'elif args[:2] == ["image", "rm"]: pathlib.Path('
        + repr(str(tmp_path / "tag-removed"))
        + ").touch()\n"
    )
    daemon.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("FINN_IMAGE_REVISION", "img-review")
    monkeypatch.setenv("FINN_SOURCE_REVISION", "a" * 40)
    monkeypatch.setenv("FINN_SOURCE_DIRTY", "0")
    target = tmp_path / "images/job/1"
    process = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "docker/image.py"),
            "publish",
            str(target),
            image_id,
            "--workspace",
            str(tmp_path / "workspace"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        for _ in range(100):
            partials = list(target.parent.glob(".finn-image-*/publication/" + images.ARCHIVE))
            if partials and partials[0].stat().st_size > 0:
                break
            if process.poll() is not None:
                pytest.fail(str(process.communicate()))
            time.sleep(0.05)
        else:
            pytest.fail("Publication did not start")
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 143, (stdout, stderr)
        assert not list(target.parent.glob(".finn-image-*"))
        assert (tmp_path / "tag-removed").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def test_xsi_key_changes_with_flags_and_pybind_version(clean_env, monkeypatch):
    from finn.xsi import _build_config, paths  # noqa: PLC0415

    baseline = paths.build_dir()
    original = paths.get_build_paths
    monkeypatch.setattr(paths, "get_build_paths", lambda: (original()[0], original()[1], ["-O0"]))
    assert paths.build_dir() != baseline
    monkeypatch.setattr(paths, "get_build_paths", original)
    version = paths.importlib.metadata.version
    monkeypatch.setattr(
        paths.importlib.metadata,
        "version",
        lambda name: "different" if name == "pybind11" else version(name),
    )
    assert paths.build_dir() != baseline
    assert _build_config.LINK_ARGS


def test_slashkit_site_input_is_staged_and_selected(clean_env, tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_inputs, "ROOT", tmp_path)
    package = tmp_path / "site slashkit.deb"
    package.write_bytes(b"private package")
    monkeypatch.setenv("SLASHKIT_DEB_PACKAGE", str(package))
    assert runtime_inputs.prepare("xrt") == "slashkit,xrt"
    assert (tmp_path / "docker/packages/slashkit.deb").read_bytes() == package.read_bytes()
    package.unlink()
    with pytest.raises(ValueError, match="SLASHKIT_DEB_PACKAGE"):
        runtime_inputs.prepare("xrt")


def test_xrt_host_cache_is_verified_and_reused(clean_env, tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_inputs, "ROOT", tmp_path)
    payload = tmp_path / "public.deb"
    payload.write_bytes(b"public pinned XRT package")
    manifests = tmp_path / "docker/runtimes"
    manifests.mkdir(parents=True)
    (manifests / "xrt.env").write_text(
        "SOURCE=fetch\nDEB=pinned-xrt\nURL="
        + payload.as_uri()
        + "\nFALLBACK_URL=\nSHA256="
        + runtime_inputs.checksum(payload)
        + "\n"
    )
    cache = tmp_path / "host cache"
    monkeypatch.setenv("FINN_XRT_PATH", str(cache))
    runtime_inputs.cache_xrt("xrt")
    staged = tmp_path / "docker/runtime-cache/pinned-xrt.deb"
    assert staged.read_bytes() == payload.read_bytes()
    payload.unlink()  # A second build works without its original download server.
    staged.unlink()
    runtime_inputs.cache_xrt("xrt")
    assert staged.is_file()
    (cache / "pinned-xrt.deb").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum mismatch"):
        runtime_inputs.cache_xrt("xrt")


def test_missing_bake_has_actionable_error(clean_env, tmp_path, monkeypatch):
    tools = tmp_path / "bin"
    tools.mkdir()
    docker = tools / "docker"
    docker.write_text('#!/bin/sh\necho "docker: buildx unavailable" >&2\nexit 1\n')
    docker.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])
    result = subprocess.run(
        [str(ROOT / "docker/build"), "--print-tag"], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "docker-buildx-plugin" in result.stderr
    assert "Traceback" not in result.stderr


def test_builder_imports_without_development_packages(clean_env):
    uv = shutil.which("uv")
    if not uv:
        pytest.skip("uv is needed to export the runtime profile")
    exported = subprocess.check_output(
        [uv, "export", "--locked", "--no-default-groups", "--no-emit-project", "--no-hashes"],
        cwd=ROOT,
        text=True,
    )
    allowed = {"finn"}
    for line in exported.splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "-")):
            allowed.add(Requirement(line).name.lower().replace("_", "-"))
    code = """
import importlib.metadata as md
from importlib.machinery import PathFinder
import json, sys
allowed = set(json.loads(sys.argv[1]))
packages = md.packages_distributions()
original = PathFinder.find_spec
def runtime_spec(fullname, path=None, target=None):
    distributions = packages.get(fullname.split('.')[0], [])
    if distributions and not any(n.lower().replace('_','-') in allowed for n in distributions):
        return None
    return original(fullname, path, target)
PathFinder.find_spec = staticmethod(runtime_spec)
assert PathFinder.find_spec('pytest') is None
from finn.builder import build_dataflow
"""
    subprocess.run([sys.executable, "-c", code, json.dumps(sorted(allowed))], cwd=ROOT, check=True)


@pytest.mark.skipif(
    os.environ.get("FINN_TEST_DOCKER_ENVIRONMENT") != "1", reason="opt-in Docker environment"
)
def test_routed_hls_tcl_uses_host_visible_library(clean_env, tmp_path, monkeypatch):
    wrappers = tmp_path / "farm-tools"
    wrappers.mkdir()
    (wrappers / "make").write_text("#!/bin/sh\nexit 99\n")
    (wrappers / "make").chmod(0o755)
    monkeypatch.setenv("FINN_TOOL_DIR_OVERRIDE", str(wrappers))
    monkeypatch.setenv("FINN_RESOURCES_OFFLINE", "1")
    monkeypatch.setenv("FINN_DOCKER_EXTRA", "--network none")
    code = """
import os, subprocess
from pathlib import Path
from types import SimpleNamespace
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
folder = Path(os.environ['FINN_BUILD_DIR']) / 'hls-probe'
folder.mkdir()
node = SimpleNamespace(onnx_node=SimpleNamespace(name='probe'), code_gen_dict={},
    get_nodeattr=lambda name: str(folder), get_ap_int_max_w=lambda: 1024,
    generate_params=lambda *args: None, global_includes=lambda: None,
    defines=lambda *args: None, blackboxfunction=lambda: None, pragmas=lambda: None,
    docompute=lambda: None, ipgen_default_directives=lambda: [], ipgen_extra_directives=lambda: [])
HLSBackend.code_generation_ipgen(node, None, 'part', 5)
assert os.environ['FINN_HLSLIB_PATH'].startswith(os.environ['FINN_HOME'] + '/resources/')
subprocess.run(['make', '--version'], check=True)
"""
    subprocess.run(
        [
            str(ROOT / "docker/run"),
            "--fpga",
            "--runtime",
            "xrt",
            "--no-build",
            "--",
            "python",
            "-c",
            code,
        ],
        cwd=ROOT,
        check=True,
    )
    build = Path(os.environ["FINN_HOST_BUILD_DIR"])
    tcl = (build / "hls-probe/hls_syn_probe.tcl").read_text()
    line = next(line for line in tcl.splitlines() if line.startswith("set config_bnnlibdir "))
    library = Path(line.split('"')[1])
    assert library.is_relative_to(build / ".finn/resources")
    # This process is outside Docker, as a routed farm invocation would be.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import pathlib,sys;assert (pathlib.Path(sys.argv[1])/'bnn-library.h').is_file()",
            str(library),
        ],
        check=True,
    )
