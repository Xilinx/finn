# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Image identity, runtime inputs, Bake and image transport in docker/launcher.py."""
import pytest

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LAUNCHER = str(REPO / "docker/launcher.py")
sys.path.insert(0, str(REPO / "docker"))
import launcher  # noqa: E402

pytestmark = pytest.mark.util


@pytest.fixture
def clean(monkeypatch):
    for name in list(os.environ):
        if name.startswith("FINN_") or name in ("SLASHKIT_DEB_PACKAGE", "UBUNTU_TAG", "REGISTRY"):
            monkeypatch.delenv(name)


def image_root(tmp_path):
    root = tmp_path / "image-root"
    (root / "docker").mkdir(parents=True)
    (root / "docker/image-inputs.txt").write_text("docker/image-inputs.txt\nimage.txt\n")
    (root / "image.txt").write_text("environment\n")
    (root / "src").mkdir()
    (root / "src/finn.py").write_text("source v1\n")
    return root


def revision(root, runtimes=()):
    return launcher.image_revision(launcher.normalize_runtimes(list(runtimes)), root)


# --------------------------------------------------------------------------
# Image identity.
# --------------------------------------------------------------------------


def test_revision_tracks_image_inputs_and_build_args_not_sources(clean, tmp_path, monkeypatch):
    root = image_root(tmp_path)
    original = revision(root)
    assert re.fullmatch(r"img-[0-9a-f]{16}", original)
    (root / "src/finn.py").write_text("source v2\n")
    assert revision(root) == original  # the checkout is mounted, not built in
    (root / "image.txt").write_text("changed environment\n")
    changed = revision(root)
    assert changed != original
    monkeypatch.setenv("UBUNTU_TAG", "jammy-override")
    assert revision(root) != changed


def test_revision_ignores_umask_but_tracks_the_executable_bit(clean, tmp_path):
    root = image_root(tmp_path)
    source = root / "image.txt"
    source.chmod(0o644)
    original = revision(root)
    source.chmod(0o660)
    assert revision(root) == original
    source.chmod(0o755)
    assert revision(root) != original


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read mode-000 fixtures")
def test_unreadable_or_missing_inputs_fail(clean, tmp_path):
    root = image_root(tmp_path)
    (root / "image.txt").chmod(0)
    try:
        with pytest.raises(launcher.LaunchError, match="image.txt"):
            revision(root)
    finally:
        (root / "image.txt").chmod(0o644)
    (root / "docker/image-inputs.txt").write_text("docker/image-inputs.txt\nmissing/*.txt\n")
    with pytest.raises(launcher.LaunchError, match="matched no files"):
        revision(root)


@pytest.mark.parametrize("runtimes", [(), ("xrt",)])
def test_only_selected_supplied_packages_change_identity(clean, tmp_path, runtimes):
    root = image_root(tmp_path)
    manifests = root / "docker/runtimes"
    manifests.mkdir()
    (manifests / "xrt.env").write_text("SOURCE=fetch\n")
    (manifests / "slashkit.env").write_text("SOURCE=supply\nFILES=slashkit.deb\n")
    (root / "docker/image-inputs.txt").write_text(
        "docker/image-inputs.txt\nimage.txt\ndocker/runtimes/*.env\n"
    )
    baseline = revision(root, runtimes)
    packages = root / "docker/packages"
    packages.mkdir()
    (packages / "slashkit.deb").write_bytes(b"leftover private package")
    (packages / "unused.rpm").write_bytes(b"unselected archive")
    (packages / "unused.rpm").chmod(0)
    assert revision(root, runtimes) == baseline
    selected = revision(root, ("slashkit", "xrt", "slashkit"))
    assert selected == revision(root, ("xrt", "slashkit"))
    (packages / "slashkit.deb").write_bytes(b"different selected package")
    assert revision(root, ("slashkit", "xrt")) != selected
    assert revision(root, runtimes) == baseline
    (packages / "slashkit.deb").unlink()
    with pytest.raises(launcher.LaunchError, match="slashkit.deb"):
        revision(root, ("slashkit",))


def test_source_commit_changes_without_changing_the_image(clean, tmp_path):
    root = image_root(tmp_path)
    source = tmp_path / "source"
    source.mkdir()

    def git(*args):
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@localhost", *args],
            cwd=source,
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    (source / "README").write_text("source\n")
    git("add", "README")
    git("commit", "-q", "-m", "initial")
    before = launcher.make_request([], source_root=source, image_root=root)
    git("commit", "-q", "--allow-empty", "-m", "source-only")
    after = launcher.make_request([], source_root=source, image_root=root)
    assert before.revision == after.revision
    assert before.source_revision != after.source_revision
    assert before.source_describe != after.source_describe
    assert before.source_dirty == after.source_dirty == "0"
    (source / "README").write_text("edited\n")
    assert launcher.make_request([], source_root=source, image_root=root).source_dirty == "1"


def test_cli_and_library_identity_agree(clean):
    """The committed tree hashes the same in-process and through the CLI."""
    proc = subprocess.run(
        [sys.executable, LAUNCHER, "build", "--print"],
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp")},
    )
    assert proc.returncode == 0, proc.stderr
    printed = dict(line.split("=", 1) for line in proc.stdout.splitlines())
    assert printed["image_revision"] == launcher.image_revision([])


# --------------------------------------------------------------------------
# Bake targets and tags.
# --------------------------------------------------------------------------


def test_targets_and_tags(clean):
    request = launcher.Request(runtimes=[], revision="img-1", source_revision="a" * 40)
    assert (request.target, request.tag) == ("finn", "xilinx/finn:img-1")
    request.runtimes = ["xrt"]
    assert (request.target, request.tag) == ("finn-xrt", "xilinx/finn:img-1.xrt")
    request.runtimes = ["slash", "xrt"]
    assert (request.target, request.tag) == ("finn-runtime", "xilinx/finn:img-1.slash.xrt")
    request.variant, request.source_dirty = "release", "1"
    assert request.target == "finn-release"
    assert request.tag == "xilinx/finn:release-img-1-aaaaaaaaaaaa-dirty.slash.xrt"
    bake = (REPO / "docker-bake.hcl").read_text()
    for target in ("finn", "finn-xrt", "finn-runtime", "finn-release"):
        assert 'target "%s"' % target in bake


@pytest.mark.skipif(
    subprocess.run(["docker", "buildx", "bake", "--help"], capture_output=True).returncode != 0
    if shutil.which("docker")
    else True,
    reason="needs Docker Buildx",
)
@pytest.mark.parametrize(
    "runtimes, variant", [((), ""), (("xrt",), ""), (("slash", "xrt"), ""), (("xrt",), "release")]
)
def test_tags_match_bake(clean, runtimes, variant):
    """The launcher names images itself; direct Bake users get the same names."""
    request = launcher.Request(
        runtimes=list(runtimes),
        variant=variant,
        revision="img-0123456789abcdef",
        source_revision="b" * 40,
        source_dirty="0",
    )
    proc = subprocess.run(
        ["docker", "buildx", "bake", "-f", "docker-bake.hcl", "--print", request.target],
        cwd=REPO,
        capture_output=True,
        text=True,
        env=dict(os.environ, **request.environment()),
    )
    assert proc.returncode == 0, proc.stderr
    printed = json.loads(proc.stdout[proc.stdout.index("{") :])
    assert printed["target"][request.target]["tags"] == [request.tag]


def test_missing_bake_has_actionable_error(clean, tmp_path, monkeypatch):
    tools = tmp_path / "bin"
    tools.mkdir()
    (tools / "docker").write_text('#!/bin/sh\necho "docker: buildx unavailable" >&2\nexit 1\n')
    (tools / "docker").chmod(0o755)
    proc = subprocess.run(
        [str(REPO / "docker/build")],
        capture_output=True,
        text=True,
        env={"PATH": str(tools) + os.pathsep + os.environ["PATH"], "HOME": str(tmp_path)},
    )
    assert proc.returncode != 0
    assert "docker-buildx-plugin" in proc.stderr
    assert "Traceback" not in proc.stderr


# --------------------------------------------------------------------------
# Runtime inputs.
# --------------------------------------------------------------------------


def test_slashkit_site_input_is_staged_and_selected(clean, tmp_path, monkeypatch):
    package = tmp_path / "site slashkit.deb"
    package.write_bytes(b"private package")
    monkeypatch.setenv("SLASHKIT_DEB_PACKAGE", str(package))
    assert launcher.prepare_runtimes(["xrt"], tmp_path) == ["slashkit", "xrt"]
    assert (tmp_path / "docker/packages/slashkit.deb").read_bytes() == package.read_bytes()
    package.unlink()
    with pytest.raises(launcher.LaunchError, match="SLASHKIT_DEB_PACKAGE"):
        launcher.prepare_runtimes(["xrt"], tmp_path)


def test_xrt_host_cache_is_verified_and_reused(clean, tmp_path, monkeypatch):
    payload = tmp_path / "public.deb"
    payload.write_bytes(b"public pinned XRT package")
    manifests = tmp_path / "docker/runtimes"
    manifests.mkdir(parents=True)
    (manifests / "xrt.env").write_text(
        "SOURCE=fetch\nDEB=pinned-xrt\nURL=%s\nFALLBACK_URL=\nSHA256=%s\n"
        % (payload.as_uri(), launcher.sha256(payload))
    )
    cache = tmp_path / "host cache"
    monkeypatch.setenv("FINN_XRT_PATH", str(cache))
    launcher.cache_xrt(["xrt"], tmp_path)
    staged = tmp_path / "docker/runtime-cache/pinned-xrt.deb"
    assert staged.read_bytes() == payload.read_bytes()
    payload.unlink()  # A second build works without its original download server.
    staged.unlink()
    launcher.cache_xrt(["xrt"], tmp_path)
    assert staged.is_file()
    (cache / "pinned-xrt.deb").write_bytes(b"corrupted")
    with pytest.raises(launcher.LaunchError, match="checksum mismatch"):
        launcher.cache_xrt(["xrt"], tmp_path)


def test_release_source_omits_supplied_runtime_archives(tmp_path):
    source = tmp_path / "source"
    packages = source / "docker/packages"
    packages.mkdir(parents=True)
    (packages / "private.deb").write_text("supplied private runtime")
    (packages / "private.rpm").write_text("supplied private runtime")
    (packages / "README.md").write_text("package instructions")
    (source / "compiler.py").write_text("compiler source")
    (source / "compiler.py").chmod(0o600)
    (source / ".gitignore").write_text("generated.onnx\n")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "-f", "."], check=True)
    (source / "generated.onnx").write_text("ignored model download")
    (source / "untracked.txt").write_text("not part of the source snapshot")
    (source / "compiler.py").write_text("edited tracked source")
    destination = tmp_path / "prepared"
    archive = tmp_path / "source.tar"
    launcher.snapshot_source(source, archive)
    subprocess.run(
        ["sh", str(REPO / "docker/copy-source.sh"), str(archive), str(destination)], check=True
    )
    assert (destination / "compiler.py").read_text() == "edited tracked source"
    assert (destination / "compiler.py").stat().st_mode & 0o004
    assert (destination / "docker/packages/README.md").is_file()
    assert not list(destination.rglob("*.deb")) and not list(destination.rglob("*.rpm"))
    assert not (destination / "generated.onnx").exists()
    assert not (destination / "untracked.txt").exists()


# --------------------------------------------------------------------------
# Image transport.
# --------------------------------------------------------------------------

IMAGE_ID = "sha256:" + "a" * 64


@pytest.fixture
def request_(clean):
    return launcher.Request(runtimes=[], revision="img-review", source_revision="a" * 40)


def publication(directory, request, archive=b"archive", **changes):
    info = {
        "schema": 1,
        "image_id": IMAGE_ID,
        "environment_revision": request.revision,
        "lock_sha256": launcher.sha256(REPO / "uv.lock"),
        "runtimes": request.runtimes,
        "source_revision": request.source_revision,
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
    }
    info.update(changes)
    launcher.write_metadata(directory, "descriptive-tag", info)
    (directory / launcher.ARCHIVE).write_bytes(archive)
    return info


def test_metadata_rejects_partial_stale_and_legacy_publications(request_, tmp_path):
    directory = tmp_path / "publication"
    publication(directory, request_)
    assert launcher.read_metadata(directory, request_)[0] == IMAGE_ID
    publication(directory, request_, environment_revision="another-environment")
    with pytest.raises(launcher.LaunchError, match="Published environment_revision"):
        launcher.read_metadata(directory, request_)
    publication(directory, request_)
    (directory / launcher.FILES[1]).write_text("sha256:" + "c" * 64)
    with pytest.raises(launcher.LaunchError, match="identity"):
        launcher.read_metadata(directory, request_)
    (directory / launcher.FILES[2]).unlink()
    with pytest.raises(launcher.LaunchError, match="legacy two-file"):
        launcher.read_metadata(directory, request_)


def test_republication_accepts_only_an_intact_archive(request_, tmp_path, monkeypatch):
    """A retry reuses the existing publication only if shards can load it."""
    directory = tmp_path / "publication"
    info = publication(directory, request_)
    monkeypatch.setattr(launcher, "image_metadata", lambda reference, request: dict(info))
    monkeypatch.setattr(launcher, "docker", lambda *args: pytest.fail("republished"))
    workspace = tmp_path / "workspace"
    launcher.publish_image(directory, IMAGE_ID, workspace, request_)
    assert json.loads((workspace / launcher.FILES[2]).read_text()) == info
    shutil.rmtree(workspace)
    (directory / launcher.ARCHIVE).write_bytes(b"truncated")
    with pytest.raises(launcher.LaunchError, match="missing or damaged archive"):
        launcher.publish_image(directory, IMAGE_ID, workspace, request_)
    (directory / launcher.ARCHIVE).unlink()
    with pytest.raises(launcher.LaunchError, match="missing or damaged archive"):
        launcher.publish_image(directory, IMAGE_ID, workspace, request_)
    assert not workspace.exists()


def test_cached_image_avoids_archive_read_and_load(request_, tmp_path, monkeypatch):
    directory = tmp_path / "publication"
    publication(directory, request_)
    monkeypatch.setattr(launcher, "_loaded", lambda image, request: True)
    monkeypatch.setattr(launcher, "_docker_load", lambda *args: pytest.fail("loaded"))
    assert launcher.load_image(directory, request_) == IMAGE_ID
    (directory / launcher.ARCHIVE).unlink()
    with pytest.raises(launcher.LaunchError, match="archive is missing"):
        launcher.load_image(directory, request_)


def test_concurrent_misses_load_archive_once(request_, tmp_path, monkeypatch):
    directory = tmp_path / "publication"
    publication(directory, request_)
    monkeypatch.setenv("FINN_DOCKER_LOAD_LOCK", str(tmp_path / "load.lock"))
    state = {"loaded": False, "loads": 0}
    monkeypatch.setattr(launcher, "_loaded", lambda image, request: state["loaded"])

    def load_once(archive, expected):
        state["loads"] += 1
        time.sleep(0.05)
        state["loaded"] = True

    monkeypatch.setattr(launcher, "_docker_load", load_once)
    with ThreadPoolExecutor(max_workers=6) as executor:
        loads = list(executor.map(lambda _: launcher.load_image(directory, request_), range(6)))
    assert loads == [IMAGE_ID] * 6
    assert state["loads"] == 1


def fake_docker(tools, script):
    tools.mkdir(exist_ok=True)
    (tools / "docker").write_text(
        "#!%s\nimport os, pathlib, sys, time\n%s" % (sys.executable, script)
    )
    (tools / "docker").chmod(0o755)


def test_load_streams_and_verifies_the_archive(request_, tmp_path, monkeypatch):
    directory = tmp_path / "publication"
    publication(directory, request_, archive=b"archive bytes")
    received = tmp_path / "received"
    fake_docker(
        tmp_path / "bin",
        "pathlib.Path(%r).write_bytes(sys.stdin.buffer.read())\n" % str(received),
    )
    monkeypatch.setenv("PATH", str(tmp_path / "bin") + os.pathsep + os.environ["PATH"])
    launcher._docker_load(
        directory / launcher.ARCHIVE, hashlib.sha256(b"archive bytes").hexdigest()
    )
    assert received.read_bytes() == b"archive bytes"
    received.unlink()
    with pytest.raises(launcher.LaunchError, match="checksum mismatch"):
        launcher._docker_load(directory / launcher.ARCHIVE, "0" * 64)


@pytest.mark.parametrize("pigz", [True, False])
def test_compression_is_gzip_and_hashed_once(tmp_path, monkeypatch, pigz):
    if pigz and not shutil.which("pigz"):
        pytest.skip("pigz is not installed")
    if not pigz:
        monkeypatch.setattr(launcher.shutil, "which", lambda name: None)
    payload = os.urandom(3 << 20) + b"x" * (5 << 20)
    source = tmp_path / "payload"
    source.write_bytes(payload)
    with source.open("rb") as handle, (tmp_path / "out.gz").open("wb") as output:
        digest = launcher.compress(handle, output)
    assert digest == launcher.sha256(tmp_path / "out.gz")
    import gzip  # noqa: PLC0415

    assert gzip.decompress((tmp_path / "out.gz").read_bytes()) == payload


def test_publication_sigterm_cleans_partial_archive(clean, tmp_path, monkeypatch):
    inspection = [
        {
            "Id": IMAGE_ID,
            "Config": {
                "Labels": {
                    "dev.finn.image-revision": "img-review",
                    "dev.finn.source-mode": "checkout",
                    "dev.finn.runtimes": "",
                }
            },
        }
    ]
    fake_docker(
        tmp_path / "bin",
        "args=sys.argv[1:]\n"
        'if args[:2] == ["image", "inspect"]: print(%r)\n' % json.dumps(inspection)
        + 'elif args[:1] == ["save"]:\n'
        " sys.stdout.buffer.write(os.urandom(2000000)); sys.stdout.flush(); time.sleep(60)\n"
        'elif args[:2] == ["image", "rm"]: pathlib.Path(%r).touch()\n'
        % str(tmp_path / "tag-removed"),
    )
    target = tmp_path / "images/job/1"
    process = subprocess.Popen(
        [
            sys.executable,
            LAUNCHER,
            "image",
            "publish",
            str(target),
            IMAGE_ID,
            "--workspace",
            str(tmp_path / "workspace"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            "PATH": str(tmp_path / "bin") + os.pathsep + os.environ["PATH"],
            "HOME": str(tmp_path),
            "FINN_IMAGE_REVISION": "img-review",
        },
    )
    try:
        for _ in range(100):
            partials = list(target.parent.glob(".finn-image-*/publication/" + launcher.ARCHIVE))
            if partials and partials[0].stat().st_size > 0:
                break
            if process.poll() is not None:
                pytest.fail(str(process.communicate()))
            time.sleep(0.05)
        else:
            pytest.fail("Publication did not start")
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=60)
        assert process.returncode == 143, (stdout, stderr)
        assert not list(target.parent.glob(".finn-image-*"))
        assert (tmp_path / "tag-removed").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


# --------------------------------------------------------------------------
# Host Python: the launcher runs before FINN's Python 3.12 environment exists.
# --------------------------------------------------------------------------


@pytest.mark.skipif(not shutil.which("python3.10"), reason="needs python3.10")
def test_launcher_runs_on_python_3_10(tmp_path):
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "FINN_XILINX_ENV": ""}
    for args in (["build", "--print"], ["run", "--print"], ["inspect", "--format", "sh"]):
        proc = subprocess.run(
            ["python3.10", "-I", LAUNCHER, *args], capture_output=True, text=True, env=env
        )
        assert proc.returncode == 0, (args, proc.stderr)
    source = (REPO / "docker/launcher.py").read_text() + (REPO / "docker/identity.py").read_text()
    for unavailable in ("tomllib", "file_digest", "datetime.UTC", "ExceptionGroup"):
        assert unavailable not in source
