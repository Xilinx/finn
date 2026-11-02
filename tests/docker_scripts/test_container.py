# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Opt-in checks against a real Docker daemon.

FINN_TEST_DOCKER_TRANSPORT=1 exercises image publication and loading with small,
explicitly labelled fixture images. FINN_TEST_DOCKER_ENVIRONMENT=1 runs a
prepared FINN image (`docker/build --runtime xrt` first) with the Xilinx tools.
"""
import pytest

import json
import os
import shutil
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "docker"))
import launcher  # noqa: E402

pytestmark = pytest.mark.util


@pytest.mark.skipif(
    os.environ.get("FINN_TEST_DOCKER_TRANSPORT") != "1", reason="opt-in Docker transport"
)
def test_transport_uses_published_identity_and_rejects_corruption(tmp_path):
    """Real save/load, a stale tag, concurrent loads and publication retries."""
    key = uuid.uuid4().hex
    tag = "finn-transport-test:" + key
    request = launcher.Request(runtimes=[], revision="fixture-" + key, source_revision="a" * 40)
    context = tmp_path / "context"
    context.mkdir()
    (context / "Dockerfile").write_text(
        'FROM scratch\nLABEL dev.finn.image-revision="' + request.revision + '"\n'
        'LABEL dev.finn.runtimes="" dev.finn.source-mode="checkout"\nCOPY marker /marker\n'
    )
    publication = tmp_path / "shared"
    workspace = tmp_path / "workspace"
    created = []
    published_tag = None

    def build(text):
        (context / "marker").write_text(text)
        subprocess.run(
            ["docker", "build", "-t", tag, str(context)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
        identity = launcher.docker("image", "inspect", "--format", "{{.Id}}", tag)
        created.append(identity)
        return identity

    try:
        original = build("original artifact\n")
        launcher.publish_image(publication, tag, workspace, request)
        published_tag = (publication / launcher.FILES[0]).read_text().strip()
        replacement = build("stale local artifact\n")
        assert original != replacement
        # Remove the published identity, then create a same-tag collision.
        # Loads must recover the archive's original bytes.
        subprocess.run(["docker", "image", "rm", original], capture_output=True)
        launcher.docker("tag", replacement, published_tag)
        with ThreadPoolExecutor(max_workers=2) as executor:
            loads = list(
                executor.map(lambda _: launcher.load_image(publication, request), range(2))
            )
        assert loads == [original] * 2
        assert launcher.docker("image", "inspect", "--format", "{{.Id}}", published_tag) == original
        assert launcher.docker("image", "inspect", "--format", "{{.Id}}", tag) == replacement
        launcher.publish_image(publication, original, workspace, request)
        assert json.loads((workspace / launcher.FILES[2]).read_text()) == json.loads(
            (publication / launcher.FILES[2]).read_text()
        )
        with pytest.raises(launcher.LaunchError, match="another image/source"):
            launcher.publish_image(publication, replacement, workspace, request)
        corrupt = tmp_path / "corrupt"
        shutil.copytree(publication, corrupt)
        with (corrupt / launcher.ARCHIVE).open("ab") as handle:
            handle.write(b"corrupted")
        # Reuse is safe by immutable ID; archive bytes matter only on a cache miss.
        assert launcher.load_image(corrupt, request) == original
        launcher.docker("image", "rm", published_tag)
        with pytest.raises(launcher.LaunchError, match="checksum mismatch"):
            launcher.load_image(corrupt, request)
        (corrupt / launcher.FILES[1]).unlink()
        with pytest.raises(launcher.LaunchError, match="Incomplete image publication"):
            launcher.load_image(corrupt, request)
    finally:
        for reference in [tag, published_tag, *created]:
            if reference:
                subprocess.run(["docker", "image", "rm", reference], capture_output=True)


@pytest.mark.skipif(
    os.environ.get("FINN_TEST_DOCKER_ENVIRONMENT") != "1", reason="opt-in Docker environment"
)
def test_routed_hls_tcl_uses_host_visible_library(tmp_path, monkeypatch):
    """A farm-routed HLS job runs outside Docker: the Tcl must name an hlslib
    copy in host-visible state, fetched without network access."""
    for name in list(os.environ):
        if name.startswith("FINN_RESOURCES_"):
            monkeypatch.delenv(name)
    wrappers = tmp_path / "farm-tools"
    wrappers.mkdir()
    (wrappers / "make").write_text("#!/bin/sh\nexit 99\n")
    (wrappers / "make").chmod(0o755)
    home = tmp_path / "finn-home"
    build = tmp_path / "build"
    monkeypatch.setenv("FINN_HOME", str(home))
    monkeypatch.setenv("FINN_HOST_BUILD_DIR", str(build))
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
        [str(REPO / "docker/run"), "--fpga", "--runtime", "xrt", "--no-build", "--"]
        + ["python", "-c", code],
        cwd=REPO,
        check=True,
    )
    tcl = (build / "hls-probe/hls_syn_probe.tcl").read_text()
    line = next(line for line in tcl.splitlines() if line.startswith("set config_bnnlibdir "))
    library = Path(line.split('"')[1])
    assert library.is_relative_to(home / "resources")
    # This process is outside Docker, as a routed farm invocation would be.
    assert (library / "bnn-library.h").is_file()
