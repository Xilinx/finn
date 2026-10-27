# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Opt-in real Docker transport check using small, explicitly labelled fixtures."""
import pytest

import json
import os
import shutil
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docker"))
import image as images  # noqa: E402

pytestmark = pytest.mark.util


@pytest.mark.skipif(
    os.environ.get("FINN_TEST_DOCKER_TRANSPORT") != "1", reason="opt-in Docker transport"
)
def test_transport_uses_published_identity_and_rejects_corruption(tmp_path, monkeypatch):
    """Exercise real save/load, a stale tag, concurrent loads and publication retries."""
    key = uuid.uuid4().hex
    tag = "finn-transport-test:" + key
    environment = "fixture-" + key
    monkeypatch.setenv("FINN_IMAGE_REVISION", environment)
    monkeypatch.setenv("FINN_RUNTIMES", "")
    monkeypatch.setenv("FINN_SOURCE_REVISION", "a" * 40)
    monkeypatch.setenv("FINN_SOURCE_DIRTY", "0")
    monkeypatch.delenv("FINN_IMAGE_VARIANT", raising=False)
    context = tmp_path / "context"
    context.mkdir()
    (context / "Dockerfile").write_text(
        'FROM scratch\nLABEL dev.finn.image-revision="' + environment + '"\n'
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
        identity = images.docker("image", "inspect", "--format", "{{.Id}}", tag)
        created.append(identity)
        return identity

    try:
        original = build("original artifact\n")
        images.publish(publication, tag, workspace)
        published_tag = (publication / images.FILES[0]).read_text().strip()
        replacement = build("stale local artifact\n")
        assert original != replacement
        # Remove the published identity from image metadata, then create a
        # same-tag collision. Loads must recover the archive's original bytes.
        subprocess.run(
            ["docker", "image", "rm", original],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        images.docker("tag", replacement, published_tag)
        with ThreadPoolExecutor(max_workers=2) as executor:
            assert (
                list(executor.map(lambda _: images.load(publication), range(2))) == [original] * 2
            )
        assert images.docker("image", "inspect", "--format", "{{.Id}}", published_tag) == original
        assert images.docker("image", "inspect", "--format", "{{.Id}}", tag) == replacement
        images.publish(publication, original, workspace)
        assert json.loads((workspace / images.FILES[2]).read_text()) == json.loads(
            (publication / images.FILES[2]).read_text()
        )
        with pytest.raises(ValueError, match="another image/source"):
            images.publish(publication, replacement, workspace)
        corrupt = tmp_path / "corrupt"
        shutil.copytree(publication, corrupt)
        with (corrupt / images.ARCHIVE).open("ab") as handle:
            handle.write(b"corrupted")
        # Reuse is safe by immutable ID; archive bytes matter only on a cache miss.
        assert images.load(corrupt) == original
        images.docker("image", "rm", published_tag)
        with pytest.raises(ValueError, match="checksum mismatch"):
            images.load(corrupt)
        (corrupt / images.FILES[1]).unlink()
        with pytest.raises(ValueError, match="Incomplete image publication"):
            images.load(corrupt)
    finally:
        for reference in [tag, published_tag, *created]:
            if reference:
                subprocess.run(
                    ["docker", "image", "rm", reference],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
