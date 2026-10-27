#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Prepare image provenance and transport immutable Docker images over NFS."""
import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FILES = ("finn-docker-tag.txt", "finn-image-digest.txt", "finn-image-provenance.json")
ARCHIVE = "finn-docker-image.tar.gz"
IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")


def docker(*args):
    """Run the configured daemon CLI, keeping stdout available for machine data."""
    return subprocess.check_output(["docker", *args], text=True).strip()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata(reference):
    """Describe actual bytes and their environment and mounted-source inputs."""
    inspected = json.loads(docker("image", "inspect", reference))[0]
    image_id = inspected["Id"]
    if not IMAGE_ID.fullmatch(image_id):
        raise ValueError("Invalid Docker image ID: " + image_id)
    labels = inspected.get("Config", {}).get("Labels") or {}
    revision = labels.get("dev.finn.image-revision")
    if revision != os.environ["FINN_IMAGE_REVISION"]:
        expected = os.environ["FINN_IMAGE_REVISION"]
        raise ValueError(f"Image environment {revision!r} differs from requested {expected!r}")
    mode = labels.get("dev.finn.source-mode", "checkout")
    expected_mode = "embedded" if os.environ.get("FINN_IMAGE_VARIANT") == "release" else "checkout"
    if mode != expected_mode:
        raise ValueError(f"Image source mode {mode!r}; requested {expected_mode!r}")
    runtimes = sorted(filter(None, os.environ.get("FINN_RUNTIMES", "").split(",")))
    if labels.get("dev.finn.runtimes", "").split(",") != (runtimes or [""]):
        raise ValueError("Image runtimes differ from the requested runtime set")
    resources = tomllib.loads((REPO / "src/finn/resources.toml").read_text())["resources"]
    return {
        "schema": 1,
        "image_id": image_id,
        "environment_revision": revision,
        "lock_sha256": sha256(REPO / "uv.lock"),
        "source_revision": os.environ["FINN_SOURCE_REVISION"],
        "source_dirty": {"0": False, "1": True}.get(os.environ["FINN_SOURCE_DIRTY"]),
        "source_mode": mode,
        "embedded_source_revision": labels.get("org.opencontainers.image.revision"),
        "runtimes": runtimes,
        "resources": resources,
        "ci": {
            key: os.environ[key]
            for key in ("JOB_NAME", "BUILD_NUMBER", "BUILD_URL")
            if key in os.environ
        },
    }


def write_metadata(dest, reference, info):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / FILES[0]).write_text(reference + "\n")
    (dest / FILES[1]).write_text(info["image_id"] + "\n")
    (dest / FILES[2]).write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")


def read_metadata(directory, archive=False):
    """Fail closed on missing, stale, partial, or old two-file publications."""
    directory = Path(directory)
    try:
        info = json.loads((directory / FILES[2]).read_text())
        image_id = (directory / FILES[1]).read_text().strip()
        reference = (directory / FILES[0]).read_text().strip()
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Incomplete image publication at {directory}: {error}. "
            "Republish using docker/image.py; legacy two-file archives require a rebuild."
        ) from error
    if (
        info.get("schema") != 1
        or not IMAGE_ID.fullmatch(image_id)
        or info.get("image_id") != image_id
    ):
        raise ValueError("Malformed or mismatched image identity")
    if not reference or "\n" in reference:
        raise ValueError("Malformed published image reference")
    for key, expected in (
        ("environment_revision", os.environ["FINN_IMAGE_REVISION"]),
        ("lock_sha256", sha256(REPO / "uv.lock")),
        ("runtimes", sorted(filter(None, os.environ.get("FINN_RUNTIMES", "").split(",")))),
    ):
        if info.get(key) != expected:
            raise ValueError(f"Published {key}={info.get(key)!r}; requested {expected!r}")
    if archive and sha256(directory / ARCHIVE) != info.get("archive_sha256"):
        raise ValueError("Published image archive checksum mismatch")
    return image_id, reference, info


def publish(directory, reference, workspace):
    """Publish one complete build directory by rename. Retries cannot replace it."""
    directory = Path(directory)
    info = metadata(reference)
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.is_dir() and any(directory.iterdir()):
        previous_id, previous_ref, previous = read_metadata(directory, archive=True)
        if (
            previous_id != info["image_id"]
            or previous["source_revision"] != info["source_revision"]
        ):
            raise ValueError(f"Build directory already contains another image/source: {directory}")
        write_metadata(workspace, previous_ref, previous)
        return
    with tempfile.TemporaryDirectory(dir=directory.parent, prefix=".finn-image-") as work:
        prepared = Path(work) / "publication"
        prepared.mkdir()
        # A private tag preserves normal docker-save transport semantics without
        # racing a rebuild that changes the human-facing tag during docker save.
        transport = "finn-transport:" + uuid.uuid4().hex
        docker("tag", info["image_id"], transport)
        try:
            with (prepared / ARCHIVE).open("wb") as raw:
                with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=1, mtime=0) as compressed:
                    process = subprocess.Popen(
                        ["docker", "save", transport], stdout=subprocess.PIPE
                    )
                    try:
                        shutil.copyfileobj(process.stdout, compressed)
                    finally:
                        process.stdout.close()
                    if process.wait():
                        raise RuntimeError("docker save failed")
        finally:
            docker("image", "rm", transport)
        info["requested_reference"] = reference
        info["transport_reference"] = transport
        info["archive_sha256"] = sha256(prepared / ARCHIVE)
        write_metadata(prepared, transport, info)
        for file in prepared.iterdir():
            with file.open("rb") as handle:
                os.fsync(handle.fileno())
        write_metadata(workspace, transport, info)
        try:
            prepared.rename(directory)
        except OSError:
            # A concurrent publisher won. Workspace provenance follows the
            # actual published archive, including its transport tag/checksum.
            previous_id, previous_ref, previous = read_metadata(directory, archive=True)
            if (
                previous_id != info["image_id"]
                or previous["source_revision"] != info["source_revision"]
            ):
                raise ValueError(
                    f"Build directory already contains another image/source: {directory}"
                )
            write_metadata(workspace, previous_ref, previous)


def load(directory):
    """Always load authoritative bytes, then run by verified ID rather than tag."""
    image_id, _, _ = read_metadata(directory, archive=True)
    result = subprocess.run(
        ["docker", "load", "--input", str(Path(directory) / ARCHIVE)], stdout=sys.stderr
    )
    if result.returncode:
        raise RuntimeError("docker load failed")
    if docker("image", "inspect", "--format", "{{.Id}}", image_id) != image_id:
        raise ValueError("Loaded image does not match the published identity")
    # Inspect actual image labels too; a forged sidecar cannot relabel an image.
    metadata(image_id)
    return image_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("inspect")
    p.add_argument("reference")
    p = sub.add_parser("metadata")
    p.add_argument("directory")
    p.add_argument("reference")
    p = sub.add_parser("publish")
    p.add_argument("directory")
    p.add_argument("reference")
    p.add_argument("--workspace", required=True)
    p = sub.add_parser("load")
    p.add_argument("directory")
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            print(metadata(args.reference)["image_id"])
        elif args.command == "load":
            print(load(args.directory))
        elif args.command == "metadata":
            write_metadata(args.directory, args.reference, metadata(args.reference))
        else:
            publish(args.directory, args.reference, args.workspace)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"finn image: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
