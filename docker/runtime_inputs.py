#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Stage site runtime inputs and reuse the Jenkins public XRT download cache."""
import argparse
import fcntl
import hashlib
import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def checksum(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def copy(source, destination):
    """Atomically stage an archive without changing an identical existing copy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and checksum(source) == checksum(destination):
        return
    with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
        temporary = Path(directory) / destination.name
        shutil.copyfile(source, temporary)
        temporary.replace(destination)


def prepare(runtimes):
    """Retain the explicit legacy slashkit input and its runtime selection."""
    selected = set(filter(None, runtimes.split(",")))
    package = os.environ.get("SLASHKIT_DEB_PACKAGE")
    if package:
        source = Path(package).expanduser().resolve()
        if not source.is_file():
            raise ValueError("SLASHKIT_DEB_PACKAGE is not a file: " + str(source))
        copy(source, ROOT / "docker/packages/slashkit.deb")
        selected.add("slashkit")
    return ",".join(sorted(selected))


def supplied_packages(runtimes, root=None):
    """List the archives actually installed by the selected supply manifests."""
    root = Path(root) if root is not None else ROOT
    packages = set()
    for name in sorted(set(filter(None, runtimes.split(",")))):
        manifest = root / "docker/runtimes" / (name + ".env")
        output = subprocess.check_output(
            [
                "sh",
                "-e",
                "-c",
                'unset SOURCE FILES; . "$1"; printf "%s\\0%s" "${SOURCE:?}" "${FILES:-}"',
                "runtime-manifest",
                str(manifest),
            ]
        ).decode()
        source, files = output.split("\0", 1)
        if source == "fetch":
            continue
        if source != "supply" or not files.split():
            raise ValueError("Invalid runtime SOURCE/FILES in " + str(manifest))
        for filename in files.split():
            relative = Path(filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Runtime package must be relative to docker/packages: " + filename)
            path = Path("docker/packages") / relative
            if not (root / path).is_file():
                raise ValueError(f"Runtime {name!r} requires readable package {root / path}")
            packages.add(path.as_posix())
    return sorted(packages)


def cache_xrt(runtimes):
    """Populate/stage the pinned public package only when a local build needs it."""
    cache = os.environ.get("FINN_XRT_PATH")
    if not cache or "xrt" not in runtimes.split(","):
        return
    # Manifests are shell declarations; evaluate their existing syntax once.
    values = (
        subprocess.check_output(
            [
                "sh",
                "-c",
                '. "$1"; printf "%s\\0" "$SOURCE" "$DEB" "$URL" "$FALLBACK_URL" "$SHA256"',
                "xrt-manifest",
                str(ROOT / "docker/runtimes/xrt.env"),
            ]
        )
        .decode()
        .split("\0")
    )
    source, name, url, fallback, expected = values[:5]
    if source != "fetch":
        return
    if not expected:
        raise ValueError("FINN_XRT_PATH reuse requires the XRT manifest's SHA256")
    cache = Path(cache).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    package = cache / (name + ".deb")
    with (cache / ".finn-xrt.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not package.is_file():
            with tempfile.TemporaryDirectory(dir=cache) as directory:
                temporary = Path(directory) / package.name
                for address in filter(None, (url, fallback)):
                    try:
                        with urllib.request.urlopen(address, timeout=120) as response:
                            with temporary.open("wb") as output:
                                shutil.copyfileobj(response, output)
                        if checksum(temporary) != expected:
                            raise ValueError("XRT download checksum mismatch: " + address)
                        temporary.replace(package)
                        break
                    except (OSError, ValueError):
                        if address == (fallback or url):
                            raise
        if checksum(package) != expected:
            raise ValueError("Cached XRT checksum mismatch: " + str(package))
        copy(package, ROOT / "docker/runtime-cache" / package.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtimes")
    parser.add_argument("--cache-xrt", action="store_true")
    parser.add_argument("--packages", action="store_true")
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    try:
        if args.packages:
            for package in supplied_packages(args.runtimes, args.root):
                print(package)
        elif args.cache_xrt:
            cache_xrt(args.runtimes)
        else:
            print(prepare(args.runtimes))
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(2, "Runtime preparation failed: " + str(error) + "\n")


if __name__ == "__main__":
    main()
