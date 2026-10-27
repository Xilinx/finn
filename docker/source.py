#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Snapshot tracked working-tree sources, excluding supplied runtime archives."""
import argparse
import os
import subprocess
import tarfile
from pathlib import Path


def snapshot(source, output):
    """Include tracked edits/deletions, never ignored downloads or notebook output."""
    source = Path(source).resolve()
    names = subprocess.check_output(["git", "-C", str(source), "ls-files", "--cached", "-z"])
    with tarfile.open(output, "w") as archive:
        for name in sorted(set(os.fsdecode(name) for name in names.split(b"\0") if name)):
            path = source / name
            if name.startswith("docker/packages/") and path.suffix in (".deb", ".rpm"):
                continue
            if not path.exists() and not path.is_symlink():
                continue
            if path.is_dir() and not path.is_symlink():
                raise ValueError("Source snapshot needs an expanded tracked file, not: " + name)
            info = archive.gettarinfo(str(path), arcname=name)
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            info.mode = 0o755 if info.mode & 0o100 else 0o644
            if info.isfile():
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
            else:
                archive.addfile(info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()
    snapshot(args.source, args.output)


if __name__ == "__main__":
    main()
