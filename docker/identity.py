#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Prepare private NSS files for caller-owned containers without editing /etc."""
import argparse
import os
import shlex
import tempfile
from pathlib import Path


def prepare(directory, uid, gid, username, home, container=False):
    """Return NSS variables for an atomic passwd/group pair owned by the caller."""
    for value in (username, home):
        if any(char in value for char in (":", "\n", "\r")):
            raise ValueError("Container identity fields cannot contain colons or newlines")
    root = Path(directory).expanduser().absolute()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    passwd = Path("/etc/passwd").read_text() if container else "root:x:0:0:root:/root:/bin/bash\n"
    groups = Path("/etc/group").read_text() if container else "root:x:0:\n"

    def replace_id(text, identity, row):
        rows = [
            line
            for line in text.splitlines()
            if len(line.split(":")) < 3 or line.split(":")[2] != str(identity)
        ]
        return "\n".join([*rows, row]) + "\n"

    group_name = next(
        (
            line.split(":")[0]
            for line in groups.splitlines()
            if len(line.split(":")) > 2 and line.split(":")[2] == str(gid)
        ),
        f"finn-{gid}",
    )
    values = {
        "passwd": replace_id(passwd, uid, f"{username}:x:{uid}:{gid}:FINN:{home}:/bin/bash"),
        "group": replace_id(groups, gid, f"{group_name}:x:{gid}:{username}"),
    }
    for name, text in values.items():
        with tempfile.NamedTemporaryFile(mode="w", dir=root, delete=False) as handle:
            handle.write(text)
            temporary = Path(handle.name)
        temporary.replace(root / name)
    return {"NSS_WRAPPER_PASSWD": str(root / "passwd"), "NSS_WRAPPER_GROUP": str(root / "group")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--home", required=True)
    args = parser.parse_args()
    for key, value in prepare(
        args.directory, os.getuid(), os.getgid(), args.username, args.home, container=True
    ).items():
        print("export " + key + "=" + shlex.quote(value))


if __name__ == "__main__":
    main()
