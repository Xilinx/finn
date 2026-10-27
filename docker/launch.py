#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Execute a resolved host request with Docker, preserving argument boundaries."""
import argparse
import config
import json
import os
import shlex
import sys


def command(args):
    """Return Docker's argv; Compose consumers use the same resolver separately."""
    resolved = os.environ.get("FINN_HOST_REQUEST")
    data = json.loads(resolved) if resolved else config.resolve_host(args.tier, "mirror")
    build = config.resolve_build_dir(create=True, fatal=True)
    argv = ["docker", "run", "--rm", "--pull=never", "--user", f"{os.getuid()}:{os.getgid()}"]
    environment = dict(
        config.container_environment(data), FINN_BUILD_DIR=build, FINN_HOME=build + "/.finn"
    )
    environment["FINN_IMAGE_ID"] = args.image
    mounts = [
        (data["workspace"]["source"], data["workspace"]["target"], "rw"),
        (build, build, "rw"),
    ]
    mounts += [(m["source"], m["target"], m["mode"]) for m in data["mounts"]]
    cache = os.environ.get("FINN_DOCKER_CACHE_DIR")
    if cache:
        cache = config.hostpath(cache)
        for directory in ("torch", "huggingface", "uv"):
            os.makedirs(os.path.join(cache, directory), exist_ok=True)
        mounts.append((cache, "/finn_cache", "rw"))
        environment.update(
            TORCH_HOME="/finn_cache/torch",
            HF_HOME="/finn_cache/huggingface",
            UV_CACHE_DIR="/finn_cache/uv",
        )
    ssh = os.environ.get("FINN_SSH_KEY_DIR")
    if ssh:
        ssh = config.hostpath(ssh)
        if not os.path.isdir(ssh):
            raise ValueError("FINN_SSH_KEY_DIR is not a directory: " + ssh)
        mounts.append((ssh, "/run/finn-ssh", "ro"))
        environment["FINN_CONTAINER_SSH_DIR"] = "/run/finn-ssh"
    for variable in ("VERIFICATION_IO", "FINN_EXAMPLES_ROOT", "FINN_PLATFORM_PATH"):
        value = os.environ.get(variable)
        if value:
            value = config.hostpath(value)
            if not os.path.isdir(value):
                raise ValueError(f"{variable} is not a directory: {value}")
            mounts.append((value, value, "rw"))
            environment[variable] = value
    environment["VERIFICATION_EN"] = os.environ.get("VERIFICATION_EN", "0")
    seen = set()
    for source, target, mode in mounts:
        if target not in seen:
            argv += ["--volume", f"{source}:{target}:{mode}"]
            seen.add(target)
    for key, value in environment.items():
        argv += ["--env", key + "=" + value]
    for spec in args.volume:
        argv += ["--volume", spec]
    for value in args.env:
        argv += ["--env", value]
    argv += ["--workdir", args.workdir or data["workspace"]["target"]]
    if args.name:
        argv += ["--name", args.name]
    if os.environ.get("FINN_DOCKER_RUN_AS_ROOT") == "1":
        argv += ["--user", "0:0"]
    # Site Docker flags are tokenized, never evaluated as shell code.
    argv += shlex.split(os.environ.get("FINN_DOCKER_EXTRA", ""))
    cmd = args.command
    if cmd[:1] == ["--"]:
        cmd = cmd[1:]
    if args.notebook:
        if cmd:
            raise ValueError("--notebook does not accept a command")
        port = os.environ.get("JUPYTER_PORT", "8888")
        netron = os.environ.get("NETRON_PORT", "8081")
        argv += ["--publish", f"{port}:{port}", "--publish", f"{netron}:{netron}"]
        environment = {"JUPYTER_PORT": port, "NETRON_PORT": netron}
        for key, value in environment.items():
            argv += ["--env", key + "=" + value]
        cmd = ["jupyter", "notebook", "--no-browser", "--ip=0.0.0.0", "--port", port]
        password = os.environ.get("JUPYTER_PASSWD_HASH")
        if password:
            cmd += ["--PasswordIdentityProvider.hashed_password=" + password]
        if os.environ.get("FINN_DOCKER_RUN_AS_ROOT") == "1" or os.getuid() == 0:
            cmd += ["--allow-root"]
        cmd += ["notebooks"]
    elif not cmd:
        cmd = ["bash"]
    if sys.stdin.isatty():
        argv += ["-i"]
        if sys.stdout.isatty():
            argv += ["-t"]
    return argv + [args.image] + cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("dev", "build"), required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--volume", action="append", default=[])
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--workdir")
    parser.add_argument("--name")
    parser.add_argument("--notebook", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        argv = command(args)
    except ValueError as error:
        parser.error(str(error))
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    main()
