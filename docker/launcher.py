#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""FINN's host launcher: build, run and transport the FINN Docker image.

One standard-library program (Python 3.10 or later) owns everything the host
decides before a container starts:

- the host file ``~/.config/finn/xilinx.env`` and AMD's install layouts;
- workspace, build directory, FINN_HOME, toolchain, licence and resource mounts;
- the image identity (a hash of ``docker/image-inputs.txt``), Bake builds and tags;
- publication and verified loading of images over shared storage (Jenkins).

``docker/build`` and ``docker/run`` are thin wrappers around the ``build`` and
``run`` commands. ``inspect`` prints the resolved host request (``--format sh``
is what ``scripts/activate.sh`` evaluates) and ``compose`` renders a Compose
override from the same resolution. Diagnostics go to stderr, because stdout is
machine-readable data.
"""
import argparse
import fcntl
import glob
import gzip
import hashlib
import json
import os
import pwd
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path

DOCKER_DIR = Path(__file__).resolve().parent
REPO = DOCKER_DIR.parent
sys.path.insert(0, str(DOCKER_DIR))
import identity  # noqa: E402  (also runs inside the image; standard library only)

DOCS = "docker/README.md"


class LaunchError(Exception):
    """A host input or Docker operation the launcher cannot proceed with."""

    def __init__(self, message, code=1):
        super().__init__(message)
        self.code = code


def warn(message):
    print("finn: " + message, file=sys.stderr)


def hostpath(value):
    """Expand ``~`` and make a host path absolute before it becomes a mount."""
    if not value:
        return value
    return os.path.abspath(os.path.expanduser(value))


def shquote(value):
    """Quote a value for both shell ``eval`` and a Compose env file."""
    return "'" + str(value).replace("'", "'\\''") + "'"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# The host file and AMD's install layouts.
# --------------------------------------------------------------------------

# Machine settings in ~/.config/finn/xilinx.env, e.g.
#
#     FINN_XILINX_PATH=/opt/Xilinx
#     FINN_XILINX_VERSION=2025.2
#     FINN_LICENSE_HOST=10.0.0.5
#     FINN_LICENSE_PORT=2100
#
# NAME=value and # comment lines, without quotes, expansion or trailing
# comments. Environment variables override these defaults. FINN_XILINX_ENV
# names another file; set it empty to use none.
HOST_KEYS = (
    "FINN_XILINX_PATH",
    "FINN_XILINX_VERSION",
    "FINN_LICENSE_HOST",
    "FINN_LICENSE_PORT",
    "FINN_LICENSE_VENDOR_PORT",
    "PLATFORM_REPO_PATHS",
    "XILINXD_LICENSE_FILE",
    "LM_LICENSE_FILE",
    "FINN_TOOL_DIR_OVERRIDE",
)
# Paths must be absolute so native and container callers resolve the same input.
HOST_PATH_KEYS = ("FINN_XILINX_PATH", "PLATFORM_REPO_PATHS")
# FlexLM reads either; one set by the caller is never replaced.
LICENSE_VARIABLES = ("XILINXD_LICENSE_FILE", "LM_LICENSE_FILE")


def host_file_path(environ=None):
    """The host file to read, or None."""
    environ = os.environ if environ is None else environ
    if "FINN_XILINX_ENV" in environ:
        return environ["FINN_XILINX_ENV"] or None
    config = environ.get("XDG_CONFIG_HOME") or os.path.join(
        environ.get("HOME") or os.path.expanduser("~"), ".config"
    )
    return os.path.join(config, "finn", "xilinx.env")


def read_host_file(path):
    """Parse a host file; raise LaunchError on malformed input."""
    values = {}
    with open(path, "r") as handle:
        for number, line in enumerate(handle, 1):
            where = "%s:%d" % (path, number)
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            name, sep, value = text.partition("=")
            name = name.strip()
            if not sep:
                raise LaunchError("%s: expected NAME=value" % where, 3)
            if name not in HOST_KEYS:
                raise LaunchError(
                    "%s: %r is not a setting of FINN host configuration (%s)"
                    % (where, name, ", ".join(HOST_KEYS)),
                    3,
                )
            if value != value.strip() or value[:1] in ("'", '"') or " #" in value:
                raise LaunchError(
                    "%s: write %s=value with no spaces, quotes or trailing comment" % (where, name),
                    3,
                )
            if name in HOST_PATH_KEYS and value and not value.startswith("/"):
                raise LaunchError("%s: %s must be an absolute path" % (where, name), 3)
            values[name] = value
    return values


def host_settings(environ=None):
    """The host file's values, each overridden by an environment variable of its name."""
    environ = os.environ if environ is None else environ
    path = host_file_path(environ)
    values = {}
    if path and os.path.exists(path):
        values = read_host_file(path)
    elif path and "FINN_XILINX_ENV" in environ:
        raise LaunchError("FINN_XILINX_ENV=%s does not exist" % path, 3)
    for key in HOST_KEYS:
        if key in environ:
            values[key] = environ[key]
    return {key: value for key, value in values.items() if value}


def license_for(environment, values):
    """The ``XILINXD_LICENSE_FILE`` (``port@host``) of the host file's ``values``, or
    None where ``environment`` already names a licence or the values name no server."""
    if any(environment.get(name) for name in LICENSE_VARIABLES):
        return None
    host, port = values.get("FINN_LICENSE_HOST"), values.get("FINN_LICENSE_PORT")
    return "%s@%s" % (port, host) if host and port else None


def apply_host_settings():
    """Read the host file into this process's environment, under the environment.

    Filling os.environ once keeps a variable set by the caller winning over the
    file without every reader knowing about the file. A licence given as host and
    port becomes the FlexLM form when no licence variable is set.
    """
    values = host_settings()
    for key, value in values.items():
        os.environ.setdefault(key, value)
    licence = license_for(os.environ, values)
    if licence:
        os.environ["XILINXD_LICENSE_FILE"] = licence


def xilinx_layout(root, version):
    """Locate Vivado / Vitis / HLS within a Xilinx install.

    AMD reorganised the install tree after 2024.2:

        <= 2024.2   $ROOT/Vivado/2022.2   $ROOT/Vitis_HLS/2022.2
        >  2024.2   $ROOT/2025.1/Vivado   $ROOT/2025.1/Vitis

    Both candidates are probed; the version string only orders them, so a site
    with a symlinked or non-standard tree is detected rather than assumed.
    """
    if not root or not version:
        return {}
    match = re.match(r"^(20\d\d)\.([12])$", version)
    if not match:
        warn(
            "FINN_XILINX_VERSION %r is not YYYY.1 or YYYY.2; probing both layouts anyway" % version
        )
        new_first = False
    else:
        new_first = (int(match.group(1)), int(match.group(2))) > (2024, 2)
    new = {
        "XILINX_VIVADO": os.path.join(root, version, "Vivado"),
        "XILINX_VITIS": os.path.join(root, version, "Vitis"),
        "XILINX_HLS": os.path.join(root, version, "Vitis"),
    }
    old = {
        "XILINX_VIVADO": os.path.join(root, "Vivado", version),
        "XILINX_VITIS": os.path.join(root, "Vitis", version),
        "XILINX_HLS": os.path.join(root, "Vitis_HLS", version),
    }
    found = {}
    for candidate in (new, old) if new_first else (old, new):
        for key, path in candidate.items():
            if key not in found and os.path.isdir(path):
                found[key] = path
    return found


# --------------------------------------------------------------------------
# Licences. Network requirements are declarations; Docker does not enforce them.
# --------------------------------------------------------------------------


def vendor_daemon_port(files):
    """The pinned FLEXlm vendor-daemon port, if the site pinned one.

    lmgrd on the advertised port hands back a second port where the vendor
    daemon does the checkout. That port is ephemeral unless a licence admin pins
    it (``DAEMON xilinxd /opt/xilinx/xilinxd port=2101``). Read it from a
    licence file where one is visible, else from FINN_LICENSE_VENDOR_PORT.
    None means unknown: the requirement covers the whole host.
    """
    explicit = os.environ.get("FINN_LICENSE_VENDOR_PORT", "").strip()
    if explicit:
        if not explicit.isdigit():
            warn("FINN_LICENSE_VENDOR_PORT=%r is not a port number; ignoring" % explicit)
        else:
            return explicit
    for path in files or ():
        try:
            with open(hostpath(path), "r", errors="replace") as handle:
                text = handle.read()
        except OSError:
            continue
        match = re.search(
            r"^\s*DAEMON\s+\S+.*?\bport\s*=\s*(\d+)", text, re.IGNORECASE | re.MULTILINE
        )
        if match:
            return match.group(1)
    return None


def classify_license(value):
    """Split a FLEXlm licence list into servers (PORT@HOST) and files.

    A server requirement covers the whole host unless the vendor port is known:
    a port-scoped rule against an unpinned daemon lets `lmutil lmstat` succeed
    while every real checkout fails with "A valid license was not found".
    """
    servers, files = [], []
    for entry in (value or "").split(":"):
        entry = entry.strip()
        if not entry:
            continue
        if "@" in entry:
            port, _, host = entry.partition("@")
            servers.append({"host": host, "advertised_port": port})
        else:
            files.append(entry)
    return servers, files


# --------------------------------------------------------------------------
# Host resolution: what a container gets, derived from the host exactly once.
# --------------------------------------------------------------------------

# A tier is a host access profile, independent of the image. `build` gets the
# Xilinx toolchain, licences and declared network requirements; `dev` gets none.
# `auto` is `build` where tools are configured, else `dev`.
TIERS = ("dev", "build")
FIXED_WORKSPACE = "/workspace/finn"

# Inputs passed through to the container when set, whatever the tier.
PASSTHROUGH = (
    "NUM_DEFAULT_WORKERS",
    "LOCALHOST_URL",
    "FINN_XELAB_MT",
    "FINN_VCK190_GOLDEN",
    "FINN_LSF_NFS_STAGING",
    "FINN_LSF_JOB_PREFIX",
    "FINN_CI_BUILD_TAG",
    "FINN_CI_JOB_KEY",
    "RTLSIM_TRACE_DEPTH",
    "LIVENESS_THRESHOLD",
)
# Host directories the container uses at the same path, read-write.
SITE_DIRECTORIES = ("FINN_BOARD_FILES_PATH", "FINN_GOLDEN_DIR", "FINN_LSF_NFS_STAGING")
# Persistent FINN state: FINN_HOME and the caches that may live elsewhere. Each
# is created on the host if needed and mounted read-write at the same path.
STATE_DIRECTORIES = ("FINN_HOME", "FINN_RESOURCES_DIR", "FINN_XSI_PATH")
RESOURCES = "FINN_RESOURCES_"


def tools_configured():
    """Whether the host names an installed Xilinx toolchain."""
    root = hostpath(os.environ.get("FINN_XILINX_PATH"))
    return bool(root and os.path.isdir(root)) or any(
        os.path.isdir(hostpath(os.environ.get(alias)) or "")
        for alias in ("VIVADO_PATH", "VITIS_PATH", "HLS_PATH")
    )


def resolve_tier(tier):
    """`auto` is `build` where tools are configured, else `dev`. Explicit tiers stand."""
    if tier != "auto":
        return tier
    return "build" if tools_configured() else "dev"


def resolve_workspace(tier, policy="auto"):
    """Resolve the host and container workspace paths.

    Tool tiers mirror the checkout at its host path, since generated Vivado
    projects embed absolute paths. The dev tier may use a fixed path, useful for
    remote daemons and Dev Containers.
    """
    source = hostpath(os.environ.get("FINN_ROOT") or str(REPO))
    if policy == "auto":
        policy = "mirror" if tier == "build" else "fixed"
    elif policy == "fixed" and tier == "build":
        raise LaunchError("workspace policy 'fixed' is incompatible with tier 'build'", 2)
    target = source if policy == "mirror" else FIXED_WORKSPACE
    return {"policy": policy, "source": source, "target": target}


def resolve_build_dir():
    """The disposable host build directory, mirrored into containers."""
    return hostpath(
        os.environ.get("FINN_HOST_BUILD_DIR")
        or os.environ.get("FINN_BUILD_DIR")
        or "/tmp/finn_build_%d" % os.getuid()
    )


def resolve_finn_home():
    """FINN's persistent host directory, shared by native use and containers."""
    return hostpath(os.environ.get("FINN_HOME") or "~/.finn")


def add_toolchain(out):
    """Add toolchain paths and their read-only root mount; return the root."""
    root = hostpath(os.environ.get("FINN_XILINX_PATH"))
    version = os.environ.get("FINN_XILINX_VERSION")
    if not root or not os.path.isdir(root):
        warn("FINN_XILINX_PATH=%s is not an installed directory; not mounted" % root)
        root = None
    if root:
        out["mounts"].append(
            {"source": root, "target": root, "mode": "ro", "reason": "xilinx-toolchain"}
        )
    layout = xilinx_layout(root, version) if root else {}
    if root and not layout:
        warn(
            "no Vivado/Vitis/HLS found under %s for version %r; "
            "the toolchain mount will be present but empty of known tools" % (root, version)
        )
    out["env"].update(layout)
    for alias, name in (
        ("VIVADO_PATH", "XILINX_VIVADO"),
        ("VITIS_PATH", "XILINX_VITIS"),
        ("HLS_PATH", "XILINX_HLS"),
    ):
        value = os.environ.get(alias) or layout.get(name)
        if value:
            value = hostpath(value)
            if not os.path.isdir(value):
                warn(alias + " is not an installed directory; not mounted: " + value)
                continue
            out["env"][alias] = value
            out["env"][name] = value
            if not root or (not value.startswith(root + os.sep) and value != root):
                out["mounts"].append(
                    {"source": value, "target": value, "mode": "ro", "reason": "tool-override"}
                )
    override = hostpath(os.environ.get("FINN_TOOL_DIR_OVERRIDE"))
    if override:
        if not os.path.isdir(override):
            raise LaunchError("FINN_TOOL_DIR_OVERRIDE is not a directory: " + override, 3)
        out["env"]["FINN_TOOL_DIR_OVERRIDE"] = override
        out["mounts"].append(
            {"source": override, "target": override, "mode": "ro", "reason": "site-tools"}
        )
    return root


# Upstream's launcher default, where Vitis platforms are conventionally installed.
DEFAULT_PLATFORMS = "/opt/xilinx/platforms"


def add_platform_repo(out, toolchain_root):
    """Add the platform repository without overlapping the toolchain."""
    value = os.environ.get("PLATFORM_REPO_PATHS")
    if value is None and os.path.isdir(DEFAULT_PLATFORMS):
        value = DEFAULT_PLATFORMS
    resolved = []
    for path in filter(None, (value or "").split(os.pathsep)):
        platforms = hostpath(path)
        if not os.path.isdir(platforms):
            warn("platform repository is not a directory; not mounted: " + platforms)
            continue
        resolved.append(platforms)
        if not toolchain_root or (
            platforms != toolchain_root and not platforms.startswith(toolchain_root + os.sep)
        ):
            out["mounts"].append(
                {"source": platforms, "target": platforms, "mode": "ro", "reason": "platform-repo"}
            )
    if resolved:
        out["env"]["PLATFORM_REPO_PATHS"] = os.pathsep.join(resolved)


def add_licenses(out):
    """Add FLEXlm environment, file mounts and declarative network requirements."""
    for var in LICENSE_VARIABLES:
        value = os.environ.get(var)
        if not value:
            continue
        out["env"][var] = value
        servers, files = classify_license(value)
        vendor = vendor_daemon_port(files)
        for server in servers:
            grant = {
                "host": server["host"],
                "reason": "flexlm",
                "advertised_port": server["advertised_port"],
                "ports": [],
            }
            if vendor:
                grant["ports"] = sorted({server["advertised_port"], vendor})
                grant["vendor_port"] = vendor
            out["egress"].append(grant)
        # Mount exactly the named file or directory: a licence's parent, such as
        # /etc, would hide the image's own files.
        for path in files:
            source = hostpath(path)
            if not os.path.exists(source):
                warn("licence path %s not found on the host; not mounted" % path)
                continue
            if not any(m["source"] == source for m in out["mounts"]):
                out["mounts"].append(
                    {"source": source, "target": source, "mode": "ro", "reason": "license-file"}
                )


def add_optional_inputs(out):
    """Add explicitly requested, tier-independent runtime inputs."""
    for variable in PASSTHROUGH:
        if os.environ.get(variable):
            out["env"][variable] = os.environ[variable]
    for variable in SITE_DIRECTORIES:
        value = hostpath(os.environ.get(variable))
        if value:
            if not os.path.isdir(value):
                raise LaunchError(variable + " is not a directory: " + value, 3)
            out["env"][variable] = value
            out["mounts"].append(
                {"source": value, "target": value, "mode": "rw", "reason": "site-state"}
            )
    imagenet = hostpath(os.environ.get("IMAGENET_VAL_PATH"))
    if imagenet:
        out["env"]["IMAGENET_VAL_PATH"] = imagenet
        if os.path.isdir(imagenet):
            out["mounts"].append(
                {"source": imagenet, "target": imagenet, "mode": "ro", "reason": "imagenet"}
            )
        else:
            warn("IMAGENET_VAL_PATH=%s is not a directory; not mounted" % imagenet)


def add_resource_overrides(out):
    """Make finn.resources' host-side settings work inside the container.

    FINN_RESOURCES_<NAME> names a host directory, mounted writable at its own
    path. Source URLs and the offline switch pass through. The system cache is
    the image's own and is never replaced from the host; FINN_RESOURCES_DIR is
    persistent state (see STATE_DIRECTORIES).
    """
    for variable, value in sorted(os.environ.items()):
        if not value or not variable.startswith(RESOURCES):
            continue
        setting = variable[len(RESOURCES) :]
        if setting in ("SYSTEM_CACHE", "DIR"):
            continue
        if setting == "OFFLINE" or variable.endswith("_URL"):
            out["env"][variable] = value
            continue
        path = hostpath(value)
        if not os.path.isdir(path):
            warn("%s is not a directory; not mounted" % path)
        elif not any(m["source"] == path for m in out["mounts"]):
            out["mounts"].append(
                {"source": path, "target": path, "mode": "rw", "reason": "resource"}
            )
        out["env"][variable] = path


def resolve_host(tier, workspace_policy="auto", runtimes=None):
    """Everything a launcher needs, derived from the host exactly once."""
    apply_host_settings()
    tier = resolve_tier(tier)
    if tier not in TIERS:
        raise LaunchError("unknown tier %r; expected one of %s" % (tier, ", ".join(TIERS)), 2)
    runtimes = normalize_runtimes(
        os.environ.get("FINN_RUNTIMES", "") if runtimes is None else runtimes
    )
    workspace = resolve_workspace(tier, workspace_policy)
    state = {"FINN_HOME": resolve_finn_home()}
    for variable in STATE_DIRECTORIES[1:]:
        if os.environ.get(variable):
            state[variable] = hostpath(os.environ[variable])
    out = {
        "tier": tier,
        "backend": "docker",
        "runtimes": runtimes,
        "runtime_csv": ",".join(runtimes),
        "platform": "linux/amd64",
        "workspace": workspace,
        "build_dir": resolve_build_dir(),
        "state": state,
        "mounts": [],
        "egress": [],
        "egress_enforcement": "declared",
        "env": {"FINN_ROOT": workspace["target"], **state},
    }
    for variable in (
        "FINN_IMAGE_REVISION",
        "FINN_SOURCE_REVISION",
        "FINN_SOURCE_DESCRIBE",
        "FINN_SOURCE_DIRTY",
    ):
        if os.environ.get(variable):
            out["env"][variable] = os.environ[variable]
    add_optional_inputs(out)
    add_resource_overrides(out)
    if tier == "dev":
        out["dev_contract"] = {"toolchain": False, "license": False, "egress": False}
        return out
    root = add_toolchain(out)
    add_platform_repo(out, root)
    add_licenses(out)
    return out


def require_tools(data):
    """`--fpga` is a promise of tools: fail before any build when there are none."""
    if not any(data["env"].get(name) for name in ("XILINX_VIVADO", "XILINX_VITIS", "XILINX_HLS")):
        raise LaunchError(
            "--fpga needs the Xilinx tools, but none were found. Set FINN_XILINX_PATH and "
            "FINN_XILINX_VERSION (or write them to %s); see %s."
            % (host_file_path() or "~/.config/finn/xilinx.env", DOCS),
            3,
        )


def ensure_directory(path, variable):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as error:
        raise LaunchError("could not create %s=%s (%s)" % (variable, path, error), 3)
    return path


def sh_assignments(data):
    """The complete launcher/activation environment as shell assignments."""
    values = dict(data["env"])
    values.update(
        {
            "FINN_BUILD_DIR": data["build_dir"],
            "FINN_HOST_BUILD_DIR": data["build_dir"],
            "FINN_GID": str(os.getgid()),
            "FINN_RUNTIMES": data["runtime_csv"],
            "FINN_UID": str(os.getuid()),
            "FINN_WORKSPACE_SOURCE": data["workspace"]["source"],
            "FINN_WORKSPACE_TARGET": data["workspace"]["target"],
        }
    )
    if data["tier"] == "build" and os.environ.get("FINN_XILINX_PATH"):
        values["FINN_XILINX_PATH"] = hostpath(os.environ["FINN_XILINX_PATH"])
    if os.environ.get("FINN_IMAGE"):
        values["FINN_IMAGE"] = os.environ["FINN_IMAGE"]
    return "".join("%s=%s\n" % (key, shquote(value)) for key, value in sorted(values.items()))


def container_environment(data):
    """The container's creation environment, including for processes started by
    docker exec, which cannot inherit changes made later by the entrypoint."""
    env = dict(data["env"])
    uid, gid = (
        (0, 0) if os.environ.get("FINN_DOCKER_RUN_AS_ROOT") == "1" else (os.getuid(), os.getgid())
    )
    # The container's HOME is throwaway; persistent state lives in FINN_HOME.
    home = "/tmp/finn-home-%d" % uid
    try:
        username = pwd.getpwuid(uid).pw_name
    except KeyError:
        username = os.environ.get("USER") or "finn"
    identity_dir = Path(data["state"]["FINN_HOME"]) / "identity" / str(uid)
    files = {
        "NSS_WRAPPER_PASSWD": identity_dir / "passwd",
        "NSS_WRAPPER_GROUP": identity_dir / "group",
    }
    if all(path.is_file() for path in files.values()):
        # The entrypoint completes these from the image's own accounts; a
        # concurrent launch must not replace them with this minimal pair.
        env.update({key: str(path) for key, path in files.items()})
    else:
        env.update(identity.prepare(identity_dir, uid, gid, username, home))
    env["FINN_IDENTITY_DIR"] = str(identity_dir)
    env.update(HOME=home, USER=username, LOGNAME=username, FINN_BUILD_DIR=data["build_dir"])
    libraries = []
    for key, suffix in (
        ("XILINX_VIVADO", "lib/lnx64.o"),
        ("XILINX_VITIS", "lnx64/tools/fpo_v7_1"),
        ("XILINX_HLS", "lnx64/tools/fpo_v7_1"),
    ):
        if env.get(key):
            libraries.append(os.path.join(env[key], suffix))
    if "xrt" in data["runtimes"]:
        env["XILINX_XRT"] = "/opt/xilinx/xrt"
        libraries.append("/opt/xilinx/xrt/lib")
    if libraries:
        env["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(libraries))
    return env


def owned_mounts(data):
    """The workspace, build directory and state: created here, mounted read-write."""
    ensure_directory(data["build_dir"], "FINN_HOST_BUILD_DIR")
    mounts = [
        (data["workspace"]["source"], data["workspace"]["target"], "rw"),
        (data["build_dir"], data["build_dir"], "rw"),
    ]
    for variable, path in data["state"].items():
        mounts.append((ensure_directory(path, variable), path, "rw"))
    return mounts


# --------------------------------------------------------------------------
# docker run.
# --------------------------------------------------------------------------


def docker_run_argv(data, image, options, command):
    """Docker's argv for one container, preserving argument boundaries."""
    argv = ["docker", "run", "--rm", "--pull=never", "--user", "%d:%d" % (os.getuid(), os.getgid())]
    mounts = owned_mounts(data)
    environment = container_environment(data)
    environment["FINN_IMAGE_ID"] = image
    mounts += [(m["source"], m["target"], m["mode"]) for m in data["mounts"]]
    cache = os.environ.get("FINN_DOCKER_CACHE_DIR")
    if cache:
        cache = hostpath(cache)
        for directory in ("torch", "huggingface", "uv"):
            os.makedirs(os.path.join(cache, directory), exist_ok=True)
        mounts.append((cache, "/finn_cache", "rw"))
        environment.update(
            TORCH_HOME="/finn_cache/torch",
            HF_HOME="/finn_cache/huggingface",
            UV_CACHE_DIR="/finn_cache/uv",
        )
    # Upstream's default: the checkout's ssh_keys directory, when it exists.
    ssh = os.environ.get("FINN_SSH_KEY_DIR")
    if not ssh and (REPO / "ssh_keys").is_dir():
        ssh = str(REPO / "ssh_keys")
    if ssh:
        ssh = hostpath(ssh)
        if not os.path.isdir(ssh):
            raise LaunchError("FINN_SSH_KEY_DIR is not a directory: " + ssh, 2)
        mounts.append((ssh, "/run/finn-ssh", "ro"))
        environment["FINN_CONTAINER_SSH_DIR"] = "/run/finn-ssh"
    for variable in ("VERIFICATION_IO", "FINN_EXAMPLES_ROOT", "FINN_PLATFORM_PATH"):
        value = os.environ.get(variable)
        if value:
            value = hostpath(value)
            if not os.path.isdir(value):
                raise LaunchError("%s is not a directory: %s" % (variable, value), 2)
            mounts.append((value, value, "rw"))
            environment[variable] = value
    environment["VERIFICATION_EN"] = os.environ.get("VERIFICATION_EN", "0")
    seen = set()
    for source, target, mode in mounts:
        if target not in seen:
            argv += ["--volume", "%s:%s:%s" % (source, target, mode)]
            seen.add(target)
    for key, value in environment.items():
        argv += ["--env", key + "=" + value]
    for spec in options["volume"]:
        argv += ["--volume", spec]
    for value in options["env"]:
        argv += ["--env", value]
    argv += ["--workdir", options["workdir"] or data["workspace"]["target"]]
    if options["name"]:
        argv += ["--name", options["name"]]
    if os.environ.get("FINN_DOCKER_RUN_AS_ROOT") == "1":
        argv += ["--user", "0:0"]
    # Site Docker flags are tokenized, never evaluated as shell code.
    argv += shlex.split(os.environ.get("FINN_DOCKER_EXTRA", ""))
    if options["notebook"]:
        if command:
            raise LaunchError("--notebook does not accept a command", 2)
        port = os.environ.get("JUPYTER_PORT", "8888")
        netron = os.environ.get("NETRON_PORT", "8081")
        argv += ["--publish", "%s:%s" % (port, port), "--publish", "%s:%s" % (netron, netron)]
        argv += ["--env", "JUPYTER_PORT=" + port, "--env", "NETRON_PORT=" + netron]
        command = ["jupyter", "notebook", "--no-browser", "--ip=0.0.0.0", "--port", port]
        password = os.environ.get("JUPYTER_PASSWD_HASH")
        if password:
            command += ["--PasswordIdentityProvider.hashed_password=" + password]
        if os.environ.get("FINN_DOCKER_RUN_AS_ROOT") == "1" or os.getuid() == 0:
            command += ["--allow-root"]
        command += ["notebooks"]
    elif not command:
        command = ["bash"]
    if sys.stdin.isatty():
        argv += ["-i"]
        if sys.stdout.isatty():
            argv += ["-t"]
    return argv + [image] + list(command)


# --------------------------------------------------------------------------
# Compose.
# --------------------------------------------------------------------------


def compose_bind(source, target, mode="rw"):
    """Return one long-form Compose bind mount."""
    mount = {
        "type": "bind",
        "source": source,
        "target": target,
        "bind": {"create_host_path": False},
    }
    if mode == "ro":
        mount["read_only"] = True
    return mount


def compose_override(data, services):
    """Render host-specific runtime configuration for Compose."""
    image = os.environ.get("FINN_IMAGE")
    if data["runtime_csv"] and not image:
        raise LaunchError(
            "FINN_IMAGE must be set when FINN_RUNTIMES is non-empty; "
            "use docker/run or `docker/build --print-tag`",
            2,
        )
    volumes, seen = [], set()
    for source, target, mode in owned_mounts(data) + [
        (m["source"], m["target"], m.get("mode", "rw")) for m in data["mounts"]
    ]:
        if target not in seen:
            volumes.append(compose_bind(source, target, mode))
            seen.add(target)
    service = {
        "environment": dict(container_environment(data), FINN_RUNTIMES=data["runtime_csv"]),
        "user": "0:0"
        if os.environ.get("FINN_DOCKER_RUN_AS_ROOT") == "1"
        else "%d:%d" % (os.getuid(), os.getgid()),
        "volumes": volumes,
        "working_dir": data["workspace"]["target"],
    }
    if image:
        service["image"] = image
    if os.environ.get("FINN_CONTAINER_NAME"):
        service["container_name"] = os.environ["FINN_CONTAINER_NAME"]
    return {"services": {name: dict(service) for name in services}}


def inputs_override(data, services):
    """Only the host inputs (toolchain, licence, resources), for a caller that
    owns the workspace, build directory, state and user itself."""
    volumes = [compose_bind(m["source"], m["target"], m.get("mode", "rw")) for m in data["mounts"]]
    owned = {"FINN_ROOT", *data["state"]}
    environment = {k: v for k, v in data["env"].items() if k not in owned}
    service = {"environment": environment, "volumes": volumes}
    return {"services": {name: dict(service) for name in services}}


# --------------------------------------------------------------------------
# Runtime content: selection, supplied packages and the public XRT cache.
# --------------------------------------------------------------------------


def normalize_runtimes(runtimes):
    """The runtime set in its canonical, duplicate-free order."""
    if isinstance(runtimes, str):
        runtimes = runtimes.replace(",", " ").split()
    return sorted(set(name for name in runtimes if name))


def copy_atomically(source, destination):
    """Stage a file without changing an identical existing copy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256(source) == sha256(destination):
        return
    with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
        temporary = Path(directory) / destination.name
        shutil.copyfile(source, temporary)
        temporary.replace(destination)


def prepare_runtimes(runtimes, root=REPO):
    """Stage the site slashkit package (SLASHKIT_DEB_PACKAGE) and select its runtime."""
    selected = set(normalize_runtimes(runtimes))
    package = os.environ.get("SLASHKIT_DEB_PACKAGE")
    if package:
        source = Path(package).expanduser().resolve()
        if not source.is_file():
            raise LaunchError("SLASHKIT_DEB_PACKAGE is not a file: " + str(source), 2)
        copy_atomically(source, Path(root) / "docker/packages/slashkit.deb")
        selected.add("slashkit")
    return sorted(selected)


def read_manifest(path, names):
    """Values of a runtime manifest, which is a shell declaration file."""
    script = 'unset %s; . "$1"; printf "%%s\\0" %s' % (
        " ".join(names),
        " ".join('"${%s:-}"' % name for name in names),
    )
    try:
        output = subprocess.check_output(["sh", "-e", "-c", script, "runtime-manifest", str(path)])
    except (OSError, subprocess.CalledProcessError) as error:
        raise LaunchError("cannot read runtime manifest %s: %s" % (path, error), 2)
    return dict(zip(names, output.decode().split("\0")))


def supplied_packages(runtimes, root=REPO):
    """The archives installed by the selected `SOURCE=supply` runtime manifests."""
    root = Path(root)
    packages = set()
    for name in normalize_runtimes(runtimes):
        manifest = root / "docker/runtimes" / (name + ".env")
        values = read_manifest(manifest, ("SOURCE", "FILES"))
        if values["SOURCE"] == "fetch":
            continue
        if values["SOURCE"] != "supply" or not values["FILES"].split():
            raise LaunchError("Invalid runtime SOURCE/FILES in " + str(manifest), 2)
        for filename in values["FILES"].split():
            relative = Path(filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise LaunchError(
                    "Runtime package must be relative to docker/packages: " + filename, 2
                )
            path = Path("docker/packages") / relative
            if not (root / path).is_file():
                raise LaunchError(
                    "Runtime %r requires readable package %s" % (name, root / path), 2
                )
            packages.add(path.as_posix())
    return sorted(packages)


def cache_xrt(runtimes, root=REPO):
    """Fill FINN_XRT_PATH with the pinned public XRT package, verified, and stage
    it for the build. Jenkins agents keep this cache across builds."""
    cache = os.environ.get("FINN_XRT_PATH")
    if not cache or "xrt" not in runtimes:
        return
    values = read_manifest(
        Path(root) / "docker/runtimes/xrt.env", ("SOURCE", "DEB", "URL", "FALLBACK_URL", "SHA256")
    )
    if values["SOURCE"] != "fetch":
        return
    if not values["SHA256"]:
        raise LaunchError("FINN_XRT_PATH reuse requires the XRT manifest's SHA256", 2)
    cache = Path(cache).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    package = cache / (values["DEB"] + ".deb")
    with (cache / ".finn-xrt.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not package.is_file():
            addresses = [a for a in (values["URL"], values["FALLBACK_URL"]) if a]
            with tempfile.TemporaryDirectory(dir=cache) as directory:
                temporary = Path(directory) / package.name
                for address in addresses:
                    try:
                        with urllib.request.urlopen(address, timeout=120) as response:
                            with temporary.open("wb") as output:
                                shutil.copyfileobj(response, output)
                        if sha256(temporary) != values["SHA256"]:
                            raise LaunchError("XRT download checksum mismatch: " + address, 2)
                        temporary.replace(package)
                        break
                    except (OSError, LaunchError):
                        if address == addresses[-1]:
                            raise
        if sha256(package) != values["SHA256"]:
            raise LaunchError("Cached XRT checksum mismatch: " + str(package), 2)
        copy_atomically(package, Path(root) / "docker/runtime-cache" / package.name)


# --------------------------------------------------------------------------
# Image identity. FINN's sources are not inputs: containers install the mounted
# checkout. Image IDs still identify the exact build.
# --------------------------------------------------------------------------

UBUNTU_TAG = "noble-20240605"


def image_revision(runtimes, root=REPO):
    """`img-<16 hex>`: a hash of the files listed in docker/image-inputs.txt, the
    selected supplied packages and the build arguments that change image content.

    Git records the owner's executable bit, not agent umasks, so modes are
    canonical and identity is stable across Jenkins nodes.
    """
    explicit = os.environ.get("FINN_IMAGE_REVISION")
    if explicit:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", explicit):
            raise LaunchError("FINN_IMAGE_REVISION contains characters invalid in a Docker tag", 2)
        return explicit
    root = Path(root)
    manifest = root / "docker/image-inputs.txt"
    if not manifest.is_file():
        raise LaunchError("image input manifest not found: %s" % manifest, 2)
    patterns = manifest.read_text().splitlines() + supplied_packages(runtimes, root)
    digest = hashlib.sha256(b"finn-image-inputs-v5\n")
    for pattern in patterns:
        if not pattern or pattern.startswith("#"):
            continue
        optional = pattern.startswith("?")
        pattern = pattern[1:] if optional else pattern
        found = False
        for path in sorted(glob.glob(pattern, root_dir=str(root), recursive=True)):
            full = root / path
            if not full.is_file() or re.search(r"(^|/)__pycache__/|\.pyc$|\.so$|\.egg-info/", path):
                continue
            found = True
            mode = "100755" if full.stat().st_mode & 0o100 else "100644"
            try:
                checksum = sha256(full)
            except OSError as error:
                raise LaunchError("cannot read image input %s: %s" % (path, error), 2)
            digest.update(("path=%s mode=%s sha256=%s\n" % (path, mode, checksum)).encode())
        if not found and not optional:
            raise LaunchError("image input pattern matched no files: " + pattern, 2)
    digest.update(("arg=UBUNTU_TAG=%s\n" % os.environ.get("UBUNTU_TAG", UBUNTU_TAG)).encode())
    return "img-" + digest.hexdigest()[:16]


def git_output(root, *args):
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


@dataclass
class Request:
    """What one launcher invocation builds or runs."""

    runtimes: list
    variant: str = ""
    revision: str = ""
    source_revision: str = "unknown"
    source_describe: str = "unknown"
    source_dirty: str = "unknown"
    rebuild: bool = False
    extra: list = field(default_factory=list)

    @property
    def target(self):
        """The Bake target: finn, finn-xrt, or the parameterized ones."""
        if self.variant == "release":
            return "finn-release"
        return {(): "finn", ("xrt",): "finn-xrt"}.get(tuple(self.runtimes), "finn-runtime")

    @property
    def tag(self):
        """The image tag; docker-bake.hcl computes the same for direct Bake use."""
        registry = os.environ.get("REGISTRY", "xilinx/finn")
        suffix = "".join("." + name for name in self.runtimes)
        if self.variant == "release":
            dirty = "-dirty" if self.source_dirty == "1" else ""
            return "%s:release-%s-%s%s%s" % (
                registry,
                self.revision,
                self.source_revision[:12],
                dirty,
                suffix,
            )
        return "%s:%s%s" % (registry, self.revision, suffix)

    def environment(self):
        """Provenance for Bake and containers."""
        return {
            "FINN_IMAGE_REVISION": self.revision,
            "FINN_SOURCE_REVISION": self.source_revision,
            "FINN_SOURCE_DESCRIBE": self.source_describe,
            "FINN_SOURCE_DIRTY": self.source_dirty,
            "FINN_RUNTIMES": ",".join(self.runtimes),
        }


def make_request(runtimes, variant="", source_root=REPO, image_root=REPO):
    """Select runtimes, stage their inputs and record the image and source identity."""
    runtimes = prepare_runtimes(runtimes, image_root)
    status = git_output(source_root, "status", "--porcelain", "--untracked-files=normal")
    request = Request(
        runtimes=runtimes,
        variant=variant,
        revision=image_revision(runtimes, image_root),
        source_revision=git_output(source_root, "rev-parse", "HEAD") or "unknown",
        source_describe=git_output(
            source_root, "describe", "--always", "--tags", "--abbrev=12", "--dirty"
        )
        or "unknown",
        source_dirty="unknown" if status is None else ("1" if status else "0"),
    )
    os.environ.update(request.environment())
    return request


# --------------------------------------------------------------------------
# Building with Bake.
# --------------------------------------------------------------------------


def bake_build(request):
    """Build one Bake target into the local daemon under the request's tag."""
    cache_xrt(request.runtimes)
    target = request.target
    argv = ["docker", "buildx", "bake", "-f", "docker-bake.hcl", "--load"]
    argv += ["--set", target + ".output=type=docker,compression=gzip,compression-level=1"]
    argv += ["--set", target + ".tags=" + request.tag]
    if request.rebuild:
        argv.append("--no-cache")
    # Legacy build flags are a word list, never evaluated as shell code.
    argv += request.extra + [target]
    warn("building " + target)
    try:
        result = subprocess.run(argv, cwd=REPO, env=dict(os.environ, **request.environment()))
    except FileNotFoundError:
        result = None
    if result is None or result.returncode:
        raise LaunchError(
            "docker buildx bake %s failed. Docker Buildx with Bake is required; install "
            "docker-buildx-plugin and check 'docker buildx bake --help'." % target
        )


def inspect_image(reference):
    """The daemon's description of an image, or None if it has none."""
    result = subprocess.run(
        ["docker", "image", "inspect", reference], capture_output=True, text=True
    )
    if result.returncode:
        return None
    return json.loads(result.stdout)[0]


def verify_image(inspected, request):
    """Check an image's labels against the request; return its immutable ID."""
    image_id = inspected["Id"]
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise LaunchError("Invalid Docker image ID: " + image_id)
    labels = inspected.get("Config", {}).get("Labels") or {}
    revision = labels.get("dev.finn.image-revision")
    if revision != request.revision:
        raise LaunchError(
            "Image environment %r differs from requested %r" % (revision, request.revision)
        )
    mode = labels.get("dev.finn.source-mode", "checkout")
    expected_mode = "embedded" if request.variant == "release" else "checkout"
    if mode != expected_mode:
        raise LaunchError("Image source mode %r; requested %r" % (mode, expected_mode))
    if labels.get("dev.finn.runtimes", "").split(",") != (request.runtimes or [""]):
        raise LaunchError("Image runtimes differ from the requested runtime set")
    return image_id


def prepare_image(request, build=False, no_build=False):
    """Return the verified ID of the image to run, building it if needed.

    Explicit builds refresh cached layers; runs reuse an image with the
    environment's tag. A shared publication (FINN_DOCKER_SHARED_IMAGE_DIR) is
    preferred; with FINN_DOCKER_PREBUILT=1 it is required.
    """
    prebuilt = os.environ.get("FINN_DOCKER_PREBUILT", "0") == "1"
    shared = os.environ.get("FINN_DOCKER_SHARED_IMAGE_DIR")
    if shared and not build:
        try:
            return load_image(shared, request)
        except (OSError, ValueError, LaunchError, RuntimeError) as error:
            if prebuilt:
                raise LaunchError(str(error))
            warn("shared image unavailable (%s); falling back to the local image" % error)
    reference = os.environ.get("FINN_IMAGE") or request.tag
    inspected = None if build else inspect_image(reference)
    if no_build or prebuilt:
        if inspected is None:
            raise LaunchError(
                "No prepared Docker image for %s; run ./docker/build first." % reference
            )
        return verify_image(inspected, request)
    if inspected is not None and not request.rebuild and not request.extra:
        return verify_image(inspected, request)
    bake_build(request)
    inspected = inspect_image(request.tag)
    if inspected is None:
        raise LaunchError("Bake did not produce " + request.tag)
    # Execute by immutable ID, including local runs.
    return verify_image(inspected, request)


# --------------------------------------------------------------------------
# Image transport over shared storage. One build publishes; every other agent
# loads the same bytes and executes them by immutable ID.
# --------------------------------------------------------------------------

FILES = ("finn-docker-tag.txt", "finn-image-digest.txt", "finn-image-provenance.json")
ARCHIVE = "finn-docker-image.tar.gz"
IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")


def docker(*args):
    """Run the daemon CLI, keeping stdout available for machine data."""
    return subprocess.check_output(["docker", *args], text=True).strip()


def image_metadata(reference, request):
    """Describe actual bytes and their environment and mounted-source inputs."""
    inspected = inspect_image(reference)
    if inspected is None:
        raise LaunchError("No Docker image " + reference)
    labels = inspected.get("Config", {}).get("Labels") or {}
    return {
        "schema": 1,
        "image_id": verify_image(inspected, request),
        "environment_revision": request.revision,
        "lock_sha256": sha256(REPO / "uv.lock"),
        "resources_sha256": sha256(REPO / "src/finn/resources.toml"),
        "source_revision": request.source_revision,
        "source_dirty": {"0": False, "1": True}.get(request.source_dirty),
        "source_mode": labels.get("dev.finn.source-mode", "checkout"),
        "embedded_source_revision": labels.get("org.opencontainers.image.revision"),
        "runtimes": request.runtimes,
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


def read_metadata(directory, request):
    """Fail closed on missing, stale, partial, or old two-file publications."""
    directory = Path(directory)
    try:
        info = json.loads((directory / FILES[2]).read_text())
        image_id = (directory / FILES[1]).read_text().strip()
        reference = (directory / FILES[0]).read_text().strip()
    except (OSError, ValueError) as error:
        raise LaunchError(
            "Incomplete image publication at %s: %s. Republish with "
            "`docker/launcher.py image publish`; legacy two-file archives require a rebuild."
            % (directory, error)
        )
    if (
        info.get("schema") != 1
        or not IMAGE_ID.fullmatch(image_id)
        or info.get("image_id") != image_id
    ):
        raise LaunchError("Malformed or mismatched image identity")
    if not reference or "\n" in reference:
        raise LaunchError("Malformed published image reference")
    for key, expected in (
        ("environment_revision", request.revision),
        ("lock_sha256", sha256(REPO / "uv.lock")),
        ("runtimes", request.runtimes),
    ):
        if info.get(key) != expected:
            raise LaunchError("Published %s=%r; requested %r" % (key, info.get(key), expected))
    return image_id, reference, info


def compress(source, output):
    """gzip a stream into output, hashing the compressed bytes as they are written.

    pigz uses every core; without it, Python's gzip is a slower fallback.
    """
    digest = hashlib.sha256()
    pigz = shutil.which("pigz")
    if pigz:
        process = subprocess.Popen(
            [pigz, "-1", "-n", "-p", str(os.cpu_count() or 1)], stdin=source, stdout=subprocess.PIPE
        )
        try:
            for block in iter(lambda: process.stdout.read(1 << 20), b""):
                digest.update(block)
                output.write(block)
        except BaseException:
            # pigz's own SIGTERM cleanup can stall with threads running; it
            # holds nothing worth keeping.
            process.kill()
            process.wait()
            raise
        finally:
            process.stdout.close()
        if process.wait():
            raise RuntimeError("pigz failed")
        return digest.hexdigest()

    class Hashing:
        def write(self, block):
            digest.update(block)
            return output.write(block)

        def flush(self):
            output.flush()

    with gzip.GzipFile(fileobj=Hashing(), mode="wb", compresslevel=1, mtime=0) as compressed:
        shutil.copyfileobj(source, compressed, 1 << 20)
    return digest.hexdigest()


def stop(process):
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def publish_image(directory, reference, workspace, request):
    """Publish one complete build directory by rename. Retries cannot replace it."""
    directory = Path(directory)
    info = image_metadata(reference, request)
    directory.parent.mkdir(parents=True, exist_ok=True)

    def previous_publication():
        previous_id, previous_ref, previous = read_metadata(directory, request)
        if (
            previous_id != info["image_id"]
            or previous["source_revision"] != info["source_revision"]
        ):
            raise LaunchError(
                "Build directory already contains another image/source: %s" % directory
            )
        # Shards load the archive, so a retry must not accept a damaged one.
        archive = directory / ARCHIVE
        try:
            intact = sha256(archive) == previous.get("archive_sha256")
        except OSError:
            intact = False
        if not intact:
            raise LaunchError(
                "Existing publication at %s has a missing or damaged archive; "
                "remove the directory and publish again" % directory
            )
        write_metadata(workspace, previous_ref, previous)

    if directory.is_dir() and any(directory.iterdir()):
        return previous_publication()
    with tempfile.TemporaryDirectory(dir=directory.parent, prefix=".finn-image-") as work:
        prepared = Path(work) / "publication"
        prepared.mkdir()
        # A private tag preserves docker-save transport semantics without racing
        # a rebuild that changes the human-facing tag during docker save.
        transport = "finn-transport:" + uuid.uuid4().hex
        docker("tag", info["image_id"], transport)
        try:
            with (prepared / ARCHIVE).open("wb") as output:
                process = subprocess.Popen(["docker", "save", transport], stdout=subprocess.PIPE)
                try:
                    info["archive_sha256"] = compress(process.stdout, output)
                except BaseException:
                    stop(process)
                    raise
                finally:
                    process.stdout.close()
                if process.wait():
                    raise RuntimeError("docker save failed")
                output.flush()
                os.fsync(output.fileno())
        finally:
            docker("image", "rm", transport)
        info["requested_reference"] = reference
        info["transport_reference"] = transport
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
            previous_publication()


def _loaded(image_id, request):
    inspected = inspect_image(image_id)
    if inspected is None:
        return False
    if inspected["Id"] != image_id:
        raise LaunchError("Loaded image does not match the published identity")
    # A matching tag is insufficient; verify the content-addressed ID and labels.
    verify_image(inspected, request)
    return True


def _docker_load(archive, expected):
    """Stream an archive into docker load, checking its checksum on the way."""
    digest = hashlib.sha256()
    process = subprocess.Popen(["docker", "load"], stdin=subprocess.PIPE, stdout=sys.stderr)
    try:
        with open(archive, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
                process.stdin.write(block)
        if digest.hexdigest() != expected:
            # Abort before docker load sees the end of the stream.
            stop(process)
            raise LaunchError("Published image archive checksum mismatch")
        process.stdin.close()
    except BaseException:
        if process.poll() is None:
            stop(process)
        raise
    if process.wait():
        raise RuntimeError("docker load failed")


def load_image(directory, request):
    """Reuse verified image bytes, or serialize archive validation/load per host."""
    image_id, _, info = read_metadata(directory, request)
    archive = Path(directory) / ARCHIVE
    if not archive.is_file():
        raise LaunchError("Incomplete image publication: archive is missing")
    if not re.fullmatch(r"[0-9a-f]{64}", info.get("archive_sha256", "")):
        raise LaunchError("Malformed archive checksum metadata")
    if _loaded(image_id, request):
        return image_id
    # This is host-local, not an NFS lock: different agents/daemons may load in parallel.
    lock = os.environ.get("FINN_DOCKER_LOAD_LOCK", "/tmp/finn-docker-load.lock")
    with open(lock, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if _loaded(image_id, request):
            return image_id
        _docker_load(archive, info["archive_sha256"])
        if not _loaded(image_id, request):
            raise LaunchError("Loaded image does not match the published identity")
    return image_id


# --------------------------------------------------------------------------
# Release sources and SIF export.
# --------------------------------------------------------------------------


def snapshot_source(source, output):
    """Tracked working-tree files, including edits and deletions; never ignored
    downloads, notebook output or supplied runtime archives."""
    import tarfile  # noqa: PLC0415

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
                raise LaunchError("Source snapshot needs an expanded tracked file, not: " + name)
            info = archive.gettarinfo(str(path), arcname=name)
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            info.mode = 0o755 if info.mode & 0o100 else 0o644
            if info.isfile():
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
            else:
                archive.addfile(info)


def export_sif(image, output):
    runtime = shutil.which("apptainer") or shutil.which("singularity")
    if not runtime:
        raise LaunchError(
            "SIF export requires apptainer or singularity on PATH.\n"
            "  Ubuntu:  sudo apt install singularity-container\n"
            "  Or:      https://github.com/apptainer/apptainer/releases"
        )
    output = Path(hostpath(output))
    if output.is_dir():
        raise LaunchError("SIF output path is a directory: %s" % output, 2)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="finn-sif-") as work:
        archive = Path(work) / "image.tar"
        warn("exporting " + image)
        subprocess.run(["docker", "save", image, "-o", str(archive)], check=True)
        warn("converting to %s" % output)
        subprocess.run(
            [runtime, "build", "--force", str(output), "docker-archive://%s" % archive], check=True
        )
    print("exported SIF: %s sha256:%s" % (output, sha256(output)))


# --------------------------------------------------------------------------
# Migration from run-docker.sh.
# --------------------------------------------------------------------------

LEGACY = {
    "FINN_DOCKER_TAG": "tags are derived from the environment inputs; set FINN_IMAGE to "
    "select a prepared image",
    "FINN_SKIP_DEP_REPOS": "dependencies are locked in uv.lock and installed in the image",
    "FINN_SKIP_BOARD_FILES": "use a prepared resource cache with FINN_RESOURCES_OFFLINE=1",
    "FINN_SKIP_XRT_DOWNLOAD": "XRT is only installed with --runtime xrt",
    "XRT_DEB_VERSION": "XRT is pinned in docker/runtimes/xrt.env",
    "FINN_SINGULARITY": "use `docker/build --export-sif PATH` and run the SIF with apptainer",
    "FINN_DOCKER_NO_CACHE": "use `docker/build --rebuild`",
    "DOCKER_INTERACTIVE": "terminals are detected; pass Docker flags in FINN_DOCKER_EXTRA",
    "FINN_INST_NAME": "use `docker/run --name NAME`",
}


def warn_legacy():
    for variable, replacement in LEGACY.items():
        if os.environ.get(variable):
            warn("%s is no longer used: %s (see %s)." % (variable, replacement, DOCS))


# --------------------------------------------------------------------------
# Commands.
# --------------------------------------------------------------------------

BUILD_USAGE = """\
Usage: docker/build [OPTIONS]

Options:
  --release          build the release image: FINN installed from this checkout
  --export-sif PATH  build the release image and export it as an Apptainer SIF
  --runtime NAME     add image content such as xrt; repeatable
  --rebuild          rebuild image layers without cache
  --print-tag        print the image reference without building
  --print            print the resolved build request
  -h, --help         show this help
"""

RUN_USAGE = """\
Usage: docker/run [--fpga | --no-tools] [--runtime NAME] [--no-build] [--rebuild]
                  [--volume SPEC] [--env NAME[=VALUE]] [--workdir DIR]
                  [--name NAME] [--notebook] [--print] [-- COMMAND...]

The Xilinx tools are mounted whenever they are configured (FINN_XILINX_PATH and
FINN_XILINX_VERSION, or ~/.config/finn/xilinx.env).
  --fpga       require the tools: fail if none are configured
  --no-tools   never mount the tools
"""


def take_value(argv, index, option):
    """The value of `--option VALUE` or `--option=VALUE` at argv[index]."""
    if "=" in argv[index]:
        return argv[index].split("=", 1)[1], index + 1
    if index + 1 >= len(argv):
        raise LaunchError("%s requires a value" % option, 2)
    return argv[index + 1], index + 2


def command_build(argv):
    runtimes = os.environ.get("FINN_RUNTIMES", "").replace(",", " ").split()
    release = rebuild = print_only = print_tag = False
    sif = None
    i = 0
    while i < len(argv):
        option = argv[i].split("=", 1)[0]
        if option == "--release":
            release, i = True, i + 1
        elif option == "--export-sif":
            sif, i = take_value(argv, i, option)
            if not sif:
                raise LaunchError("--export-sif needs a path", 2)
        elif option == "--runtime":
            value, i = take_value(argv, i, option)
            runtimes.append(value)
        elif option == "--rebuild":
            rebuild, i = True, i + 1
        elif option == "--print":
            print_only, i = True, i + 1
        elif option == "--print-tag":
            print_tag, i = True, i + 1
        elif option in ("-h", "--help"):
            sys.stdout.write(BUILD_USAGE)
            return 0
        else:
            sys.stderr.write("Unknown option: %s\n%s" % (argv[i], BUILD_USAGE))
            return 2
    warn_legacy()
    variant = "release" if release or sif else ""
    request = make_request(runtimes, variant)
    request.rebuild = rebuild or os.environ.get("FINN_CONTAINER_REBUILD") == "1"
    request.extra = shlex.split(os.environ.get("FINN_DOCKER_BUILD_EXTRA", ""))
    if print_tag:
        print(request.tag)
        return 0
    if print_only:
        for key, value in (
            ("artifact", "sif" if sif else "docker"),
            ("runtimes", ",".join(request.runtimes)),
            ("bake_target", request.target),
            ("image_revision", request.revision),
            ("source_revision", request.source_revision),
            ("rebuild", "1" if request.rebuild else "0"),
            ("output", sif or ""),
            ("tag", request.tag),
        ):
            print("%s=%s" % (key, value))
        return 0
    with tempfile.TemporaryDirectory(prefix="finn-release-source-") as work:
        if variant == "release":
            snapshot_source(REPO, Path(work) / "source.tar")
            os.environ["FINN_RELEASE_SOURCE_DIR"] = work
        image = prepare_image(request, build=True)
    if sif:
        export_sif(image, sif)
    return 0


def command_run(argv):
    # Select the checkout containing this launcher, even after native activation of another.
    os.environ["FINN_ROOT"] = str(REPO)
    runtimes = os.environ.get("FINN_RUNTIMES", "").replace(",", " ").split()
    tier = "auto"
    print_only = False
    rebuild = os.environ.get("FINN_CONTAINER_REBUILD") == "1"
    no_build = os.environ.get("FINN_CONTAINER_NO_BUILD") == "1"
    options = {"volume": [], "env": [], "workdir": None, "name": None, "notebook": False}
    i = 0
    while i < len(argv):
        option = argv[i].split("=", 1)[0]
        if argv[i] == "--":
            i += 1
            break
        if option == "--fpga":
            tier, i = "build", i + 1
        elif option == "--no-tools":
            tier, i = "dev", i + 1
        elif option == "--runtime":
            value, i = take_value(argv, i, option)
            runtimes.append(value)
        elif option in ("--volume", "--env"):
            value, i = take_value(argv, i, option)
            options[option[2:]].append(value)
        elif option in ("--workdir", "--name", "-n"):
            value, i = take_value(argv, i, option)
            options["name" if option == "-n" else option[2:]] = value
        elif option == "--notebook":
            options["notebook"], i = True, i + 1
        elif option == "--rebuild":
            rebuild, i = True, i + 1
        elif option == "--no-build":
            no_build, i = True, i + 1
        elif option == "--print":
            print_only, i = True, i + 1
        elif option in ("-h", "--help"):
            sys.stdout.write(RUN_USAGE)
            return 0
        elif argv[i].startswith("-"):
            sys.stderr.write("Unknown option: %s\n%s" % (argv[i], RUN_USAGE))
            return 2
        else:
            break
    command = argv[i:]
    if rebuild and no_build:
        raise LaunchError("--rebuild and --no-build are mutually exclusive", 2)
    warn_legacy()
    request = make_request(runtimes)
    request.rebuild = rebuild
    request.extra = shlex.split(os.environ.get("FINN_DOCKER_BUILD_EXTRA", ""))
    # Resolve host inputs before an expensive build, and reuse exactly that request.
    data = resolve_host(tier, "mirror", request.runtimes)
    if tier == "build":
        require_tools(data)
    if print_only:
        print("tier=%s" % data["tier"])
        print("runtimes=%s" % ",".join(request.runtimes))
        print("bake_target=%s" % request.target)
        print("image_revision=%s" % request.revision)
        print("tag=%s" % request.tag)
        print("workspace=%s" % data["workspace"]["source"])
        print("command=%s" % " ".join(shlex.quote(word) for word in command))
        return 0
    image = prepare_image(request, no_build=no_build)
    argv = docker_run_argv(data, image, options, command)
    os.execvp(argv[0], argv)


def command_inspect(argv):
    parser = argparse.ArgumentParser(
        prog="docker/launcher.py inspect", description="resolve host configuration"
    )
    add_resolution_arguments(parser)
    parser.add_argument("--format", default="json", choices=("json", "sh"))
    args = parser.parse_args(argv)
    data = resolve_host(args.tier, args.workspace_policy)
    if args.format == "json":
        json.dump(data, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        # Native activation evaluates this; inspection never allocates directories.
        sys.stdout.write(sh_assignments(data))
    return 0


def command_compose(argv):
    parser = argparse.ArgumentParser(
        prog="docker/launcher.py compose", description="render a Compose override"
    )
    add_resolution_arguments(parser)
    parser.add_argument("--service", action="append", help="service to configure; repeatable")
    parser.add_argument(
        "--inputs-only",
        action="store_true",
        help="only toolchain, licence and resource inputs; the caller owns workspace and state",
    )
    args = parser.parse_args(argv)
    data = resolve_host(args.tier, args.workspace_policy)
    render = inputs_override if args.inputs_only else compose_override
    json.dump(render(data, args.service or [data["tier"]]), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def add_resolution_arguments(parser):
    parser.add_argument(
        "--tier", default=os.environ.get("FINN_DOCKER_TARGET", "auto"), choices=TIERS + ("auto",)
    )
    parser.add_argument("--workspace-policy", default="auto", choices=("auto", "fixed", "mirror"))


def command_image(argv):
    parser = argparse.ArgumentParser(prog="docker/launcher.py image", description="image transport")
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("metadata", help="record the image's provenance in a directory")
    p.add_argument("directory")
    p.add_argument("reference", nargs="?")
    p = sub.add_parser("publish", help="publish the image to a shared directory")
    p.add_argument("directory")
    p.add_argument("reference", nargs="?")
    p.add_argument("--workspace", required=True)
    p = sub.add_parser("load", help="load a published image; print its ID")
    p.add_argument("directory")
    for p in sub.choices.values():
        p.add_argument("--runtime", action="append", default=[])
    args = parser.parse_args(argv)
    runtimes = args.runtime or os.environ.get("FINN_RUNTIMES", "").replace(",", " ").split()
    request = make_request(runtimes)
    if args.action == "load":
        print(load_image(args.directory, request))
    elif args.action == "metadata":
        reference = args.reference or request.tag
        write_metadata(args.directory, reference, image_metadata(reference, request))
    else:
        publish_image(args.directory, args.reference or request.tag, args.workspace, request)
    return 0


COMMANDS = {
    "build": command_build,
    "run": command_run,
    "inspect": command_inspect,
    "compose": command_compose,
    "image": command_image,
}


def _terminate(signum, frame):
    # Unwind temporary directories and child processes on a Jenkins abort.
    raise SystemExit(128 + signum)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        sys.stderr.write("usage: docker/launcher.py {%s} ...\n%s\n" % (",".join(COMMANDS), __doc__))
        return 2
    signal.signal(signal.SIGTERM, _terminate)
    try:
        return COMMANDS[argv[0]](argv[1:])
    except LaunchError as error:
        print("finn: error: %s" % error, file=sys.stderr)
        return error.code
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print("finn: error: %s" % error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
