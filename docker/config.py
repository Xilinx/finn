#!/usr/bin/env python3
"""Resolve FINN host configuration for Docker and native installation.

This module is host-side only. It is the sole owner of workspace, toolchain,
licence, mount and egress discovery for Docker/native callers. ``compose.yaml``
holds static Docker behavior; this executable renders shell assignments and
Compose overrides. The host file (``~/.config/finn/xilinx.env``) is read by
FINN's one reader, ``docker/host_config.py``, and AMD's install layouts
by ``xilinx_install.py``, which loads the reader and which native setup shares.
Network descriptions are declarative; Docker does not enforce these permissions.
Diagnostics go to stderr because stdout is machine-readable data.
"""

import argparse
import json
import os
import pwd
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import identity  # noqa: E402
import xilinx_install  # noqa: E402  (standard library only; docker/xilinx_install.py)

host_file = xilinx_install.host_file

# A tier is a host access profile, independent of the package artifact. What differs
# is what the launcher mounts and which network requirements it reports.
#
# `build-xrt` used to be a third tier. It differed from `build` by one mount,
# and its real content -- XRT -- is now a runtime target baked into the image
# and named in its tag. See docker/runtimes/README.md.
TIERS = ("dev", "build")

# Tiers that get a toolchain, a licence and declared network requirements. `dev`
# gets none of it: that boundary is the reason the tier exists, and it is
# enforced here rather than left to whether the caller's shell happens to have
# FINN_XILINX_PATH set.
#
# This is now the ONLY place the boundary is drawn. It used to be drawn twice --
# here, and again by which image you pulled -- and the two could disagree.
FPGA_TIERS = ("build",)


def normalize_runtimes(runtimes=None):
    """Return the runtime set in its canonical, duplicate-free order."""
    if runtimes is None:
        runtimes = os.environ.get("FINN_RUNTIMES", "")
    return sorted(set(n for n in runtimes.replace(",", " ").split() if n))


# Workspace mount policy is independent of Python package selection. Preserve
# the FPGA mirror policy for existing absolute build/checkpoint paths. Installed
# resource paths additionally require the same installation when replaying a
# project outside its original runtime; mirroring a checkout cannot supply it.
# The dev fixed path remains useful for remote daemons and Dev Containers.
FIXED_WORKSPACE = "/workspace/finn"
DEFAULT_DEV_WORKSPACE_POLICY = "fixed"


def warn(msg):
    print("docker/config.py: %s" % msg, file=sys.stderr)


def die(msg, code=1):
    print("docker/config.py: error: %s" % msg, file=sys.stderr)
    sys.exit(code)


# --------------------------------------------------------------------------
# Shared resolution. Used by both the host and container sides.
# --------------------------------------------------------------------------


def hostpath(value):
    """Expand ``~`` and make a host path absolute before it becomes a mount."""
    if not value:
        return value
    return os.path.abspath(os.path.expanduser(value))


def xilinx_layout(root, version):
    """Locate Vivado / Vitis / HLS within a Xilinx install (both AMD layouts)."""
    return xilinx_install.layout(root, version)


def vendor_daemon_port(files):
    """The pinned FLEXlm vendor-daemon port, if the site pinned one.

    FLEXlm needs TWO connections. lmgrd on the advertised port is a directory
    service; it hands back a second port where the vendor daemon (xilinxd) does
    the actual checkout. That second port is EPHEMERAL by default, which is why
    the declarative network requirement covers the whole host.

    It does not have to be ephemeral. A licence admin can pin it:

        DAEMON xilinxd /opt/xilinx/xilinxd port=2101

    When we can see a licence file, read it. When we cannot -- a bare PORT@HOST
    tells us nothing about the DAEMON line -- FINN_LICENSE_VENDOR_PORT lets the
    user state it.

    Returns None when unknown, meaning "report the whole host". Docker does not
    enforce these network requirements.
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
        m = re.search(r"^\s*DAEMON\s+\S+.*?\bport\s*=\s*(\d+)", text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1)
    return None


def classify_license(value):
    """Split a FLEXlm licence variable into server and file entries.

    XILINXD_LICENSE_FILE / LM_LICENSE_FILE take a colon-separated list of
    either form:

        PORT@HOST     floating server. Nothing to mount; needs TCP egress.
        /path/to.lic  node-locked file. Must be readable inside the container.

    The declared network requirement covers the whole HOST, because the vendor-daemon port is
    ephemeral unless the site pinned it -- see vendor_daemon_port(). A
    port-scoped rule against an unpinned daemon lets `lmutil lmstat` succeed,
    because lmstat only talks to lmgrd, while every real checkout fails with
    "A valid license was not found" -- a thoroughly misleading error for a
    firewall problem.

    So: narrow ONLY when the vendor port is known, and say so out loud when you
    do. lmstat is not a test of the narrowed rule.
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


def apply_machine_settings():
    """Read the host file into this process's environment, under the environment.

    The toolchain and licence code below reads os.environ; filling it here, once,
    keeps a variable set by the caller winning over the file without every reader
    knowing about the file. A licence given as host and port becomes the FlexLM
    form when no licence variable is set.
    """
    try:
        values = host_file.settings()
    except host_file.ConfigError as exc:
        die(str(exc), 3)
    for key, value in values.items():
        os.environ.setdefault(key, value)
    licence = host_file.license_for(os.environ, values)
    if licence:
        os.environ["XILINXD_LICENSE_FILE"] = licence


def resolve_tier(tier):
    """Resolve `auto` to a real tier, here rather than in every launcher.

    `auto` means "build if this machine has a toolchain, dev if it does not".
    That degrade used to live in the legacy launcher, which made it a fact derived in
    a launcher -- the exact shape of the four defects this program exists to
    prevent. It belongs here, once.

    `dev` and `build` stay EXPLICIT requests. An explicit `--tier build` with no
    toolchain is still a hard error: that is a request, not a default, and
    silently handing back a narrower environment than the caller asked for is
    how a CI shard ends up passing without testing anything.
    """
    if tier != "auto":
        return tier
    root = hostpath(os.environ.get("FINN_XILINX_PATH"))
    if root and os.path.isdir(root):
        return "build"
    warn(
        "no FINN_XILINX_PATH (environment or %s); resolving --tier auto to 'dev'. "
        "Vivado, Vitis, HLS and rtlsim will be unavailable."
        % (host_file.file_path() or "no host file")
    )
    return "dev"


def resolve_workspace(tier, policy="auto"):
    """Resolve the host and container workspace paths."""
    workspace_host = hostpath(
        os.environ.get("FINN_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    forced_mirror = tier in FPGA_TIERS
    if policy == "auto":
        policy = "mirror" if forced_mirror else DEFAULT_DEV_WORKSPACE_POLICY
    elif policy == "fixed" and forced_mirror:
        die(
            "workspace policy 'fixed' is incompatible with tier %r" % tier,
            2,
        )
    workspace_target = workspace_host if policy == "mirror" else FIXED_WORKSPACE
    return {"policy": policy, "source": workspace_host, "target": workspace_target}


def resolve_build_dir(create=False, fatal=False):
    """Resolve the per-user host build directory, optionally creating it."""
    path = hostpath(
        os.environ.get("FINN_HOST_BUILD_DIR")
        or os.environ.get("FINN_BUILD_DIR")
        or "/tmp/finn_build_%d" % os.getuid()
    )
    if create:
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as exc:
            message = "could not create the build directory %s (%s)" % (path, exc)
            if fatal:
                die(message, 3)
            warn(message)
    return path


def add_toolchain(out):
    """Add toolchain paths and their read-only root mount."""
    root = hostpath(os.environ.get("FINN_XILINX_PATH"))
    version = os.environ.get("FINN_XILINX_VERSION")
    if not root:
        die(
            "FINN_XILINX_PATH is unset (environment or %s); tier %r requires it"
            % (host_file.file_path() or "no host file", out["tier"]),
            3,
        )
    if not os.path.isdir(root):
        die("FINN_XILINX_PATH=%s is not a directory" % root, 3)

    out["mounts"].append(
        {"source": root, "target": root, "mode": "ro", "reason": "xilinx-toolchain"}
    )
    layout = xilinx_layout(root, version)
    if not layout:
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
            out["env"][alias] = value
            out["env"][name] = value
            if not value.startswith(root + os.sep) and value != root:
                out["mounts"].append(
                    {"source": value, "target": value, "mode": "ro", "reason": "tool-override"}
                )
    override = hostpath(os.environ.get("FINN_TOOL_DIR_OVERRIDE"))
    if override:
        if not os.path.isdir(override):
            die("FINN_TOOL_DIR_OVERRIDE is not a directory: " + override, 3)
        out["env"]["FINN_TOOL_DIR_OVERRIDE"] = override
        out["mounts"].append(
            {"source": override, "target": override, "mode": "ro", "reason": "site-tools"}
        )
    return root


def add_platform_repo(out, toolchain_root):
    """Add the optional platform repository without overlapping the toolchain."""
    paths = os.environ.get("PLATFORM_REPO_PATHS", "").split(os.pathsep)
    resolved = []
    for value in filter(None, paths):
        platforms = hostpath(value)
        if not os.path.isdir(platforms):
            die("platform repository is not a directory: " + platforms, 3)
        resolved.append(platforms)
        if platforms != toolchain_root and not platforms.startswith(toolchain_root + os.sep):
            out["mounts"].append(
                {"source": platforms, "target": platforms, "mode": "ro", "reason": "platform-repo"}
            )
    if resolved:
        out["env"]["PLATFORM_REPO_PATHS"] = os.pathsep.join(resolved)


def add_licenses(out):
    """Add FLEXlm environment, file mounts and declarative network requirements."""
    for var in ("XILINXD_LICENSE_FILE", "LM_LICENSE_FILE"):
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
        for path in files:
            directory = os.path.dirname(hostpath(path))
            if not os.path.isdir(directory):
                warn("licence path %s not found on the host; not mounted" % path)
                continue
            if not any(m["source"] == directory for m in out["mounts"]):
                out["mounts"].append(
                    {
                        "source": directory,
                        "target": directory,
                        "mode": "ro",
                        "reason": "license-file",
                    }
                )


def add_optional_inputs(out):
    """Add explicitly requested, tier-independent runtime inputs."""
    for variable in (
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
    ):
        if os.environ.get(variable):
            out["env"][variable] = os.environ[variable]
    for variable in ("FINN_BOARD_FILES_PATH", "FINN_GOLDEN_DIR", "FINN_LSF_NFS_STAGING"):
        value = hostpath(os.environ.get(variable))
        if value:
            if not os.path.isdir(value):
                die(variable + " is not a directory: " + value, 3)
            out["env"][variable] = value
            out["mounts"].append(
                {"source": value, "target": value, "mode": "rw", "reason": "site-state"}
            )
    imagenet = hostpath(os.environ.get("IMAGENET_VAL_PATH"))
    if not imagenet:
        return
    out["env"]["IMAGENET_VAL_PATH"] = imagenet
    if os.path.isdir(imagenet):
        out["mounts"].append(
            {"source": imagenet, "target": imagenet, "mode": "ro", "reason": "imagenet"}
        )
    else:
        warn("IMAGENET_VAL_PATH=%s is not a directory; not mounted" % imagenet)


RESOURCES = "FINN_RESOURCES_"
# finn.resources settings that are not resource overrides. The system cache is
# the image's own and is never replaced from the host.
RESOURCE_SETTINGS = ("FILES", "DIR", "OFFLINE", "SYSTEM_CACHE")


def add_resource_overrides(out):
    """Make finn.resources' host-side settings work inside the container.

    FINN_RESOURCES_<NAME> and FINN_RESOURCES_DIR name host directories: each is
    mounted at its own path, writable, since it is the user's checkout or cache.
    FINN_RESOURCES_FILES lists declaration files, whose directories are mounted
    read-only. Paths are passed in absolute, so a relative one means the same
    directory inside as on the host (relative to the checkout, where docker/run
    runs). Source URLs and the offline switch pass through unchanged.
    """

    def mount(path, mode, reason):
        if not os.path.isdir(path):
            warn("%s is not a directory; not mounted" % path)
        elif not any(m["source"] == path for m in out["mounts"]):
            out["mounts"].append({"source": path, "target": path, "mode": mode, "reason": reason})

    for variable, value in sorted(os.environ.items()):
        if not value or not variable.startswith(RESOURCES):
            continue
        setting = variable[len(RESOURCES) :]
        if setting == "SYSTEM_CACHE":
            continue
        if setting == "OFFLINE" or variable.endswith("_URL"):
            out["env"][variable] = value
        elif setting == "FILES":
            files = [hostpath(f) for f in value.split(os.pathsep) if f]
            for path in files:
                mount(os.path.dirname(path), "ro", "resource-declarations")
            out["env"][variable] = os.pathsep.join(files)
        else:
            path = hostpath(value)
            mount(path, "rw", "resource")
            out["env"][variable] = path


def resolve_host(tier, workspace_policy="auto"):
    """Everything a launcher needs, derived from the host exactly once."""
    apply_machine_settings()
    tier = resolve_tier(tier)
    if tier not in TIERS:
        die("unknown tier %r; expected one of %s" % (tier, ", ".join(TIERS)), 2)

    runtimes = normalize_runtimes()
    workspace = resolve_workspace(tier, workspace_policy)

    out = {
        "tier": tier,
        "backend": "docker",
        "runtimes": runtimes,
        "runtime_csv": ",".join(runtimes),
        "platform": "linux/amd64",
        "workspace": workspace,
        "build_dir": resolve_build_dir(create=False),
        "mounts": [],
        "egress": [],
        "egress_enforcement": "declared",
        "env": {
            "FINN_ROOT": workspace["target"],
        },
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
        out["dev_contract"] = {
            "toolchain": False,
            "license": False,
            "egress": False,
        }
        return out

    root = add_toolchain(out)
    add_platform_repo(out, root)
    add_licenses(out)

    return out


def shquote(value):
    """Quote a value for both shell ``eval`` and a Compose env file."""
    return "'" + str(value).replace("'", "'\\''") + "'"


def sh_assignments(data, create_build_dir=True):
    """Return the complete launcher/Compose environment as shell assignments."""
    build_dir = resolve_build_dir(create=create_build_dir)
    values = dict(data["env"])
    values.update(
        {
            "FINN_BUILD_DIR": build_dir,
            "FINN_HOST_BUILD_DIR": build_dir,
            "FINN_GID": str(os.getgid()),
            "FINN_RUNTIMES": data["runtime_csv"],
            "FINN_UID": str(os.getuid()),
            "FINN_WORKSPACE_SOURCE": data["workspace"]["source"],
            "FINN_WORKSPACE_TARGET": data["workspace"]["target"],
        }
    )
    if data["tier"] in FPGA_TIERS and os.environ.get("FINN_XILINX_PATH"):
        values["FINN_XILINX_PATH"] = hostpath(os.environ["FINN_XILINX_PATH"])
    if os.environ.get("FINN_IMAGE"):
        values["FINN_IMAGE"] = os.environ["FINN_IMAGE"]
    return "".join("%s=%s\n" % (key, shquote(value)) for key, value in sorted(values.items()))


def container_environment(data):
    """Persist essential tool inputs for processes started by docker exec too.

    Vendor settings still run in Bash. Bare Python/exec processes need the
    selected roots and simulation libraries in Docker's creation environment;
    they cannot inherit changes made later by the entrypoint.
    """
    env = dict(data["env"])
    uid, gid = (
        (0, 0) if os.environ.get("FINN_DOCKER_RUN_AS_ROOT") == "1" else (os.getuid(), os.getgid())
    )
    home = "/tmp/finn-home-%d" % uid
    try:
        username = pwd.getpwuid(uid).pw_name
    except KeyError:
        username = os.environ.get("USER") or "finn"
    build = resolve_build_dir(create=True, fatal=True)
    identity_dir = Path(build) / ".finn" / "identity" / str(uid)
    env.update(identity.prepare(identity_dir, uid, gid, username, home))
    env["FINN_IDENTITY_DIR"] = str(identity_dir)
    env.update(HOME=home, USER=username, LOGNAME=username)
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
        die(
            "FINN_IMAGE must be set when FINN_RUNTIMES is non-empty; "
            "use docker/run or resolve the tag through Bake",
            2,
        )
    build_dir = resolve_build_dir(create=True, fatal=True)
    volumes = [
        compose_bind(data["workspace"]["source"], data["workspace"]["target"]),
        compose_bind(build_dir, build_dir),
    ]
    seen_targets = {mount["target"] for mount in volumes}
    for mount in data["mounts"]:
        if mount["target"] in seen_targets:
            continue
        volumes.append(compose_bind(mount["source"], mount["target"], mount.get("mode", "rw")))
        seen_targets.add(mount["target"])

    environment = container_environment(data)
    environment.update(
        {
            "FINN_BUILD_DIR": build_dir,
            # Fetched resources and finn_xsi builds outlive the container with
            # the build directory; the container's own home does not.
            "FINN_HOME": build_dir + "/.finn",
            "FINN_RUNTIMES": data["runtime_csv"],
        }
    )
    service = {
        "environment": environment,
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


def cmd_inspect(args):
    data = resolve_host(args.tier, args.workspace_policy)
    if args.format == "json":
        json.dump(data, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        # Inspection is also used by native shell activation. Allocation belongs
        # to explicit Compose preparation or the build operation, not inspection.
        sys.stdout.write(sh_assignments(data, create_build_dir=False))
    return 0


def inputs_override(data, services):
    """Only the host inputs (toolchain, licence, resources), for a caller that
    owns the workspace, build directory and user itself: a caller that owns its workspace."""
    volumes = [compose_bind(m["source"], m["target"], m.get("mode", "rw")) for m in data["mounts"]]
    environment = {k: v for k, v in data["env"].items() if k != "FINN_ROOT"}
    service = {"environment": environment, "volumes": volumes}
    return {"services": {name: dict(service) for name in services}}


def cmd_compose(args):
    data = resolve_host(args.tier, args.workspace_policy)
    services = args.service or [data["tier"]]
    render = inputs_override if args.inputs_only else compose_override
    json.dump(render(data, services), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def add_resolution_arguments(parser):
    parser.add_argument(
        "--tier", default=os.environ.get("FINN_DOCKER_TARGET", "auto"), choices=TIERS + ("auto",)
    )
    parser.add_argument("--workspace-policy", default="auto", choices=("auto", "fixed", "mirror"))


def main():
    parser = argparse.ArgumentParser(prog="docker/config.py", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect", help="resolve host configuration")
    add_resolution_arguments(p)
    p.add_argument("--format", default="json", choices=("json", "sh"))
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("compose", help="render an ephemeral Compose override")
    add_resolution_arguments(p)
    p.add_argument(
        "--service", action="append", help="service to configure; repeat for multiple services"
    )
    p.add_argument(
        "--inputs-only",
        action="store_true",
        help="render only toolchain, licence and resource inputs; caller owns the workspace",
    )
    p.set_defaults(func=cmd_compose)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
