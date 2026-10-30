# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Host resolution in docker/launcher.py: tools, licences, mounts and state.

These run outside a container against synthetic Xilinx trees, so they cover the
layouts and licence forms that a single development host cannot exercise.
"""
import pytest

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LAUNCHER = str(REPO / "docker/launcher.py")
sys.path.insert(0, str(REPO / "docker"))
import launcher  # noqa: E402

pytestmark = pytest.mark.util

# Inside an --fpga container these name the real toolchain; tests choose their own.
TOOL_VARIABLES = (
    "XILINXD_LICENSE_FILE",
    "LM_LICENSE_FILE",
    "VIVADO_PATH",
    "VITIS_PATH",
    "HLS_PATH",
    "PLATFORM_REPO_PATHS",
)


@pytest.fixture
def host(tmp_path, monkeypatch):
    """An in-process host with no toolchain, host file or FINN settings."""
    for key in list(os.environ):
        if key.startswith(("FINN_", "XILINX_")) or key in TOOL_VARIABLES:
            monkeypatch.delenv(key)
    monkeypatch.setenv("FINN_XILINX_ENV", "")
    monkeypatch.setenv("FINN_HOME", str(tmp_path / "finn-home"))
    monkeypatch.setenv("FINN_HOST_BUILD_DIR", str(tmp_path / "build with spaces"))
    monkeypatch.setattr(launcher, "DEFAULT_PLATFORMS", str(tmp_path / "no-platforms"))
    return tmp_path


def make_tree(base, layout, version):
    """Build a synthetic Xilinx install in one of the two layouts."""
    if layout == "new":
        dirs = ["%s/Vivado" % version, "%s/Vitis" % version]
    else:
        dirs = ["Vivado/%s" % version, "Vitis/%s" % version, "Vitis_HLS/%s" % version]
    for d in dirs:
        os.makedirs(os.path.join(base, d))
    return str(base)


def launch(*args, env=None, cwd=None):
    """Run the real CLI in a clean environment, so nothing leaks in."""
    base = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ.get("HOME", "/tmp"),
        # No host file unless a test names one: the developer's own must not leak in.
        "FINN_XILINX_ENV": "",
    }
    base.update(env or {})
    return subprocess.run(
        [sys.executable, LAUNCHER, *args], capture_output=True, text=True, env=base, cwd=cwd
    )


def inspect(env, tier, cwd=None):
    proc = launch("inspect", "--tier", tier, env=env, cwd=cwd)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def host_file(tmp_path, text):
    path = tmp_path / "xilinx.env"
    path.write_text(text)
    return str(path)


# --------------------------------------------------------------------------
# AMD's install layouts.
# --------------------------------------------------------------------------


def test_new_layout_resolves(tmp_path):
    root = make_tree(tmp_path / "Xilinx", "new", "2025.1")
    found = launcher.xilinx_layout(root, "2025.1")
    assert found["XILINX_VIVADO"] == os.path.join(root, "2025.1", "Vivado")
    assert found["XILINX_VITIS"] == os.path.join(root, "2025.1", "Vitis")


def test_old_layout_resolves(tmp_path):
    root = make_tree(tmp_path / "Xilinx", "old", "2022.2")
    found = launcher.xilinx_layout(root, "2022.2")
    assert found["XILINX_VIVADO"] == os.path.join(root, "Vivado", "2022.2")
    assert found["XILINX_HLS"] == os.path.join(root, "Vitis_HLS", "2022.2")


def test_layout_is_probed_not_assumed(tmp_path):
    """A site whose tree disagrees with its version string still resolves."""
    root = make_tree(tmp_path / "Xilinx", "old", "2025.1")
    found = launcher.xilinx_layout(root, "2025.1")
    assert found["XILINX_VIVADO"] == os.path.join(root, "Vivado", "2025.1")


def test_missing_toolchain_yields_nothing(tmp_path):
    (tmp_path / "empty").mkdir()
    assert launcher.xilinx_layout(str(tmp_path / "empty"), "2022.2") == {}


def test_malformed_version_still_probes(tmp_path, capsys):
    """The version string only orders the two candidate layouts."""
    root = make_tree(tmp_path / "Xilinx", "old", "2022.2-patched")
    found = launcher.xilinx_layout(root, "2022.2-patched")
    assert "not YYYY.1" in capsys.readouterr().err
    assert found["XILINX_VIVADO"] == os.path.join(root, "Vivado", "2022.2-patched")


# --------------------------------------------------------------------------
# Licences.
# --------------------------------------------------------------------------


def test_license_lists_are_classified():
    assert launcher.classify_license("2100@licsrv.example") == (
        [{"host": "licsrv.example", "advertised_port": "2100"}],
        [],
    )
    assert launcher.classify_license("/opt/lic/Xilinx.lic") == ([], ["/opt/lic/Xilinx.lic"])
    servers, files = launcher.classify_license("2100@a.example:/opt/lic/x.lic::27000@b.example")
    assert [s["host"] for s in servers] == ["a.example", "b.example"]
    assert files == ["/opt/lic/x.lic"]
    assert launcher.classify_license("") == launcher.classify_license(None) == ([], [])


def test_vendor_port_comes_from_a_licence_file_or_the_environment(tmp_path, monkeypatch):
    """None means the whole host. It must never mean 'guess a port'."""
    monkeypatch.delenv("FINN_LICENSE_VENDOR_PORT", raising=False)
    lic = tmp_path / "Xilinx.lic"
    lic.write_text("SERVER licsrv 0011aabb 2100\nDAEMON xilinxd /opt/xilinx/xilinxd port=2101\n")
    assert launcher.vendor_daemon_port([str(lic)]) == "2101"
    lic.write_text("SERVER licsrv 0011aabb 2100\nDAEMON xilinxd /opt/xilinx/xilinxd\n")
    assert launcher.vendor_daemon_port([str(lic)]) is None
    assert launcher.vendor_daemon_port([]) is None
    monkeypatch.setenv("FINN_LICENSE_VENDOR_PORT", "2222")
    assert launcher.vendor_daemon_port([]) == "2222"
    monkeypatch.setenv("FINN_LICENSE_VENDOR_PORT", "not-a-port")
    assert launcher.vendor_daemon_port([]) is None


def test_node_locked_licences_are_mounted_exactly(tmp_path):
    """A licence file is mounted alone: its parent, e.g. /etc, would hide the
    image's own files. A licence directory is mounted as itself."""
    root = make_tree(tmp_path / "Xilinx", "new", "2025.1")
    licdir = tmp_path / "lic"
    licdir.mkdir()
    (licdir / "Xilinx.lic").write_text("x")
    sitedir = tmp_path / "site-licences"
    sitedir.mkdir()
    data = inspect(
        {
            "FINN_XILINX_PATH": root,
            "FINN_XILINX_VERSION": "2025.1",
            "XILINXD_LICENSE_FILE": str(licdir / "Xilinx.lic"),
            "LM_LICENSE_FILE": str(sitedir),
        },
        "build",
    )
    licences = [m["source"] for m in data["mounts"] if m["reason"] == "license-file"]
    assert licences == [str(licdir / "Xilinx.lic"), str(sitedir)]
    assert data["egress"] == []


def test_floating_license_requirements(tmp_path):
    """The whole host unless the vendor port is pinned, then both ports: lmgrd
    hands the checkout to the vendor daemon."""
    root = make_tree(tmp_path / "Xilinx", "new", "2025.1")
    env = {
        "FINN_XILINX_PATH": root,
        "FINN_XILINX_VERSION": "2025.1",
        "XILINXD_LICENSE_FILE": "2100@licsrv.example",
    }
    data = inspect(env, "build")
    assert data["egress"] == [
        {"host": "licsrv.example", "reason": "flexlm", "advertised_port": "2100", "ports": []}
    ]
    assert data["egress_enforcement"] == "declared"
    data = inspect(dict(env, FINN_LICENSE_VENDOR_PORT="2101"), "build")
    assert data["egress"][0]["ports"] == ["2100", "2101"]
    assert data["egress"][0]["vendor_port"] == "2101"


# --------------------------------------------------------------------------
# Tiers: tools are mounted when configured; --fpga requires them.
# --------------------------------------------------------------------------


def test_auto_tier_follows_the_configured_toolchain(host, monkeypatch):
    assert launcher.resolve_tier("auto") == "dev"
    monkeypatch.setenv("FINN_XILINX_PATH", str(host))
    assert launcher.resolve_tier("auto") == "build"
    monkeypatch.delenv("FINN_XILINX_PATH")
    assert launcher.resolve_tier("build") == "build"  # explicit tiers are requests


def test_dev_requires_nothing(tmp_path):
    """The Xilinx variables are set here deliberately: dev must ignore them."""
    root = make_tree(tmp_path / "Xilinx", "new", "2025.1")
    data = inspect(
        {
            "FINN_XILINX_PATH": root,
            "FINN_XILINX_VERSION": "2025.1",
            "XILINXD_LICENSE_FILE": "2100@licsrv.example",
            "PLATFORM_REPO_PATHS": str(tmp_path),
        },
        "dev",
    )
    assert data["mounts"] == [] and data["egress"] == []
    assert data["dev_contract"] == {"toolchain": False, "license": False, "egress": False}
    for leaked in ("XILINX_VIVADO", "XILINXD_LICENSE_FILE", "PLATFORM_REPO_PATHS"):
        assert leaked not in data["env"]


def test_build_mounts_root_read_only(tmp_path):
    root = make_tree(tmp_path / "Xilinx", "old", "2022.2")
    data = inspect({"FINN_XILINX_PATH": root, "FINN_XILINX_VERSION": "2022.2"}, "build")
    toolchain = [m for m in data["mounts"] if m["reason"] == "xilinx-toolchain"]
    assert toolchain == [
        {"source": root, "target": root, "mode": "ro", "reason": "xilinx-toolchain"}
    ]


def test_missing_xilinx_path_warns_without_blocking_inspection(tmp_path):
    proc = launch("inspect", "--tier", "build")
    assert proc.returncode == 0
    assert "FINN_XILINX_PATH" in proc.stderr
    assert json.loads(proc.stdout)["mounts"] == []


def test_stdout_is_pure_json_even_with_warnings(tmp_path):
    (tmp_path / "Xilinx").mkdir()
    proc = launch(
        "inspect",
        "--tier",
        "build",
        env={"FINN_XILINX_PATH": str(tmp_path / "Xilinx"), "FINN_XILINX_VERSION": "2022.2"},
    )
    assert proc.returncode == 0
    json.loads(proc.stdout)
    assert "no Vivado/Vitis/HLS found" in proc.stderr


def test_unknown_tier_is_rejected():
    proc = launch("inspect", "--tier", "nonsense")
    assert proc.returncode == 2
    assert "invalid choice: 'nonsense'" in proc.stderr


def run_print(*args, env=None):
    return launch("run", *args, "--print", env=dict(env or {}, FINN_IMAGE_REVISION="img-test"))


def test_run_mounts_configured_tools_without_a_flag(tmp_path):
    root = make_tree(tmp_path / "Xilinx", "new", "2025.2")
    tools = {"FINN_XILINX_PATH": root, "FINN_XILINX_VERSION": "2025.2"}
    assert "tier=build" in run_print(env=tools).stdout
    assert "tier=dev" in run_print("--no-tools", env=tools).stdout
    proc = run_print()
    assert proc.returncode == 0 and "tier=dev" in proc.stdout
    assert proc.stderr == ""  # no tools configured is an ordinary software setup


def test_fpga_fails_before_any_build_without_tools(tmp_path):
    proc = run_print("--fpga")
    assert proc.returncode == 3
    assert "--fpga needs the Xilinx tools" in proc.stderr
    (tmp_path / "Xilinx").mkdir()
    proc = run_print(
        "--fpga",
        env={"FINN_XILINX_PATH": str(tmp_path / "Xilinx"), "FINN_XILINX_VERSION": "2025.2"},
    )
    assert proc.returncode == 3  # a root with no tools of that version is not enough
    root = make_tree(tmp_path / "Real", "new", "2025.2")
    proc = run_print("--fpga", env={"FINN_XILINX_PATH": root, "FINN_XILINX_VERSION": "2025.2"})
    assert proc.returncode == 0, proc.stderr
    assert "tier=build" in proc.stdout


def test_build_tier_from_the_host_file_alone(tmp_path):
    """`docker/run --fpga` with nothing exported: the file supplies the toolchain
    and the licence, composed into FlexLM's PORT@HOST."""
    root = make_tree(tmp_path / "Xilinx", "new", "2025.2")
    make_tree(root, "new", "2026.1")
    path = host_file(
        tmp_path,
        "FINN_XILINX_PATH=%s\nFINN_XILINX_VERSION=2025.2\n"
        "FINN_LICENSE_HOST=192.0.2.1\nFINN_LICENSE_PORT=2100\nFINN_LICENSE_VENDOR_PORT=2101\n"
        % root,
    )
    data = inspect({"FINN_XILINX_ENV": path}, "auto")
    assert data["tier"] == "build"
    assert data["env"]["XILINX_VIVADO"] == os.path.join(root, "2025.2", "Vivado")
    assert data["env"]["XILINXD_LICENSE_FILE"] == "2100@192.0.2.1"
    assert data["egress"][0]["ports"] == ["2100", "2101"]
    other = inspect({"FINN_XILINX_ENV": path, "FINN_XILINX_VERSION": "2026.1"}, "build")
    assert other["env"]["XILINX_VIVADO"] == os.path.join(root, "2026.1", "Vivado")
    explicit = inspect({"FINN_XILINX_ENV": path, "XILINXD_LICENSE_FILE": "27000@lic"}, "build")
    assert explicit["env"]["XILINXD_LICENSE_FILE"] == "27000@lic"
    dev = inspect({"FINN_XILINX_ENV": path}, "dev")
    assert dev["mounts"] == [] and not any("XILINX" in key for key in dev["env"])


# --------------------------------------------------------------------------
# Platforms and tool overrides.
# --------------------------------------------------------------------------


def test_platform_repositories(host, tmp_path, monkeypatch):
    root = make_tree(tmp_path / "Xilinx", "new", "2025.1")
    monkeypatch.setenv("FINN_XILINX_PATH", root)
    monkeypatch.setenv("FINN_XILINX_VERSION", "2025.1")
    inside = Path(root) / "platforms"
    inside.mkdir()
    outside = tmp_path / "platforms"
    outside.mkdir()
    monkeypatch.setenv("PLATFORM_REPO_PATHS", "%s:%s:%s" % (inside, tmp_path / "absent", outside))
    data = launcher.resolve_host("build", "mirror")
    assert data["env"]["PLATFORM_REPO_PATHS"] == "%s:%s" % (inside, outside)
    assert [m["source"] for m in data["mounts"] if m["reason"] == "platform-repo"] == [str(outside)]
    # Unset, upstream's /opt/xilinx/platforms is used where it exists.
    monkeypatch.delenv("PLATFORM_REPO_PATHS")
    monkeypatch.setattr(launcher, "DEFAULT_PLATFORMS", str(outside))
    assert launcher.resolve_host("build", "mirror")["env"]["PLATFORM_REPO_PATHS"] == str(outside)
    monkeypatch.setenv("PLATFORM_REPO_PATHS", "")  # set empty: none
    assert "PLATFORM_REPO_PATHS" not in launcher.resolve_host("build", "mirror")["env"]


def test_missing_optional_installation_is_skipped(host, tmp_path, monkeypatch):
    hls = tmp_path / "HLS"
    hls.mkdir()
    monkeypatch.setenv("FINN_XILINX_PATH", str(tmp_path / "not-installed"))
    monkeypatch.setenv("HLS_PATH", str(hls))
    request = launcher.resolve_host("build", "mirror")
    assert request["env"]["XILINX_HLS"] == str(hls)
    assert all(Path(m["source"]).is_dir() for m in request["mounts"])


# --------------------------------------------------------------------------
# The host file: ~/.config/finn/xilinx.env.
# --------------------------------------------------------------------------


def test_host_file_is_read_with_comments(tmp_path):
    path = host_file(
        tmp_path, "# this machine\n\nFINN_XILINX_PATH=/opt/Xilinx\nFINN_XILINX_VERSION=2025.2\n"
    )
    assert launcher.read_host_file(path) == {
        "FINN_XILINX_PATH": "/opt/Xilinx",
        "FINN_XILINX_VERSION": "2025.2",
    }


@pytest.mark.parametrize(
    "line, complaint",
    [
        ("XILINX_VIVADO=/opt/Xilinx/2025.2/Vivado", "not a setting"),
        ("FINN_XILINX_PATH=~/Xilinx", "absolute"),
        ("FINN_XILINX_PATH=Xilinx", "absolute"),
        ('FINN_XILINX_VERSION="2025.2"', "quotes"),
        ("FINN_LICENSE_PORT=2100 # lmgrd", "trailing comment"),
        ("FINN_XILINX_VERSION", "NAME=value"),
    ],
)
def test_host_file_rejects_ambiguous_shell_syntax(tmp_path, line, complaint):
    """Host defaults have one literal interpretation in every consumer."""
    with pytest.raises(launcher.LaunchError, match=complaint):
        launcher.read_host_file(host_file(tmp_path, line + "\n"))


def test_host_file_location_and_precedence(tmp_path):
    path = host_file(tmp_path, "FINN_XILINX_PATH=/opt/Xilinx\nFINN_XILINX_VERSION=2025.2\n")
    values = launcher.host_settings({"FINN_XILINX_ENV": path, "FINN_XILINX_VERSION": "2026.1"})
    assert values == {"FINN_XILINX_PATH": "/opt/Xilinx", "FINN_XILINX_VERSION": "2026.1"}
    config = tmp_path / "config" / "finn"
    config.mkdir(parents=True)
    (config / "xilinx.env").write_text("FINN_XILINX_VERSION=2025.2\n")
    assert launcher.host_settings({"XDG_CONFIG_HOME": str(tmp_path / "config")}) == {
        "FINN_XILINX_VERSION": "2025.2"
    }
    home = {"HOME": str(tmp_path / "nobody")}
    assert launcher.host_file_path(home) == str(tmp_path / "nobody/.config/finn/xilinx.env")
    assert launcher.host_settings(home) == {}
    with pytest.raises(launcher.LaunchError, match="does not exist"):
        launcher.host_settings({"FINN_XILINX_ENV": str(tmp_path / "missing.env")})
    assert launcher.host_settings({"FINN_XILINX_ENV": ""}) == {}


def test_a_malformed_host_file_is_an_error(tmp_path):
    proc = launch(
        "inspect",
        "--tier",
        "dev",
        env={"FINN_XILINX_ENV": host_file(tmp_path, "XILINX_VIVADO=/x\n")},
    )
    assert proc.returncode == 3
    assert "not a setting" in proc.stderr


@pytest.mark.parametrize("licensed", (True, False))
def test_the_host_files_licence_reaches_activation(tmp_path, licensed):
    """scripts/activate.sh evals `inspect --format sh`: a licence server named in
    the host file arrives as XILINXD_LICENSE_FILE, and without one nothing is said."""
    root = make_tree(tmp_path / "Xilinx", "new", "2025.1")
    lines = ["FINN_XILINX_PATH=%s" % root, "FINN_XILINX_VERSION=2025.1"]
    if licensed:
        lines += ["FINN_LICENSE_HOST=licsrv.example", "FINN_LICENSE_PORT=2100"]
    proc = launch(
        "inspect",
        "--tier",
        "build",
        "--format",
        "sh",
        env={"HOME": str(tmp_path), "FINN_XILINX_ENV": host_file(tmp_path, "\n".join(lines))},
    )
    assert proc.returncode == 0, proc.stderr
    licence = [line for line in proc.stdout.splitlines() if "LICENSE" in line]
    assert licence == (["XILINXD_LICENSE_FILE='2100@licsrv.example'"] if licensed else [])
    assert "licen" not in proc.stderr.lower()


# --------------------------------------------------------------------------
# Workspace, build directory and FINN_HOME.
# --------------------------------------------------------------------------


def test_workspace_policy(tmp_path):
    """Tool tiers mirror the checkout (generated .xpr files embed absolute paths);
    dev uses a fixed path. FINN_ROOT is always the container path."""
    root = make_tree(tmp_path / "Xilinx", "new", "2025.1")
    tools = {"FINN_XILINX_PATH": root, "FINN_XILINX_VERSION": "2025.1"}
    data = inspect(dict(tools, FINN_ROOT="/somewhere/finn"), "build")
    assert data["workspace"] == {
        "policy": "mirror",
        "source": "/somewhere/finn",
        "target": "/somewhere/finn",
    }
    data = inspect({"FINN_ROOT": "/somewhere/finn"}, "dev")
    assert data["workspace"]["target"] == launcher.FIXED_WORKSPACE
    assert data["env"]["FINN_ROOT"] == data["workspace"]["target"]
    proc = launch("inspect", "--tier", "build", "--workspace-policy", "fixed", env=tools)
    assert proc.returncode == 2


def test_state_defaults_and_overrides(tmp_path):
    """FINN_HOME is persistent (default ~/.finn); the build directory is scratch
    (default /tmp/finn_build_<uid>). Each cache may live elsewhere."""
    data = inspect({"HOME": str(tmp_path)}, "dev")
    assert data["state"] == {"FINN_HOME": str(tmp_path / ".finn")}
    assert data["env"]["FINN_HOME"] == str(tmp_path / ".finn")
    assert data["build_dir"] == "/tmp/finn_build_%d" % os.getuid()
    data = inspect(
        {
            "HOME": str(tmp_path),
            "FINN_HOME": "~/state",
            "FINN_RESOURCES_DIR": str(tmp_path / "resources"),
            "FINN_XSI_PATH": str(tmp_path / "xsi"),
            "FINN_BUILD_DIR": str(tmp_path / "build"),
        },
        "dev",
    )
    assert data["state"] == {
        "FINN_HOME": str(tmp_path / "state"),
        "FINN_RESOURCES_DIR": str(tmp_path / "resources"),
        "FINN_XSI_PATH": str(tmp_path / "xsi"),
    }
    assert data["build_dir"] == str(tmp_path / "build")
    assert not any((tmp_path / name).exists() for name in ("state", "resources", "xsi", "build"))


def test_run_mounts_state_at_its_host_path(host, tmp_path, monkeypatch):
    monkeypatch.setenv("FINN_XSI_PATH", str(tmp_path / "shared xsi"))
    data = launcher.resolve_host("dev", "mirror")
    argv = launcher.docker_run_argv(data, "image", run_options(), ["true"])
    home = tmp_path / "finn-home"
    for path in (home, tmp_path / "shared xsi", tmp_path / "build with spaces"):
        assert path.is_dir()
        assert "%s:%s:rw" % (path, path) in argv
    assert "FINN_HOME=%s" % home in argv
    assert "FINN_IDENTITY_DIR=%s" % (home / "identity" / str(os.getuid())) in argv
    assert "HOME=/tmp/finn-home-%d" % os.getuid() in argv  # throwaway
    # The entrypoint completes the identity files; another launch keeps them.
    passwd = home / "identity" / str(os.getuid()) / "passwd"
    passwd.write_text("completed by the entrypoint\n")
    launcher.docker_run_argv(data, "image", run_options(), ["true"])
    assert passwd.read_text() == "completed by the entrypoint\n"


def test_shell_inspection_never_allocates(tmp_path):
    proc = launch(
        "inspect",
        "--tier",
        "dev",
        "--format",
        "sh",
        env={"FINN_HOST_BUILD_DIR": str(tmp_path / "scratch"), "FINN_HOME": str(tmp_path / "h")},
    )
    assert "FINN_BUILD_DIR='%s'" % (tmp_path / "scratch") in proc.stdout
    assert "FINN_HOME='%s'" % (tmp_path / "h") in proc.stdout
    assert not (tmp_path / "scratch").exists() and not (tmp_path / "h").exists()


def test_hostpath_expansion(monkeypatch):
    """A tilde in a variable is never expanded by a shell or by Compose."""
    monkeypatch.setenv("HOME", "/home/someone")
    assert launcher.hostpath("~/builds") == "/home/someone/builds"
    assert launcher.hostpath("x") == os.path.join(os.getcwd(), "x")
    assert launcher.hostpath("") == "" and launcher.hostpath(None) is None


def test_emitted_paths_have_no_tilde(tmp_path):
    root = make_tree(tmp_path / "Xilinx", "new", "2025.1")
    proc = launch(
        "inspect",
        "--tier",
        "build",
        "--format",
        "sh",
        env={
            "HOME": "/home/someone",
            "FINN_XILINX_PATH": root,
            "FINN_XILINX_VERSION": "2025.1",
            "FINN_HOST_BUILD_DIR": "~/builds",
            "FINN_ROOT": "~/finn",
        },
    )
    assert proc.returncode == 0, proc.stderr
    assert "~" not in proc.stdout
    assert "FINN_HOST_BUILD_DIR='/home/someone/builds'" in proc.stdout
    assert "FINN_HOME='/home/someone/.finn'" in proc.stdout


# --------------------------------------------------------------------------
# The `sh` output: what scripts/activate.sh evaluates.
# --------------------------------------------------------------------------


def test_sh_output(tmp_path):
    proc = launch(
        "inspect",
        "--tier",
        "dev",
        "--format",
        "sh",
        env={"FINN_ROOT": "/w/finn", "FINN_RUNTIMES": "xrt,slash,xrt", "FINN_XELAB_MT": "3"},
    )
    assert "FINN_WORKSPACE_SOURCE='/w/finn'" in proc.stdout
    assert "FINN_ROOT='%s'" % launcher.FIXED_WORKSPACE in proc.stdout
    assert "FINN_RUNTIMES='slash,xrt'" in proc.stdout
    assert "FINN_XELAB_MT='3'" in proc.stdout
    assert "FINN_UID=" in proc.stdout and "FINN_GID=" in proc.stdout
    assert "FINN_IMAGE=" not in proc.stdout


def test_sh_output_names_the_xilinx_path_only_for_tools(tmp_path):
    root = make_tree(tmp_path / "Xilinx", "new", "2025.1")
    tools = {"FINN_XILINX_PATH": root, "FINN_XILINX_VERSION": "2025.1"}
    dev = launch("inspect", "--tier", "dev", "--format", "sh", env=tools)
    assert "FINN_XILINX_PATH" not in dev.stdout
    build = launch("inspect", "--tier", "build", "--format", "sh", env=tools)
    assert "FINN_XILINX_PATH=" in build.stdout


def test_sh_output_survives_eval_with_spaces_and_quotes():
    assert launcher.shquote("/a b/c") == "'/a b/c'"
    assert launcher.shquote("x`id`") == "'x`id`'"
    assert launcher.shquote("a'b") == "'a'\\''b'"
    proc = subprocess.run(
        [
            "bash",
            "-c",
            "eval \"$(%s %s inspect --tier dev --format sh | sed 's/^/export /')\"; "
            'printf "%%s" "$FINN_WORKSPACE_SOURCE"' % (sys.executable, LAUNCHER),
        ],
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"], "HOME": "/home/someone", "FINN_ROOT": "/tmp/a b/finn"},
    )
    assert proc.stdout == "/tmp/a b/finn", proc.stderr


# --------------------------------------------------------------------------
# Resource overrides and site directories reach the container.
# --------------------------------------------------------------------------


def test_resource_overrides_are_mounted_and_passed(tmp_path):
    user_lib = tmp_path / "user_lib"
    user_lib.mkdir()
    cache = tmp_path / "resources"
    data = inspect(
        {
            "FINN_RESOURCES_USER_LIB": str(user_lib),
            "FINN_RESOURCES_HLSLIB": str(user_lib),
            "FINN_RESOURCES_DIR": str(cache),
            "FINN_RESOURCES_OFFLINE": "1",
            "FINN_RESOURCES_HLSLIB_URL": "https://git.example/hlslib.git",
            "FINN_RESOURCES_SYSTEM_CACHE": str(tmp_path),
        },
        "dev",
    )
    assert data["mounts"] == [
        {"source": str(user_lib), "target": str(user_lib), "mode": "rw", "reason": "resource"}
    ]
    assert data["state"]["FINN_RESOURCES_DIR"] == str(cache)
    assert data["env"]["FINN_RESOURCES_USER_LIB"] == str(user_lib)
    assert data["env"]["FINN_RESOURCES_OFFLINE"] == "1"
    assert data["env"]["FINN_RESOURCES_HLSLIB_URL"] == "https://git.example/hlslib.git"
    assert "FINN_RESOURCES_SYSTEM_CACHE" not in data["env"]  # the image's own


def test_relative_resource_override_becomes_absolute(tmp_path):
    (tmp_path / "user_lib").mkdir()
    data = inspect({"FINN_RESOURCES_USER_LIB": "user_lib"}, "dev", cwd=tmp_path)
    assert data["env"]["FINN_RESOURCES_USER_LIB"] == str(tmp_path / "user_lib")


def test_site_directories_must_exist(tmp_path):
    golden = tmp_path / "golden"
    proc = launch("inspect", "--tier", "dev", env={"FINN_GOLDEN_DIR": str(golden)})
    assert proc.returncode == 3 and "FINN_GOLDEN_DIR is not a directory" in proc.stderr
    golden.mkdir()
    data = inspect({"FINN_GOLDEN_DIR": str(golden)}, "dev")
    assert data["mounts"][0]["source"] == str(golden) and data["mounts"][0]["mode"] == "rw"


# --------------------------------------------------------------------------
# docker run.
# --------------------------------------------------------------------------


def run_options(**changes):
    options = {"volume": [], "env": [], "workdir": None, "name": None, "notebook": False}
    options.update(changes)
    return options


def test_run_preserves_arguments_and_site_flags(host, monkeypatch):
    monkeypatch.setenv("FINN_DOCKER_EXTRA", "--hostname farm-agent --env 'SITE_VALUE=a b'")
    data = launcher.resolve_host("dev", "mirror")
    options = run_options(
        volume=["/model with spaces:/model with spaces"],
        env=["VALUE=two words"],
        workdir="/model with spaces",
    )
    argv = launcher.docker_run_argv(
        data, "sha256:" + "a" * 64, options, ["python", "f.py", "x'`$()"]
    )
    assert argv[-4:] == ["sha256:" + "a" * 64, "python", "f.py", "x'`$()"]
    assert "SITE_VALUE=a b" in argv
    assert "/model with spaces:/model with spaces" in argv
    assert argv[argv.index("--hostname") + 1] == "farm-agent"
    assert argv[argv.index("--workdir") + 1] == "/model with spaces"


@pytest.mark.parametrize("terminal", [True, False])
def test_explicit_commands_receive_terminal_when_attached(host, monkeypatch, terminal):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: terminal)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: terminal)
    argv = launcher.docker_run_argv(
        launcher.resolve_host("dev"), "image", run_options(), ["ipython"]
    )
    assert ("-i" in argv) is terminal and ("-t" in argv) is terminal
    assert argv[-1] == "ipython"


def test_site_inputs_are_mounted(host, tmp_path, monkeypatch):
    io = tmp_path / "verification"
    io.mkdir()
    keys = tmp_path / "keys"
    keys.mkdir()
    boards = tmp_path / "site-boards"
    boards.mkdir()
    monkeypatch.setenv("VERIFICATION_IO", str(io))
    monkeypatch.setenv("FINN_SSH_KEY_DIR", str(keys))
    monkeypatch.setenv("FINN_BOARD_FILES_PATH", str(boards))
    argv = launcher.docker_run_argv(launcher.resolve_host("dev"), "image", run_options(), [])
    assert "VERIFICATION_IO=%s" % io in argv and "%s:%s:rw" % (io, io) in argv
    assert "%s:/run/finn-ssh:ro" % keys in argv
    assert "%s:%s:rw" % (boards, boards) in argv
    assert "FINN_BOARD_FILES_PATH=%s" % boards in argv
    assert argv[-1] == "bash"


def test_ssh_keys_default_to_the_checkouts_directory(host, tmp_path, monkeypatch):
    """Upstream's default, so existing setups keep password-less deployment."""
    monkeypatch.setattr(launcher, "REPO", tmp_path)
    argv = launcher.docker_run_argv(launcher.resolve_host("dev"), "image", run_options(), [])
    assert not any(arg.endswith(":/run/finn-ssh:ro") for arg in argv)
    (tmp_path / "ssh_keys").mkdir()
    argv = launcher.docker_run_argv(launcher.resolve_host("dev"), "image", run_options(), [])
    assert "%s:/run/finn-ssh:ro" % (tmp_path / "ssh_keys") in argv


def test_tool_libraries_are_in_the_creation_environment(host, tmp_path, monkeypatch):
    """Direct docker exec does not run the entrypoint."""
    root = tmp_path / "tools"
    for directory in ("Vivado/2024.2", "Vitis/2024.2", "Vitis_HLS/2024.2"):
        (root / directory).mkdir(parents=True)
    monkeypatch.setenv("FINN_XILINX_PATH", str(root))
    monkeypatch.setenv("FINN_XILINX_VERSION", "2024.2")
    env = launcher.container_environment(launcher.resolve_host("build", "mirror", ["xrt"]))
    assert env["HLS_PATH"] == str(root / "Vitis_HLS/2024.2")
    assert env["XILINX_XRT"] == "/opt/xilinx/xrt"
    assert str(root / "Vivado/2024.2/lib/lnx64.o") in env["LD_LIBRARY_PATH"].split(":")
    assert "/opt/xilinx/xrt/lib" in env["LD_LIBRARY_PATH"].split(":")


def test_run_selects_its_own_checkout(tmp_path):
    proc = run_print(env={"FINN_ROOT": "/a/different/checkout"})
    assert proc.returncode == 0, proc.stderr
    assert "workspace=%s" % REPO in proc.stdout


def test_broken_prebuilt_does_not_fall_back_to_a_local_image(tmp_path):
    proc = launch(
        "run",
        "--",
        "true",
        env={
            "FINN_IMAGE_REVISION": "img-test",
            "FINN_DOCKER_PREBUILT": "1",
            "FINN_DOCKER_SHARED_IMAGE_DIR": str(tmp_path / "missing"),
            "FINN_HOME": str(tmp_path / "home"),
        },
    )
    assert proc.returncode != 0
    assert "Incomplete image publication" in proc.stderr
    assert "building" not in proc.stderr


def test_legacy_variables_point_to_their_replacement():
    proc = run_print(env={"FINN_DOCKER_TAG": "x", "FINN_SKIP_DEP_REPOS": "1"})
    assert proc.returncode == 0
    assert "FINN_DOCKER_TAG is no longer used" in proc.stderr
    assert "FINN_SKIP_DEP_REPOS is no longer used" in proc.stderr
    assert "docker/README.md" in proc.stderr


@pytest.mark.parametrize(
    "args, expected",
    [
        ([], "docker/run"),
        (["test"], "docker/run --fpga -- python -m pytest"),
        (["quicktest", "main"], "docker/run -- bash docker/quicktest.sh main"),
        (["notebook"], "docker/run --notebook"),
        (["build_custom", "/m d", "flow", "--x"], "--workdir /m\\ d -- python flow.py --x"),
        (["python", "a b.py"], "docker/run -- python a\\ b.py"),
    ],
)
def test_run_docker_stub_prints_the_replacement(args, expected):
    proc = subprocess.run([str(REPO / "run-docker.sh"), *args], capture_output=True, text=True)
    assert proc.returncode == 2
    assert expected in proc.stderr


def test_nothing_requests_host_privilege():
    """No launcher asks for --privileged, an added capability, or the docker socket."""
    for rel in ("compose.yaml", "docker/launcher.py"):
        for number, line in enumerate((REPO / rel).read_text().splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            for needle in ("--privileged", "--cap-add", "/var/run/docker.sock"):
                assert needle not in line, "%s:%d %s" % (rel, number, needle)


# --------------------------------------------------------------------------
# Compose.
# --------------------------------------------------------------------------


def test_compose_override_consumes_resolved_mounts(tmp_path):
    root = make_tree(tmp_path / "Xilinx", "new", "2025.1")
    licdir = tmp_path / "lic"
    licdir.mkdir()
    licence = licdir / "Xilinx.lic"
    licence.write_text("FEATURE x\n")
    proc = launch(
        "compose",
        "--tier",
        "build",
        "--service",
        "build",
        env={
            "HOME": str(tmp_path),
            "FINN_XILINX_PATH": root,
            "FINN_XILINX_VERSION": "2025.1",
            "XILINXD_LICENSE_FILE": str(licence),
            "FINN_HOST_BUILD_DIR": str(tmp_path / "build"),
        },
    )
    assert proc.returncode == 0, proc.stderr
    service = json.loads(proc.stdout)["services"]["build"]
    mounts = {m["target"]: m for m in service["volumes"]}
    assert mounts[root]["read_only"] is True
    assert mounts[str(licence)]["read_only"] is True
    assert str(licdir) not in mounts
    assert "read_only" not in mounts[str(tmp_path / ".finn")]
    assert service["environment"]["XILINXD_LICENSE_FILE"] == str(licence)
    assert service["environment"]["FINN_HOME"] == str(tmp_path / ".finn")


def test_compose_uses_the_resolved_image(tmp_path):
    env = {
        "HOME": str(tmp_path),
        "FINN_HOST_BUILD_DIR": str(tmp_path / "build"),
        "FINN_RUNTIMES": "xrt",
    }
    proc = launch("compose", "--tier", "dev", "--service", "dev", env=env)
    assert proc.returncode == 2
    assert "FINN_IMAGE must be set" in proc.stderr
    proc = launch(
        "compose",
        "--tier",
        "dev",
        "--service",
        "dev",
        env=dict(env, FINN_IMAGE="xilinx/finn:test.xrt", FINN_IMAGE_REVISION="env-test"),
    )
    service = json.loads(proc.stdout)["services"]["dev"]
    assert service["image"] == "xilinx/finn:test.xrt"
    assert "build" not in service
    assert service["environment"]["FINN_IMAGE_REVISION"] == "env-test"
    compose = (REPO / "compose.yaml").read_text()
    assert "${FINN_IMAGE:-xilinx/finn:local}" in compose


def test_dev_container_inputs_carry_the_toolchain_but_not_owned_state(tmp_path):
    root = make_tree(tmp_path / "Xilinx", "new", "2025.2")
    path = host_file(
        tmp_path,
        "FINN_XILINX_PATH=%s\nFINN_XILINX_VERSION=2025.2\n"
        "FINN_LICENSE_HOST=192.0.2.1\nFINN_LICENSE_PORT=2100\n" % root,
    )
    proc = launch(
        "compose",
        "--tier",
        "auto",
        "--inputs-only",
        "--service",
        "dev",
        env={"HOME": str(tmp_path), "FINN_XILINX_ENV": path},
    )
    assert proc.returncode == 0, proc.stderr
    service = json.loads(proc.stdout)["services"]["dev"]
    assert [v["target"] for v in service["volumes"]] == [root]
    assert service["volumes"][0]["read_only"] is True
    assert service["environment"]["XILINXD_LICENSE_FILE"] == "2100@192.0.2.1"
    for owned in ("FINN_ROOT", "FINN_BUILD_DIR", "FINN_HOME"):
        assert owned not in service["environment"]
    assert set(service) == {"environment", "volumes"}
    proc = launch("compose", "--inputs-only", "--service", "dev", env={"HOME": str(tmp_path)})
    assert json.loads(proc.stdout)["services"]["dev"]["volumes"] == []
