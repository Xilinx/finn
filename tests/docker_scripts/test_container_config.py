"""Tests for docker/config.py, the host/container environment resolver.

These run outside a container against synthetic Xilinx trees, so they cover the
layouts and licence forms that a single development host cannot exercise. That
is the whole point: the defects this resolver replaces were all cases where a
code path was only ever run against one host's configuration.

Both supported AMD installation layouts must resolve independent of the host
running these tests.
"""


import pytest

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOCKER_DIR = os.path.join(REPO, "docker")
FINN_ENV = os.path.join(DOCKER_DIR, "config.py")
sys.path.insert(0, DOCKER_DIR)
import config as finn_env  # noqa: E402
from config import host_file  # noqa: E402

pytestmark = pytest.mark.util


def _make_tree(base, layout, version):
    """Build a synthetic Xilinx install in one of the two layouts."""
    if layout == "new":
        dirs = ["%s/Vivado" % version, "%s/Vitis" % version]
    else:
        dirs = ["Vivado/%s" % version, "Vitis/%s" % version, "Vitis_HLS/%s" % version]
    for d in dirs:
        os.makedirs(os.path.join(base, d))
    return base


# --------------------------------------------------------------------------
# Layout resolution -- defect 2.
# --------------------------------------------------------------------------


def test_new_layout_resolves(tmp_path):
    root = _make_tree(str(tmp_path / "Xilinx"), "new", "2025.1")
    found = finn_env.xilinx_layout(root, "2025.1")
    assert found["XILINX_VIVADO"] == os.path.join(root, "2025.1", "Vivado")
    assert found["XILINX_VITIS"] == os.path.join(root, "2025.1", "Vitis")


def test_old_layout_resolves(tmp_path):
    """Resolve the ROOT/Vivado/VERSION layout used by older installations."""
    root = _make_tree(str(tmp_path / "Xilinx"), "old", "2022.2")
    found = finn_env.xilinx_layout(root, "2022.2")
    assert found["XILINX_VIVADO"] == os.path.join(root, "Vivado", "2022.2")
    assert found["XILINX_HLS"] == os.path.join(root, "Vitis_HLS", "2022.2")


def test_layout_is_probed_not_assumed(tmp_path):
    """A site whose tree disagrees with its version string still resolves.

    Deciding purely from the version number gives a confidently wrong answer on
    a symlinked or relocated install. Probing detects it.
    """
    root = _make_tree(str(tmp_path / "Xilinx"), "old", "2025.1")
    found = finn_env.xilinx_layout(root, "2025.1")
    assert found["XILINX_VIVADO"] == os.path.join(root, "Vivado", "2025.1")


def test_missing_toolchain_yields_nothing(tmp_path):
    empty = str(tmp_path / "empty")
    os.makedirs(empty)
    assert finn_env.xilinx_layout(empty, "2022.2") == {}


def test_malformed_version_still_probes(tmp_path, capsys):
    """An unparseable version warns but does not give up.

    The version string only orders the two candidate layouts; it is never the
    thing that decides. A site running a patched or vendor-relabelled release
    still resolves, and gets told its version string is unusual.
    """
    root = _make_tree(str(tmp_path / "Xilinx"), "old", "2022.2-patched")
    found = finn_env.xilinx_layout(root, "2022.2-patched")
    assert "not YYYY.1" in capsys.readouterr().err
    assert found["XILINX_VIVADO"] == os.path.join(root, "Vivado", "2022.2-patched")


# --------------------------------------------------------------------------
# Licence classification.
# --------------------------------------------------------------------------


def test_floating_license():
    servers, files = finn_env.classify_license("2100@licsrv.example")
    assert servers == [{"host": "licsrv.example", "advertised_port": "2100"}]
    assert files == []


def test_node_locked_license():
    servers, files = finn_env.classify_license("/opt/lic/Xilinx.lic")
    assert servers == []
    assert files == ["/opt/lic/Xilinx.lic"]


def test_mixed_license_list():
    servers, files = finn_env.classify_license("2100@a.example:/opt/lic/x.lic::27000@b.example")
    assert [s["host"] for s in servers] == ["a.example", "b.example"]
    assert files == ["/opt/lic/x.lic"]


def test_empty_license():
    assert finn_env.classify_license("") == ([], [])
    assert finn_env.classify_license(None) == ([], [])


# --------------------------------------------------------------------------
# Host resolution and the dev contract.
# --------------------------------------------------------------------------


def _inspect(env, tier):
    """Run the real CLI in a clean environment, so nothing leaks in."""
    # No host file unless a test names one: the developer's own must not leak in.
    base = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ.get("HOME", "/tmp"),
        "FINN_XILINX_ENV": "",
    }
    base.update(env)
    command = [FINN_ENV, "inspect", "--tier", tier]
    command.extend(["--format", "json"])
    proc = subprocess.run(
        command,
        capture_output=True,
        env=base,
        text=True,
    )
    return proc


def test_dev_requires_nothing(tmp_path):
    """D2's contract, asserted rather than assumed.

    The Xilinx variables are deliberately set here: dev must ignore them. The
    original bug class was a tier widening because a variable happened to be
    exported in the caller's shell.
    """
    root = _make_tree(str(tmp_path / "Xilinx"), "new", "2025.1")
    proc = _inspect(
        {
            "FINN_XILINX_PATH": root,
            "FINN_XILINX_VERSION": "2025.1",
            "XILINXD_LICENSE_FILE": "2100@licsrv.example",
            "PLATFORM_REPO_PATHS": "/opt/xilinx/platforms",
        },
        "dev",
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["mounts"] == []
    assert data["egress"] == []
    assert data["dev_contract"] == {"toolchain": False, "license": False, "egress": False}
    for leaked in ("XILINX_VIVADO", "XILINXD_LICENSE_FILE", "PLATFORM_REPO_PATHS"):
        assert leaked not in data["env"]


def test_build_mounts_root_read_only(tmp_path):
    """Defect 1: the root is mounted, and it is mounted :ro."""
    root = _make_tree(str(tmp_path / "Xilinx"), "old", "2022.2")
    proc = _inspect({"FINN_XILINX_PATH": root, "FINN_XILINX_VERSION": "2022.2"}, "build")
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    toolchain = [m for m in data["mounts"] if m["reason"] == "xilinx-toolchain"]
    assert len(toolchain) == 1
    assert toolchain[0]["source"] == root
    assert toolchain[0]["target"] == root
    assert toolchain[0]["mode"] == "ro"


def test_platform_repo_is_mounted(tmp_path):
    """Defect 3: it used to be exported as env with no mount."""
    root = _make_tree(str(tmp_path / "Xilinx"), "new", "2025.1")
    platforms = str(tmp_path / "platforms")
    os.makedirs(platforms)
    proc = _inspect(
        {
            "FINN_XILINX_PATH": root,
            "FINN_XILINX_VERSION": "2025.1",
            "PLATFORM_REPO_PATHS": platforms,
        },
        "build",
    )
    data = json.loads(proc.stdout)
    assert any(m["source"] == platforms and m["mode"] == "ro" for m in data["mounts"])
    assert data["env"]["PLATFORM_REPO_PATHS"] == platforms


def test_platform_repo_inside_root_is_not_double_mounted(tmp_path):
    root = _make_tree(str(tmp_path / "Xilinx"), "new", "2025.1")
    platforms = os.path.join(root, "platforms")
    os.makedirs(platforms)
    proc = _inspect(
        {
            "FINN_XILINX_PATH": root,
            "FINN_XILINX_VERSION": "2025.1",
            "PLATFORM_REPO_PATHS": platforms,
        },
        "build",
    )
    data = json.loads(proc.stdout)
    assert [m["source"] for m in data["mounts"]] == [root]


def test_node_locked_license_dir_is_mounted(tmp_path):
    """Licence file mounts survive host resolution."""
    root = _make_tree(str(tmp_path / "Xilinx"), "new", "2025.1")
    licdir = tmp_path / "lic"
    licdir.mkdir()
    (licdir / "Xilinx.lic").write_text("x")
    proc = _inspect(
        {
            "FINN_XILINX_PATH": root,
            "FINN_XILINX_VERSION": "2025.1",
            "XILINXD_LICENSE_FILE": str(licdir / "Xilinx.lic"),
        },
        "build",
    )
    data = json.loads(proc.stdout)
    assert any(m["source"] == str(licdir) and m["reason"] == "license-file" for m in data["mounts"])
    assert data["egress"] == []


def test_floating_license_grants_the_whole_host_when_unpinned(tmp_path):
    """An unpinned vendor daemon means a port-scoped grant would let lmstat
    pass -- it only talks to lmgrd -- while every real checkout failed."""
    root = _make_tree(str(tmp_path / "Xilinx"), "new", "2025.1")
    proc = _inspect(
        {
            "FINN_XILINX_PATH": root,
            "FINN_XILINX_VERSION": "2025.1",
            "XILINXD_LICENSE_FILE": "2100@licsrv.example",
        },
        "build",
    )
    data = json.loads(proc.stdout)
    assert data["egress"] == [
        {"host": "licsrv.example", "reason": "flexlm", "advertised_port": "2100", "ports": []}
    ]
    assert not any(m["reason"] == "license-file" for m in data["mounts"])


def test_floating_license_narrows_to_two_ports_when_pinned(tmp_path):
    """Both ports, never just the advertised one: lmgrd hands the checkout to
    the vendor daemon."""
    root = _make_tree(str(tmp_path / "Xilinx"), "new", "2025.1")
    proc = _inspect(
        {
            "FINN_XILINX_PATH": root,
            "FINN_XILINX_VERSION": "2025.1",
            "XILINXD_LICENSE_FILE": "2100@licsrv.example",
            "FINN_LICENSE_VENDOR_PORT": "2101",
        },
        "build",
    )
    data = json.loads(proc.stdout)
    assert data["egress"][0]["ports"] == ["2100", "2101"]
    assert data["egress"][0]["vendor_port"] == "2101"


def test_docker_egress_is_declarative(tmp_path):
    """The dev tier used to report egress:false on docker, where a container
    reaches pypi.org. Claiming a property you do not enforce is worse than not
    claiming it."""
    root = _make_tree(str(tmp_path / "Xilinx"), "new", "2025.1")
    env = {"FINN_XILINX_PATH": root, "FINN_XILINX_VERSION": "2025.1"}
    data = json.loads(_inspect(env, "build").stdout)
    assert data["egress_enforcement"] == "declared"


def test_missing_xilinx_path_warns_without_blocking_software(tmp_path):
    proc = _inspect({}, "build")
    assert proc.returncode == 0
    assert "FINN_XILINX_PATH" in proc.stderr
    assert json.loads(proc.stdout)["mounts"] == []


def test_unknown_tier_is_rejected():
    proc = subprocess.run(
        [FINN_ENV, "inspect", "--tier", "nonsense"], capture_output=True, text=True
    )
    # argparse's usage error, not a crash
    assert proc.returncode == 2
    assert "argument --tier: invalid choice: 'nonsense'" in proc.stderr


# --------------------------------------------------------------------------
# Workspace policy -- D4.
# --------------------------------------------------------------------------


def test_fpga_tiers_always_mirror(tmp_path):
    """LIMITATION(finn-root-absolute): generated .xpr files embed FINN_ROOT."""
    root = _make_tree(str(tmp_path / "Xilinx"), "new", "2025.1")
    proc = _inspect(
        {"FINN_XILINX_PATH": root, "FINN_XILINX_VERSION": "2025.1", "FINN_ROOT": "/somewhere/finn"},
        "build",
    )
    data = json.loads(proc.stdout)
    assert data["workspace"]["policy"] == "mirror"


def test_env_root_matches_workspace_target(tmp_path):
    """FINN_ROOT must be the CONTAINER path, never the host one."""
    proc = _inspect({"FINN_ROOT": "/somewhere/finn"}, "dev")
    data = json.loads(proc.stdout)
    assert data["env"]["FINN_ROOT"] == data["workspace"]["target"]


# --------------------------------------------------------------------------
# Output discipline.
# --------------------------------------------------------------------------


def test_stdout_is_pure_json_even_with_warnings(tmp_path):
    """Diagnostics go to stderr so callers can pipe stdout directly."""
    empty = str(tmp_path / "Xilinx")
    os.makedirs(empty)
    proc = _inspect({"FINN_XILINX_PATH": empty, "FINN_XILINX_VERSION": "2022.2"}, "build")
    assert proc.returncode == 0
    json.loads(proc.stdout)  # must not raise
    assert "no Vivado/Vitis/HLS found" in proc.stderr


def test_sh_format_is_shell_assignments(tmp_path):
    """The shell assignment format used by launchers and bare-host setup.

    Under the dev policy the SOURCE is the host checkout and the TARGET is the
    fixed path -- they are deliberately different, and conflating them is what
    would mount the workspace in the wrong place.
    """
    proc = subprocess.run(
        [FINN_ENV, "inspect", "--tier", "dev", "--format", "sh"],
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"], "FINN_ROOT": "/w/finn"},
    )
    # Values are single-quoted: three callers eval this output.
    assert "FINN_WORKSPACE_SOURCE='/w/finn'" in proc.stdout
    assert "FINN_WORKSPACE_TARGET='%s'" % finn_env.FIXED_WORKSPACE in proc.stdout
    # FINN_ROOT is the CONTAINER path, so it tracks the target, not the source.
    assert "FINN_ROOT='%s'" % finn_env.FIXED_WORKSPACE in proc.stdout
    # Launchers use these for Compose identity and bind ownership.
    assert "FINN_UID=" in proc.stdout
    assert "FINN_GID=" in proc.stdout


# --------------------------------------------------------------------------
# Idempotence.
# --------------------------------------------------------------------------

TOOLCHAIN_SH = os.path.join(os.path.dirname(FINN_ENV), "finn-toolchain.sh")


def _clean_env(env):
    """os.environ minus everything finn-toolchain.sh reads, plus `env`.

    Stripping matters: these tests also run INSIDE the container, where
    XILINX_VIVADO is set and a real settings64.sh would put Vivado's bin on
    PATH. Without this the dedup assertions compare against the host's
    toolchain instead of the fixture.

    BASH_ENV and ENV go too: bash reads them even with --noprofile --norc, and
    BASH_ENV may export XILINX_* again inside the child.
    """
    child = dict(os.environ)
    child.pop("FINN_ENV_APPLIED", None)
    for key in (
        "BASH_ENV",
        "VIRTUAL_ENV",
        "ENV",
        "XILINX_VIVADO",
        "XILINX_VITIS",
        "XILINX_HLS",
        "XILINX_XRT",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "PYTHONPATH",
    ):
        child.pop(key, None)
    child.update(env)
    return child


def _apply_toolchain(env, probe='printf "%s" "$PATH"'):
    """Source finn-toolchain.sh with a controlled environment and read a value.

    The dedup used to be finn_env.dedupe_paths(). It is shell now, because
    sourcing settings64.sh (7 ms) is far cheaper than the Python that existed to
    extract its effect (124 ms). The tests follow the code.
    """
    proc = subprocess.run(
        ["/bin/bash", "--noprofile", "--norc", "-c", ". %s; %s" % (TOOLCHAIN_SH, probe)],
        capture_output=True,
        text=True,
        env=_clean_env(env),
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_toolchain_dedupes_repeated_path_entries():
    """Repeated vendor settings must not grow PATH."""
    assert _apply_toolchain({"PATH": "/a:/b:/a:/c:/b"}) == "/a:/b:/c"


def test_toolchain_dedupes_ld_preload():
    out = _apply_toolchain({"PATH": "/a", "LD_PRELOAD": "/x:/x"}, probe='printf "%s" "$LD_PRELOAD"')
    assert out == "/x"


def test_toolchain_drops_empty_path_segments():
    assert _apply_toolchain({"PATH": "/a::/b:"}) == "/a:/b"


def test_toolchain_is_idempotent():
    """FINN_ENV_APPLIED short-circuits, so a nested shell must not re-apply."""
    proc = subprocess.run(
        [
            "/bin/bash",
            "--noprofile",
            "--norc",
            "-c",
            '. {0}; . {0}; printf "%s|%s" "$PATH" "$FINN_ENV_APPLIED"'.format(TOOLCHAIN_SH),
        ],
        capture_output=True,
        text=True,
        env=_clean_env({"PATH": "/a:/b"}),
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "/a:/b|1"


def test_toolchain_never_writes_to_stdout():
    """BASH_ENV sources this before every `bash -c`, so a stray echo corrupts
    the output of every command in the image."""
    assert _apply_toolchain({"PATH": "/a"}, probe="true") == ""


# --------------------------------------------------------------------------
# Tier resolution and the licence vendor port.
# --------------------------------------------------------------------------


def test_auto_tier_becomes_build_when_a_toolchain_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("FINN_XILINX_PATH", str(tmp_path))
    assert finn_env.resolve_tier("auto") == "build"


def test_auto_tier_degrades_to_dev_without_a_toolchain(monkeypatch):
    monkeypatch.delenv("FINN_XILINX_PATH", raising=False)
    assert finn_env.resolve_tier("auto") == "dev"


def test_explicit_tiers_are_not_degraded(monkeypatch):
    """`dev` and `build` are requests. Only `auto` is a question."""
    monkeypatch.delenv("FINN_XILINX_PATH", raising=False)
    assert finn_env.resolve_tier("build") == "build"
    assert finn_env.resolve_tier("dev") == "dev"


def test_vendor_port_read_from_a_licence_file(tmp_path, monkeypatch):
    monkeypatch.delenv("FINN_LICENSE_VENDOR_PORT", raising=False)
    lic = tmp_path / "Xilinx.lic"
    lic.write_text("SERVER licsrv 0011aabb 2100\nDAEMON xilinxd /opt/xilinx/xilinxd port=2101\n")
    assert finn_env.vendor_daemon_port([str(lic)]) == "2101"


def test_vendor_port_is_none_when_unpinned(tmp_path, monkeypatch):
    """None means grant the whole host. It must never mean 'guess a port'."""
    monkeypatch.delenv("FINN_LICENSE_VENDOR_PORT", raising=False)
    lic = tmp_path / "Xilinx.lic"
    lic.write_text("SERVER licsrv 0011aabb 2100\nDAEMON xilinxd /opt/xilinx/xilinxd\n")
    assert finn_env.vendor_daemon_port([str(lic)]) is None
    assert finn_env.vendor_daemon_port([]) is None


def test_vendor_port_env_override_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("FINN_LICENSE_VENDOR_PORT", "2222")
    assert finn_env.vendor_daemon_port([]) == "2222"


def test_vendor_port_rejects_a_non_numeric_override(monkeypatch):
    monkeypatch.setenv("FINN_LICENSE_VENDOR_PORT", "not-a-port")
    assert finn_env.vendor_daemon_port([]) is None


def test_dev_uses_the_fixed_workspace_path(tmp_path):
    """D4, as flipped in stage 7.

    dev is Python-only and generates no Vivado projects, so mirroring the host
    path buys it nothing while costing remote-daemon support and reproducible
    diagnostics. The FPGA tiers still mirror because generated .xpr files embed
    $::env(FINN_ROOT) as an absolute path.
    """
    proc = _inspect({"FINN_ROOT": "/somewhere/finn"}, "dev")
    data = json.loads(proc.stdout)
    assert data["workspace"]["policy"] == "fixed"
    assert data["workspace"]["source"] == "/somewhere/finn"
    assert data["workspace"]["target"] == finn_env.FIXED_WORKSPACE


# --------------------------------------------------------------------------
# Host path hygiene.
# --------------------------------------------------------------------------


def test_hostpath_expands_tilde(monkeypatch):
    """A tilde in a variable is never expanded by a shell or by Compose.

    This repo contains a stray directory literally named `~`, holding HLS output
    from a run where something passed `~/builds` through a variable. `.gitignore`
    has `*~` for editor backups, which hides it from `git status`.

    Every path docker/config.py emits can become a mount argument, so all are expanded.
    """
    monkeypatch.setenv("HOME", "/home/someone")
    assert finn_env.hostpath("~/builds") == "/home/someone/builds"


def test_hostpath_makes_relative_absolute():
    assert finn_env.hostpath("x") == os.path.join(os.getcwd(), "x")


def test_hostpath_passes_through_empty():
    assert finn_env.hostpath("") == ""
    assert finn_env.hostpath(None) is None


def test_emitted_paths_have_no_tilde(tmp_path):
    """End to end: a tilde must not reach launcher assignments."""
    root = _make_tree(str(tmp_path / "Xilinx"), "new", "2025.1")
    proc = subprocess.run(
        [FINN_ENV, "inspect", "--tier", "build", "--format", "sh"],
        capture_output=True,
        text=True,
        env={
            "PATH": os.environ["PATH"],
            "HOME": "/home/someone",
            "FINN_XILINX_PATH": root,
            "FINN_XILINX_VERSION": "2025.1",
            "FINN_HOST_BUILD_DIR": "~/builds",
            "FINN_ROOT": "~/finn",
        },
    )
    assert proc.returncode == 0, proc.stderr
    for line in proc.stdout.splitlines():
        assert "~" not in line, "tilde survived into output: %s" % line
    assert "FINN_HOST_BUILD_DIR='/home/someone/builds'" in proc.stdout


# --------------------------------------------------------------------------
# The `sh` output -- the format launchers and bare-host activation consume.
# --------------------------------------------------------------------------
#
# Every defect below was live and none was caught, because the existing tests
# checked the JSON output while Compose reads the sh output.


def test_sh_output_carries_the_canonical_runtime_set():
    proc = subprocess.run(
        [FINN_ENV, "inspect", "--tier", "dev", "--format", "sh"],
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"], "HOME": "/home/someone", "FINN_RUNTIMES": "xrt,slash,xrt"},
    )
    assert proc.returncode == 0, proc.stderr
    assert "FINN_RUNTIMES='slash,xrt'" in proc.stdout
    assert "FINN_IMAGE=" not in proc.stdout


def test_sh_output_carries_xelab_thread_override():
    proc = subprocess.run(
        [FINN_ENV, "inspect", "--tier", "dev", "--format", "sh"],
        capture_output=True,
        text=True,
        env={
            "PATH": os.environ["PATH"],
            "HOME": "/home/someone",
            "FINN_XELAB_MT": "3",
        },
    )
    assert proc.returncode == 0, proc.stderr
    assert "FINN_XELAB_MT='3'" in proc.stdout


def test_slashkit_has_a_transparent_tool_shim():
    # slashkit is one of the tools the shim loop links into the toolchain-bin directory
    assert re.search(
        r"for t in [^;\n]*\bslashkit\b[^;\n]*; do \\\n"
        r'\s*ln -sf /usr/local/lib/finn/toolchain-shim /usr/local/lib/finn/toolchain-bin/"\$t"',
        (Path(REPO) / "docker/Dockerfile.finn").read_text(),
    )


def test_sh_output_does_not_leak_xilinx_path_on_dev(tmp_path):
    """The dev contract, checked in the format that is actually consumed."""
    root = _make_tree(str(tmp_path / "Xilinx"), "new", "2025.1")
    proc = subprocess.run(
        [FINN_ENV, "inspect", "--tier", "dev", "--format", "sh"],
        capture_output=True,
        text=True,
        env={
            "PATH": os.environ["PATH"],
            "HOME": "/home/someone",
            "FINN_XILINX_PATH": root,
            "FINN_XILINX_VERSION": "2025.1",
        },
    )
    assert "FINN_XILINX_PATH" not in proc.stdout
    # ...and is present for a tier that may have a toolchain.
    proc = subprocess.run(
        [FINN_ENV, "inspect", "--tier", "build", "--format", "sh"],
        capture_output=True,
        text=True,
        env={
            "PATH": os.environ["PATH"],
            "HOME": "/home/someone",
            "FINN_XILINX_PATH": root,
            "FINN_XILINX_VERSION": "2025.1",
        },
    )
    assert "FINN_XILINX_PATH=" in proc.stdout


def test_sh_output_survives_eval_with_spaces():
    """Three callers eval this output. A space used to truncate the value."""
    assert finn_env.shquote("/a b/c") == "'/a b/c'"
    proc = subprocess.run(
        [
            "bash",
            "-c",
            "eval \"$(%s %s inspect --tier dev --format sh | sed 's/^/export /')\"; "
            'printf "%%s" "$FINN_WORKSPACE_SOURCE"' % (sys.executable, FINN_ENV),
        ],
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"], "HOME": "/home/someone", "FINN_ROOT": "/tmp/a b/finn"},
    )
    assert proc.stdout == "/tmp/a b/finn", proc.stderr


def test_sh_output_is_not_an_injection_path():
    """XILINXD_LICENSE_FILE passes through verbatim and is then eval'd."""
    assert finn_env.shquote("x`id`") == "'x`id`'"
    assert finn_env.shquote("a'b") == "'a'\\''b'"


# --------------------------------------------------------------------------
# Static launcher properties.
# --------------------------------------------------------------------------


def test_nothing_requests_host_privilege():
    """Container root does not require host privileges.

    "No privileges" is ambiguous and the wrong reading is the natural one. This
    asserts the reading that matters: no launcher asks for --privileged, an
    added capability, or the docker socket.
    """
    launchers = [
        "compose.yaml",
        "docker/run",
    ]
    offenders = []
    for rel in launchers:
        path = os.path.join(REPO, rel)
        if not os.path.exists(path):
            continue
        with open(path, errors="replace") as handle:
            for n, line in enumerate(handle, 1):
                if line.lstrip().startswith("#"):
                    continue
                for needle in ("--privileged", "--cap-add", "/var/run/docker.sock"):
                    if needle in line:
                        offenders.append("%s:%d %s" % (rel, n, needle))
    assert not offenders, offenders


def test_setup_local_has_not_regrown_the_hardcoded_layout():
    """The pre-2024.2 layout, hardcoded, was defect 4 of the original four.

        VIVADO_PATH="$FINN_XILINX_PATH/Vivado/$FINN_XILINX_VERSION"

    AMD reorganised the tree after 2024.2, so that reported "Vivado not found"
    at a path the user could see was right there. docker/config.py probes both.
    """
    for rel in ("setup-local.sh", "scripts/activate.sh"):
        with open(os.path.join(REPO, rel), errors="replace") as handle:
            body = handle.read()
        for n, line in enumerate(body.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            assert "$FINN_XILINX_PATH/Vivado/" not in line, "%s:%d" % (rel, n)


def test_custom_runtime_sets_use_the_parameterized_bake_target():
    proc = subprocess.run(
        [
            "bash",
            "-c",
            '. ./docker/lib.sh; printf "%s|%s|%s" '
            '"$(finn_bake_target xrt)" '
            '"$(finn_bake_target xrt,slash)" '
            '"$(finn_bake_target xrt,slash release)"',
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "finn-xrt|finn-runtime|finn-release"
    with open(os.path.join(REPO, "docker-bake.hcl"), errors="replace") as handle:
        bake = handle.read()
    assert 'target "finn-runtime"' in bake
    assert 'target "finn-release"' in bake


def _provenance(image_root, source_root=None, extra_env=None):
    env = {
        "PATH": os.environ["PATH"],
        "FINN_IMAGE_INPUT_ROOT": str(image_root),
    }
    if source_root is not None:
        env["FINN_SOURCE_ROOT"] = str(source_root)
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        [
            "bash",
            "-c",
            '. ./docker/lib.sh; finn_set_provenance; printf "%s|%s|%s|%s" '
            '"$FINN_IMAGE_REVISION" "$FINN_SOURCE_REVISION" '
            '"$FINN_SOURCE_DESCRIBE" "$FINN_SOURCE_DIRTY"',
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.split("|")


def _make_image_input_fixture(tmp_path):
    root = tmp_path / "image-root"
    (root / "docker").mkdir(parents=True)
    (root / "docker/image-inputs.txt").write_text("docker/image-inputs.txt\nimage.txt\n")
    (root / "image.txt").write_text("environment\n")
    (root / "src").mkdir()
    (root / "src/finn.py").write_text("source v1\n")
    return root


def test_image_revision_ignores_mounted_source_changes(tmp_path):
    root = _make_image_input_fixture(tmp_path)
    before = _provenance(root)[0]
    (root / "src/finn.py").write_text("source v2\n")
    after = _provenance(root)[0]
    assert before == after


def test_image_revision_changes_with_image_inputs_and_build_args(tmp_path):
    root = _make_image_input_fixture(tmp_path)
    original = _provenance(root)[0]
    (root / "image.txt").write_text("changed environment\n")
    changed_file = _provenance(root)[0]
    changed_arg = _provenance(root, extra_env={"UBUNTU_TAG": "jammy-override"})[0]
    assert original != changed_file
    assert changed_file != changed_arg


def test_source_commit_changes_without_changing_image_revision(tmp_path):
    image_root = _make_image_input_fixture(tmp_path)
    source_root = tmp_path / "source"
    source_root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=source_root, check=True)
    subprocess.run(["git", "config", "user.name", "FINN Test"], cwd=source_root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "finn-test@example.invalid"],
        cwd=source_root,
        check=True,
    )
    (source_root / "README").write_text("source\n")
    subprocess.run(["git", "add", "README"], cwd=source_root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=source_root, check=True)
    before = _provenance(image_root, source_root)
    subprocess.run(
        ["git", "commit", "-q", "--allow-empty", "-m", "source-only"],
        cwd=source_root,
        check=True,
    )
    after = _provenance(image_root, source_root)
    assert before[0] == after[0]
    assert before[1] != after[1]
    assert before[2] != after[2]
    assert before[3] == after[3] == "0"


def test_shell_inspection_never_allocates_scratch(tmp_path):
    scratch = tmp_path / "scratch"
    proc = subprocess.run(
        [FINN_ENV, "inspect", "--tier", "dev", "--format", "sh"],
        env={"PATH": os.environ["PATH"], "FINN_HOST_BUILD_DIR": str(scratch)},
        capture_output=True,
        text=True,
        check=True,
    )
    assert str(scratch) in proc.stdout
    assert not scratch.exists()


def test_compose_override_consumes_resolved_mounts(tmp_path):
    root = _make_tree(str(tmp_path / "Xilinx"), "new", "2025.1")
    licdir = tmp_path / "lic"
    licdir.mkdir()
    licence = licdir / "Xilinx.lic"
    licence.write_text("FEATURE x\n")
    proc = subprocess.run(
        [FINN_ENV, "compose", "--tier", "build", "--service", "build"],
        capture_output=True,
        text=True,
        env={
            "PATH": os.environ["PATH"],
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
    assert mounts[str(licdir)]["read_only"] is True
    assert service["environment"]["XILINXD_LICENSE_FILE"] == str(licence)


def test_compose_override_uses_the_bake_resolved_image(tmp_path):
    proc = subprocess.run(
        [FINN_ENV, "compose", "--tier", "dev", "--service", "dev"],
        capture_output=True,
        text=True,
        env={
            "PATH": os.environ["PATH"],
            "HOME": str(tmp_path),
            "FINN_HOST_BUILD_DIR": str(tmp_path / "build"),
            "FINN_IMAGE": "xilinx/finn:test.xrt",
            "FINN_IMAGE_REVISION": "env-test",
            "FINN_SOURCE_REVISION": "abc123",
            "FINN_SOURCE_DESCRIBE": "v1-test",
            "FINN_SOURCE_DIRTY": "0",
            "FINN_RUNTIMES": "xrt",
        },
    )
    assert proc.returncode == 0, proc.stderr
    service = json.loads(proc.stdout)["services"]["dev"]
    assert service["image"] == "xilinx/finn:test.xrt"
    assert "build" not in service  # The Bake-resolved image is authoritative.
    assert service["environment"]["FINN_IMAGE_REVISION"] == "env-test"
    assert service["environment"]["FINN_SOURCE_REVISION"] == "abc123"


def test_compose_rejects_runtime_content_without_a_resolved_image(tmp_path):
    proc = subprocess.run(
        [FINN_ENV, "compose", "--tier", "dev", "--service", "dev"],
        capture_output=True,
        text=True,
        env={
            "PATH": os.environ["PATH"],
            "HOME": str(tmp_path),
            "FINN_HOST_BUILD_DIR": str(tmp_path / "build"),
            "FINN_RUNTIMES": "xrt",
        },
    )
    assert proc.returncode == 2
    assert "FINN_IMAGE must be set" in proc.stderr


def test_python_dependency_pins_are_not_duplicated_in_installers():
    with open(os.path.join(REPO, "docker/Dockerfile.finn")) as handle:
        dockerfile = handle.read()
    with open(os.path.join(REPO, "setup-local.sh")) as handle:
        local_setup = handle.read()
    # The Python pins live in pyproject.toml and uv.lock, where the installers sync from;
    # no ``name==version`` pin (extras included) is repeated in either installer.
    pin = re.compile(r"[A-Za-z0-9_.\-]+(?:\[[A-Za-z0-9_,.\-]+\])?==[0-9]")
    assert pin.findall(dockerfile) == []
    assert pin.findall(local_setup) == []


def test_compose_uses_complete_image_references():
    with open(os.path.join(REPO, "compose.yaml")) as handle:
        compose = handle.read()
    assert "${FINN_IMAGE:-xilinx/finn:local}" in compose
    assert "FINN_RUNTIME_TAG" not in compose


def test_native_callers_execute_resolver_in_an_isolated_installation(tmp_path):
    """Exercise activation and setup's resolver call without installing anything."""
    checkout = tmp_path / "finn"
    for relative in (
        "scripts/activate.sh",
        "docker/config.py",
        "docker/identity.py",
        "docker/xilinx_install.py",
        "docker/finn-toolchain.sh",
        "docker/host_config.py",
    ):
        destination = checkout / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(REPO) / relative, destination)
    activate = checkout / ".venv/bin/activate"
    activate.parent.mkdir(parents=True)
    activate.write_text(": # test-owned virtual environment stand-in\n")
    (tmp_path / "hlslib").mkdir()
    (tmp_path / "boards").mkdir()
    root = _make_tree(str(tmp_path / "Xilinx"), "old", "2022.2")
    environment = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path / "home"),
        "FINN_ROOT": str(checkout),
        "FINN_RESOURCES_HLSLIB": str(tmp_path / "hlslib"),
        "FINN_BOARD_FILES_PATH": str(tmp_path / "boards"),
        "FINN_HOST_BUILD_DIR": str(tmp_path / "build"),
        "FINN_XILINX_PATH": root,
        "FINN_XILINX_VERSION": "2022.2",
    }
    expected = os.path.join(root, "Vivado", "2022.2")
    proc = subprocess.run(
        [
            "bash",
            "-c",
            'source "$FINN_ROOT/scripts/activate.sh"; '
            'test "$XILINX_VIVADO" = "$1" && test ! -e "$FINN_BUILD_DIR"',
            "test",
            expected,
        ],
        env=environment,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr


# --------------------------------------------------------------------------
# The shared native/container host file: ~/.config/finn/xilinx.env.
# --------------------------------------------------------------------------


def _host_file(tmp_path, text):
    path = tmp_path / "xilinx.env"
    path.write_text(text)
    return str(path)


def test_host_file_is_read_with_comments(tmp_path):
    path = _host_file(
        tmp_path, "# this machine\n\nFINN_XILINX_PATH=/opt/Xilinx\nFINN_XILINX_VERSION=2025.2\n"
    )
    assert host_file.read_file(path) == {
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
    path = _host_file(tmp_path, line + "\n")
    with pytest.raises(host_file.ConfigError, match=complaint):
        host_file.read_file(path)


def test_environment_wins_over_the_host_file(tmp_path):
    path = _host_file(tmp_path, "FINN_XILINX_PATH=/opt/Xilinx\nFINN_XILINX_VERSION=2025.2\n")
    values = host_file.settings({"FINN_XILINX_ENV": path, "FINN_XILINX_VERSION": "2026.1"})
    assert values == {"FINN_XILINX_PATH": "/opt/Xilinx", "FINN_XILINX_VERSION": "2026.1"}


@pytest.mark.parametrize("licensed", (True, False))
def test_the_host_files_licence_reaches_activation(tmp_path, licensed):
    """scripts/activate.sh evals `inspect --tier build --format sh`: a licence
    server named in the host file arrives as XILINXD_LICENSE_FILE (FlexLM's
    port@host), and without one the variable is absent and nothing is said."""
    root = _make_tree(str(tmp_path / "Xilinx"), "new", "2025.1")
    lines = ["FINN_XILINX_PATH=%s" % root, "FINN_XILINX_VERSION=2025.1"]
    if licensed:
        lines += ["FINN_LICENSE_HOST=licsrv.example", "FINN_LICENSE_PORT=2100"]
    path = _host_file(tmp_path, "\n".join(lines) + "\n")
    proc = subprocess.run(
        [FINN_ENV, "inspect", "--tier", "build", "--format", "sh"],
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"], "HOME": str(tmp_path), "FINN_XILINX_ENV": path},
    )
    assert proc.returncode == 0, proc.stderr
    licence = [line for line in proc.stdout.splitlines() if "LICENSE" in line]
    assert licence == (["XILINXD_LICENSE_FILE='2100@licsrv.example'"] if licensed else [])
    assert "licen" not in proc.stderr.lower()


def test_default_host_file_location(tmp_path):
    config = tmp_path / "config" / "finn"
    config.mkdir(parents=True)
    (config / "xilinx.env").write_text("FINN_XILINX_VERSION=2025.2\n")
    assert host_file.settings({"XDG_CONFIG_HOME": str(tmp_path / "config")}) == {
        "FINN_XILINX_VERSION": "2025.2"
    }
    home = {"HOME": str(tmp_path / "nobody")}
    assert host_file.file_path(home) == str(tmp_path / "nobody/.config/finn/xilinx.env")
    assert host_file.settings(home) == {}  # no file: nothing configured


def test_a_named_host_file_must_exist(tmp_path):
    with pytest.raises(host_file.ConfigError, match="does not exist"):
        host_file.settings({"FINN_XILINX_ENV": str(tmp_path / "missing.env")})
    assert host_file.settings({"FINN_XILINX_ENV": ""}) == {}


def test_build_tier_from_the_host_file_alone(tmp_path):
    """`docker/run --fpga` with nothing exported: the file supplies the toolchain
    and the licence, composed into FlexLM's PORT@HOST."""
    root = _make_tree(str(tmp_path / "Xilinx"), "new", "2025.2")
    _make_tree(root, "new", "2026.1")
    path = _host_file(
        tmp_path,
        "FINN_XILINX_PATH=%s\nFINN_XILINX_VERSION=2025.2\n"
        "FINN_LICENSE_HOST=192.0.2.1\nFINN_LICENSE_PORT=2100\nFINN_LICENSE_VENDOR_PORT=2101\n"
        % root,
    )
    data = json.loads(_inspect({"FINN_XILINX_ENV": path}, "build").stdout)
    assert data["env"]["XILINX_VIVADO"] == os.path.join(root, "2025.2", "Vivado")
    assert data["env"]["XILINXD_LICENSE_FILE"] == "2100@192.0.2.1"
    assert data["egress"][0]["ports"] == ["2100", "2101"]
    # One container selects another installed version.
    other = json.loads(
        _inspect({"FINN_XILINX_ENV": path, "FINN_XILINX_VERSION": "2026.1"}, "build").stdout
    )
    assert other["env"]["XILINX_VIVADO"] == os.path.join(root, "2026.1", "Vivado")


def test_an_explicit_licence_variable_wins_over_host_and_port(tmp_path):
    root = _make_tree(str(tmp_path / "Xilinx"), "new", "2025.2")
    path = _host_file(
        tmp_path,
        "FINN_XILINX_PATH=%s\nFINN_XILINX_VERSION=2025.2\n"
        "FINN_LICENSE_HOST=192.0.2.1\nFINN_LICENSE_PORT=2100\n" % root,
    )
    data = json.loads(
        _inspect(
            {"FINN_XILINX_ENV": path, "XILINXD_LICENSE_FILE": "27000@lic.example"}, "build"
        ).stdout
    )
    assert data["env"]["XILINXD_LICENSE_FILE"] == "27000@lic.example"


def test_dev_tier_ignores_the_host_file(tmp_path):
    root = _make_tree(str(tmp_path / "Xilinx"), "new", "2025.2")
    path = _host_file(
        tmp_path,
        "FINN_XILINX_PATH=%s\nFINN_XILINX_VERSION=2025.2\n"
        "FINN_LICENSE_HOST=192.0.2.1\nFINN_LICENSE_PORT=2100\n" % root,
    )
    data = json.loads(_inspect({"FINN_XILINX_ENV": path}, "dev").stdout)
    assert data["mounts"] == [] and data["egress"] == []
    assert not any("XILINX" in key for key in data["env"])


def test_a_malformed_host_file_is_an_error(tmp_path):
    path = _host_file(tmp_path, "XILINX_VIVADO=/opt/Xilinx\n")
    proc = _inspect({"FINN_XILINX_ENV": path}, "dev")
    assert proc.returncode == 3
    assert "not a setting" in proc.stderr


def test_installation_cli_prints_exports(tmp_path):
    """The standalone installation CLI prints shell exports."""
    root = _make_tree(str(tmp_path / "Xilinx"), "old", "2024.2")
    env = {
        "PATH": os.environ["PATH"],
        "FINN_XILINX_ENV": "",
        "FINN_XILINX_PATH": root,
        "FINN_XILINX_VERSION": "2024.2",
    }
    script = os.path.join(DOCKER_DIR, "xilinx_install.py")
    proc = subprocess.run([sys.executable, script, "sh"], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "export XILINX_VIVADO='%s'" % os.path.join(root, "Vivado", "2024.2") in proc.stdout
    configured = subprocess.run([sys.executable, script, "configured"], env=env)
    assert configured.returncode == 0
    del env["FINN_XILINX_PATH"]
    assert subprocess.run([sys.executable, script, "configured"], env=env).returncode == 1


# --------------------------------------------------------------------------
# Resource overrides reach the container.
# --------------------------------------------------------------------------


def test_resource_overrides_are_mounted_and_passed(tmp_path):
    user_lib = tmp_path / "user_lib"
    user_lib.mkdir()
    cache = tmp_path / "resources"
    cache.mkdir()
    decl = tmp_path / "decl"
    decl.mkdir()
    (decl / "extra.toml").write_text("")
    retired = tmp_path / "retired"
    retired.mkdir()
    data = json.loads(
        _inspect(
            {
                "FINN_RESOURCES_USER_LIB": str(user_lib),
                "FINN_RESOURCES_HLSLIB": str(user_lib),
                "FINN_HLSLIB_PATH": str(retired),
                "FINN_RESOURCES_DIR": str(cache),
                "FINN_RESOURCES_FILES": str(decl / "extra.toml"),
                "FINN_RESOURCES_OFFLINE": "1",
                "FINN_RESOURCES_HLSLIB_URL": "https://git.example/hlslib.git",
                "FINN_RESOURCES_SYSTEM_CACHE": str(cache),
            },
            "dev",
        ).stdout
    )
    mounts = {m["source"]: m for m in data["mounts"]}
    assert mounts[str(user_lib)]["mode"] == "rw" and mounts[str(user_lib)]["target"] == str(
        user_lib
    )
    assert mounts[str(cache)]["mode"] == "rw"
    assert mounts[str(decl)]["mode"] == "ro"
    assert len(data["mounts"]) == 3  # user_lib once, though two variables name it
    assert data["env"]["FINN_RESOURCES_USER_LIB"] == str(user_lib)
    assert data["env"]["FINN_RESOURCES_OFFLINE"] == "1"
    assert data["env"]["FINN_RESOURCES_HLSLIB_URL"] == "https://git.example/hlslib.git"
    assert "FINN_RESOURCES_SYSTEM_CACHE" not in data["env"]  # the image's own
    assert "FINN_HLSLIB_PATH" not in data["env"]  # a retired name selects nothing


def test_relative_resource_override_becomes_absolute(tmp_path):
    (tmp_path / "user_lib").mkdir()
    proc = subprocess.run(
        [FINN_ENV, "inspect", "--tier", "dev"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={
            "PATH": os.environ["PATH"],
            "FINN_XILINX_ENV": "",
            "FINN_RESOURCES_USER_LIB": "user_lib",
        },
    )
    data = json.loads(proc.stdout)
    assert data["env"]["FINN_RESOURCES_USER_LIB"] == str(tmp_path / "user_lib")


# --------------------------------------------------------------------------
# The Dev Container's host inputs.
# --------------------------------------------------------------------------


def test_dev_container_inputs_carry_the_toolchain_but_not_the_workspace(tmp_path):
    root = _make_tree(str(tmp_path / "Xilinx"), "new", "2025.2")
    path = _host_file(
        tmp_path,
        "FINN_XILINX_PATH=%s\nFINN_XILINX_VERSION=2025.2\n"
        "FINN_LICENSE_HOST=192.0.2.1\nFINN_LICENSE_PORT=2100\n" % root,
    )
    proc = subprocess.run(
        [FINN_ENV, "compose", "--tier", "auto", "--inputs-only", "--service", "dev"],
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"], "HOME": str(tmp_path), "FINN_XILINX_ENV": path},
    )
    assert proc.returncode == 0, proc.stderr
    service = json.loads(proc.stdout)["services"]["dev"]
    assert [v["target"] for v in service["volumes"]] == [root]
    assert service["volumes"][0]["read_only"] is True
    assert service["environment"]["XILINX_VIVADO"] == os.path.join(root, "2025.2", "Vivado")
    assert service["environment"]["XILINXD_LICENSE_FILE"] == "2100@192.0.2.1"
    for owned in ("FINN_ROOT", "FINN_BUILD_DIR"):  # the Dev Container's own
        assert owned not in service["environment"]
    assert set(service) == {"environment", "volumes"}


def test_dev_container_inputs_without_a_toolchain(tmp_path):
    proc = subprocess.run(
        [FINN_ENV, "compose", "--tier", "auto", "--inputs-only", "--service", "dev"],
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"], "HOME": str(tmp_path), "FINN_XILINX_ENV": ""},
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["services"]["dev"]["volumes"] == []


def test_environment_identity_ignores_checkout_umask_but_tracks_executable_bit(tmp_path):
    root = _make_image_input_fixture(tmp_path)
    source = root / "image.txt"
    source.chmod(0o644)
    original = _provenance(root)[0]
    source.chmod(0o660)
    assert _provenance(root)[0] == original
    source.chmod(0o755)
    assert _provenance(root)[0] != original


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read mode-000 fixtures")
def test_unreadable_image_input_fails_identity_preparation(tmp_path):
    root = _make_image_input_fixture(tmp_path)
    source = root / "image.txt"
    source.chmod(0)
    try:
        result = subprocess.run(
            ["bash", "-c", ". ./docker/lib.sh; finn_image_revision"],
            cwd=REPO,
            env={"PATH": os.environ["PATH"], "FINN_IMAGE_INPUT_ROOT": str(root)},
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert not result.stdout.strip()
    finally:
        source.chmod(0o644)
