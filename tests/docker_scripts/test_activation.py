# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Tool activation shared by native use and containers: finn-toolchain.sh,
scripts/activate.sh, container identity and the installers."""
import pytest

import glob
import json
import os
import re
import shutil
import subprocess
import sys
from packaging.requirements import Requirement
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOOLCHAIN_SH = str(REPO / "docker/finn-toolchain.sh")
sys.path.insert(0, str(REPO / "docker"))

pytestmark = pytest.mark.util


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.startswith(("FINN_", "XILINX_")) or name in (
            "BASH_ENV",
            "ENV",
            "SLASHKIT_DEB_PACKAGE",
            "PLATFORM_REPO_PATHS",
            "HLS_PATH",
            "VIVADO_PATH",
            "VITIS_PATH",
            "LD_PRELOAD",
            "XILINXD_LICENSE_FILE",
            "LM_LICENSE_FILE",
        ):
            monkeypatch.delenv(name)
    monkeypatch.setenv("FINN_XILINX_ENV", "")
    monkeypatch.setenv("FINN_ROOT", str(REPO))
    monkeypatch.setenv("FINN_HOST_BUILD_DIR", str(tmp_path / "build"))
    monkeypatch.setenv("FINN_HOME", str(tmp_path / "state"))


# --------------------------------------------------------------------------
# finn-toolchain.sh: sourced natively, by the entrypoint, BASH_ENV and shims.
# --------------------------------------------------------------------------


def toolchain_env(env):
    """os.environ minus everything finn-toolchain.sh reads, plus `env`.

    These tests also run inside the container, where XILINX_VIVADO is set and a
    real settings64.sh would put Vivado's bin on PATH. BASH_ENV and ENV go too:
    bash reads them even with --noprofile --norc.
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


def apply_toolchain(env, probe='printf "%s" "$PATH"'):
    proc = subprocess.run(
        ["/bin/bash", "--noprofile", "--norc", "-c", ". %s; %s" % (TOOLCHAIN_SH, probe)],
        capture_output=True,
        text=True,
        env=toolchain_env(env),
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_toolchain_dedupes_path_variables():
    """settings64.sh prepends unconditionally; nested shells must not grow PATH."""
    assert apply_toolchain({"PATH": "/a:/b:/a:/c:/b"}) == "/a:/b:/c"
    assert apply_toolchain({"PATH": "/a::/b:"}) == "/a:/b"
    assert (
        apply_toolchain({"PATH": "/a", "LD_PRELOAD": "/x:/x"}, 'printf "%s" "$LD_PRELOAD"') == "/x"
    )


@pytest.mark.skipif(
    not glob.glob("/lib/*-linux-gnu/libudev.so.1"), reason="no system libudev to preload"
)
def test_toolchain_preloads_libudev_on_merged_usr_hosts():
    """/lib is a symlink on merged-/usr hosts; discovery must follow it."""
    preload = apply_toolchain({}, 'printf "%s" "$LD_PRELOAD"')
    assert preload.endswith("-linux-gnu/libudev.so.1")


def test_toolchain_is_idempotent_and_silent():
    """BASH_ENV sources this before every `bash -c`, so a stray echo corrupts
    the output of every command in the image."""
    assert apply_toolchain({"PATH": "/a"}, probe="true") == ""
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
        env=toolchain_env({"PATH": "/a:/b"}),
    )
    assert proc.stdout == "/a:/b|1", proc.stderr


def test_vendor_settings_tolerate_unset_variables_and_keep_selected_python(tmp_path):
    tool = tmp_path / "Vivado"
    tool.mkdir()
    (tool / "settings64.sh").write_text(
        'echo "${SITE_UNSET_SETTING}" >/dev/null\nexport PATH="/vendor/python:$PATH"\n'
    )
    env = toolchain_env({"XILINX_VIVADO": str(tool), "VIRTUAL_ENV": str(tmp_path / "venv")})
    result = subprocess.run(
        ["bash", "-eu", "-c", 'source "$1"; test "${PATH%%:*}" = "$VIRTUAL_ENV/bin"', "probe"]
        + [TOOLCHAIN_SH],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_farm_wrappers_do_not_shadow_bare_tools(clean_env, tmp_path, monkeypatch):
    wrappers = tmp_path / "wrappers"
    wrappers.mkdir()
    for tool in ("make", "vivado"):
        (wrappers / tool).write_text("#!/bin/sh\nexit 99\n")
        (wrappers / tool).chmod(0o755)
    monkeypatch.setenv("FINN_TOOL_DIR_OVERRIDE", str(wrappers))
    result = subprocess.run(
        ["bash", "-c", ". docker/finn-toolchain.sh; command -v make"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() != str(wrappers / "make")
    from finn.util.basic import resolve_xilinx_tool  # noqa: PLC0415

    assert resolve_xilinx_tool("vivado") == str(wrappers / "vivado")


def test_slashkit_has_a_transparent_tool_shim():
    assert re.search(
        r"for t in [^;\n]*\bslashkit\b[^;\n]*; do \\\n"
        r'\s*ln -sf /usr/local/lib/finn/toolchain-shim /usr/local/lib/finn/toolchain-bin/"\$t"',
        (REPO / "docker/Dockerfile.finn").read_text(),
    )


# --------------------------------------------------------------------------
# Native activation.
# --------------------------------------------------------------------------


def make_old_layout(root, version):
    for directory in ("Vivado", "Vitis", "Vitis_HLS"):
        (root / directory / version).mkdir(parents=True)
    return root


def standin_checkout(tmp_path):
    """Activation's own files, with a test-owned virtual environment."""
    checkout = tmp_path / "finn"
    for relative in (
        "scripts/activate.sh",
        "docker/launcher.py",
        "docker/identity.py",
        "docker/finn-toolchain.sh",
    ):
        (checkout / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, checkout / relative)
    (checkout / ".venv/bin").mkdir(parents=True)
    (checkout / ".venv/bin/activate").write_text(": # test-owned virtual environment stand-in\n")
    (tmp_path / "hlslib").mkdir()
    (tmp_path / "boards").mkdir()
    return checkout


def test_activation_resolves_tools_without_allocating(tmp_path):
    """Activation in a stand-in checkout: no installation, no directories."""
    checkout = standin_checkout(tmp_path)
    root = make_old_layout(tmp_path / "Xilinx", "2022.2")
    proc = subprocess.run(
        [
            "bash",
            "-c",
            'source "$FINN_ROOT/scripts/activate.sh"; test "$XILINX_VIVADO" = "$1" '
            '&& test "$FINN_HOME" = "$HOME/.finn" && test ! -e "$FINN_HOME" '
            '&& test "$FINN_BUILD_DIR" = "$2" && test ! -e "$FINN_BUILD_DIR"',
            "test",
            str(root / "Vivado/2022.2"),
            str(tmp_path / "build"),
        ],
        env={
            "PATH": os.environ["PATH"],
            "HOME": str(tmp_path / "home"),
            "FINN_ROOT": str(checkout),
            "FINN_XILINX_ENV": "",
            "FINN_RESOURCES_HLSLIB": str(tmp_path / "hlslib"),
            "FINN_BOARD_FILES_PATH": str(tmp_path / "boards"),
            "FINN_HOST_BUILD_DIR": str(tmp_path / "build"),
            "FINN_XILINX_PATH": str(root),
            "FINN_XILINX_VERSION": "2022.2",
        },
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_reactivation_recomputes_what_activation_computed(tmp_path):
    """Activation's outputs are not read back as overrides: a changed version
    selects the new tools. A value the caller sets in between still wins."""
    checkout = standin_checkout(tmp_path)
    root = make_old_layout(tmp_path / "Xilinx", "2024.2")
    make_old_layout(root, "2025.2")
    proc = subprocess.run(
        [
            "bash",
            "-c",
            'source "$FINN_ROOT/scripts/activate.sh" && echo "$VIVADO_PATH" '
            '&& FINN_XILINX_VERSION=2025.2 && source "$FINN_ROOT/scripts/activate.sh" '
            '&& echo "$VIVADO_PATH" && echo "$XILINX_VIVADO" '
            '&& VITIS_PATH=$1 && source "$FINN_ROOT/scripts/activate.sh" && echo "$XILINX_VITIS"',
            "test",
            str(root / "Vitis/2024.2"),
        ],
        env={
            "PATH": os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"],
            "HOME": str(tmp_path / "home"),
            "FINN_ROOT": str(checkout),
            "FINN_XILINX_ENV": "",
            "FINN_RESOURCES_HLSLIB": str(tmp_path / "hlslib"),
            "FINN_BOARD_FILES_PATH": str(tmp_path / "boards"),
            "FINN_XILINX_PATH": str(root),
            "FINN_XILINX_VERSION": "2024.2",
        },
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == [
        str(root / "Vivado/2024.2"),
        str(root / "Vivado/2025.2"),
        str(root / "Vivado/2025.2"),
        str(root / "Vitis/2024.2"),
    ]


def test_native_hls_only_activation_does_not_fetch_boards(clean_env, tmp_path, monkeypatch):
    root = tmp_path / "tools"
    (root / "Vitis_HLS/2024.2").mkdir(parents=True)
    library = tmp_path / "hlslib"
    library.mkdir()
    monkeypatch.setenv("FINN_XILINX_PATH", str(root))
    monkeypatch.setenv("FINN_XILINX_VERSION", "2024.2")
    monkeypatch.setenv("FINN_RESOURCES_HLSLIB", str(library))
    monkeypatch.setenv("FINN_RESOURCES_OFFLINE", "1")
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source scripts/activate.sh && test -n "$XILINX_HLS" && test -z "${XILINX_VIVADO:-}"',
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not (Path(os.environ["FINN_HOME"]) / "resources").exists()


def test_installers_do_not_hardcode_layouts_or_pins():
    """The pre-2024.2 layout, hardcoded, reported "Vivado not found" at a path
    the user could see; Python pins live only in pyproject.toml and uv.lock."""
    pin = re.compile(r"[A-Za-z0-9_.\-]+(?:\[[A-Za-z0-9_,.\-]+\])?==[0-9]")
    for rel in ("setup-local.sh", "scripts/activate.sh", "docker/Dockerfile.finn"):
        body = (REPO / rel).read_text()
        for number, line in enumerate(body.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            assert "$FINN_XILINX_PATH/Vivado/" not in line, "%s:%d" % (rel, number)
        assert pin.findall(body) == [], rel


def test_nothing_writes_to_the_checkout_by_default():
    """Persistent state belongs in FINN_HOME and scratch in FINN_BUILD_DIR."""
    for rel in ("setup-local.sh", "scripts/activate.sh", "scripts/ensure-uv.sh"):
        for line in (REPO / rel).read_text().splitlines():
            if not line.lstrip().startswith("#"):
                assert ".cache/" not in line, (rel, line)


# --------------------------------------------------------------------------
# Container identity.
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not Path("/usr/lib/x86_64-linux-gnu/libnss_wrapper.so").exists(),
    reason="libnss_wrapper is installed in the container",
)
def test_container_identity_is_visible_to_python_and_openssh(tmp_path):
    import identity  # noqa: PLC0415

    home = tmp_path / "home"
    home.mkdir()
    values = identity.prepare(tmp_path / "nss", os.getuid(), os.getgid(), "finn-test", str(home))
    env = dict(
        os.environ,
        **values,
        HOME=str(home),
        LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libnss_wrapper.so",
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import os,pwd; assert pwd.getpwuid(os.getuid()).pw_name == 'finn-test'",
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["ssh", "-G", "-F", "/dev/null", "user@127.0.0.1"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


# --------------------------------------------------------------------------
# The runtime profile and CI scripts.
# --------------------------------------------------------------------------


def test_builder_imports_without_development_packages(clean_env):
    uv = shutil.which("uv")
    if not uv:
        pytest.skip("uv is needed to export the runtime profile")
    exported = subprocess.check_output(
        [uv, "export", "--locked", "--no-default-groups", "--no-emit-project", "--no-hashes"],
        cwd=REPO,
        text=True,
    )
    allowed = {"finn"}
    for line in exported.splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "-")):
            allowed.add(Requirement(line).name.lower().replace("_", "-"))
    code = """
import importlib.metadata as md
from importlib.machinery import PathFinder
import json, sys
allowed = set(json.loads(sys.argv[1]))
packages = md.packages_distributions()
original = PathFinder.find_spec
def runtime_spec(fullname, path=None, target=None):
    distributions = packages.get(fullname.split('.')[0], [])
    if distributions and not any(n.lower().replace('_','-') in allowed for n in distributions):
        return None
    return original(fullname, path, target)
PathFinder.find_spec = staticmethod(runtime_spec)
assert PathFinder.find_spec('pytest') is None
from finn.builder import build_dataflow
"""
    subprocess.run([sys.executable, "-c", code, json.dumps(sorted(allowed))], cwd=REPO, check=True)


def test_candidate_failure_cannot_publish_previous_results(clean_env, tmp_path, monkeypatch):
    # Use an isolated checkout-shaped cwd so real reports are not removed.
    root = tmp_path / "candidate"
    root.mkdir()
    for name in ("brevitas_suite.xml", "brevitas_suite.html"):
        (root / name).write_text("old report")
    for name in ("brevitas-provenance", "coverage_brevitas_suite"):
        (root / name).mkdir()
        (root / name / "old.json").write_text("old evidence")
    monkeypatch.setenv("FINN_ROOT", str(root))
    monkeypatch.setenv("BREVITAS_COMMIT", "invalid")
    result = subprocess.run(
        ["bash", str(REPO / "ci/scripts/brevitas_candidate.sh")], capture_output=True
    )
    assert result.returncode == 2
    assert list(root.iterdir()) == []
