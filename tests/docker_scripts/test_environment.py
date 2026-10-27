# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Environment integration contracts that do not require licensed tools."""
import pytest

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docker"))
import config  # noqa: E402
import image as images  # noqa: E402
import launch  # noqa: E402

pytestmark = pytest.mark.util


@pytest.fixture
def host(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith("FINN_") or key in ("XILINXD_LICENSE_FILE", "LM_LICENSE_FILE"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("FINN_XILINX_ENV", "")
    monkeypatch.setenv("FINN_ROOT", str(ROOT))
    monkeypatch.setenv("FINN_HOST_BUILD_DIR", str(tmp_path / "build with spaces"))


def test_launcher_preserves_arguments_and_site_flags(host, monkeypatch):
    monkeypatch.setenv("FINN_DOCKER_EXTRA", "--hostname farm-agent --env 'SITE_VALUE=a b'")
    args = argparse.Namespace(
        tier="dev",
        image="sha256:" + "a" * 64,
        volume=["/model with spaces:/model with spaces"],
        env=["VALUE=two words"],
        workdir="/model with spaces",
        name=None,
        notebook=False,
        command=["--", "python", "flow with spaces.py", "x'`$()"],
    )
    argv = launch.command(args)
    assert argv[-3:] == ["python", "flow with spaces.py", "x'`$()"]
    assert "SITE_VALUE=a b" in argv
    assert "/model with spaces:/model with spaces" in argv
    assert argv[argv.index("--hostname") + 1] == "farm-agent"
    assert argv[argv.index("--workdir") + 1] == "/model with spaces"


def test_software_launch_does_not_require_tools_and_keeps_verification_io(
    host, tmp_path, monkeypatch
):
    io = tmp_path / "verification"
    io.mkdir()
    monkeypatch.setenv("VERIFICATION_IO", str(io))
    args = argparse.Namespace(
        tier="dev",
        image="image",
        volume=[],
        env=[],
        workdir=None,
        name=None,
        notebook=False,
        command=["true"],
    )
    argv = launch.command(args)
    assert f"VERIFICATION_IO={io}" in argv
    assert f"{io}:{io}:rw" in argv


def test_broken_prebuilt_does_not_fall_back_to_local_image(host, tmp_path):
    env = dict(
        os.environ, FINN_DOCKER_PREBUILT="1", FINN_DOCKER_SHARED_IMAGE_DIR=str(tmp_path / "missing")
    )
    result = subprocess.run(
        [str(ROOT / "docker/run"), "--", "true"], env=env, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "Incomplete image publication" in result.stderr
    assert "Building" not in result.stderr


@pytest.fixture
def publication(host, tmp_path, monkeypatch):
    monkeypatch.setenv("FINN_IMAGE_REVISION", "img-test")
    monkeypatch.setenv("FINN_SOURCE_REVISION", "a" * 40)
    monkeypatch.setenv("FINN_SOURCE_DIRTY", "0")
    data = {
        "schema": 1,
        "image_id": "sha256:" + "b" * 64,
        "environment_revision": "img-test",
        "lock_sha256": images.sha256(ROOT / "uv.lock"),
        "runtimes": [],
        "source_revision": "a" * 40,
    }
    directory = tmp_path / "publication"
    images.write_metadata(directory, "descriptive-tag", data)
    return directory, data


def test_image_metadata_rejects_corruption_and_wrong_environment(publication):
    directory, data = publication
    (directory / images.ARCHIVE).write_bytes(b"corrupt archive")
    with pytest.raises(ValueError, match="checksum mismatch"):
        images.read_metadata(directory, archive=True)
    data["environment_revision"] = "another-environment"
    images.write_metadata(directory, "descriptive-tag", data)
    with pytest.raises(ValueError, match="Published environment_revision"):
        images.read_metadata(directory)


def test_image_metadata_rejects_legacy_and_mismatched_identity(publication):
    directory, data = publication
    (directory / images.FILES[1]).write_text("sha256:" + "c" * 64)
    with pytest.raises(ValueError, match="identity"):
        images.read_metadata(directory)
    (directory / images.FILES[2]).unlink()
    with pytest.raises(ValueError, match="legacy two-file"):
        images.read_metadata(directory)


def test_xsi_output_is_separate_and_changes_with_tool_or_source(host, tmp_path, monkeypatch):
    from finn.xsi.paths import build_dir  # noqa: PLC0415

    checkout = tmp_path / "source"
    (checkout / "finn_xsi").mkdir(parents=True)
    cpp = checkout / "finn_xsi/xsi_finn.cpp"
    cpp.write_text("// version one")
    monkeypatch.setenv("FINN_ROOT", str(checkout))
    monkeypatch.setenv("FINN_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XILINX_VIVADO", str(tmp_path / "2024.2/Vivado"))
    original = build_dir()
    cpp.write_text("// version two")
    assert build_dir() != original
    changed_source = build_dir()
    monkeypatch.setenv("XILINX_VIVADO", str(tmp_path / "2025.1/Vivado"))
    assert build_dir() != changed_source
    assert not original.exists()
    assert str(original).startswith(str(tmp_path / "state"))


def test_site_state_paths_are_mounted_and_ssh_does_not_own_home(host, tmp_path, monkeypatch):
    board_files = tmp_path / "site-boards"
    board_files.mkdir()
    keys = tmp_path / "keys"
    keys.mkdir()
    monkeypatch.setenv("FINN_BOARD_FILES_PATH", str(board_files))
    monkeypatch.setenv("FINN_SSH_KEY_DIR", str(keys))
    args = argparse.Namespace(
        tier="dev",
        image="image",
        volume=[],
        env=[],
        workdir=None,
        name=None,
        notebook=False,
        command=["true"],
    )
    argv = launch.command(args)
    assert f"{board_files}:{board_files}:rw" in argv
    assert f"FINN_BOARD_FILES_PATH={board_files}" in argv
    assert f"{keys}:/run/finn-ssh:ro" in argv
    assert not any(item.endswith(":/tmp/finn-home/.ssh:ro") for item in argv)


def test_vendor_settings_tolerate_unset_variables_and_keep_selected_python(tmp_path):
    tool = tmp_path / "Vivado"
    tool.mkdir()
    (tool / "settings64.sh").write_text(
        'echo "${SITE_UNSET_SETTING}" >/dev/null\nexport PATH="/vendor/python:$PATH"\n'
    )
    venv = tmp_path / "venv"
    env = dict(os.environ, XILINX_VIVADO=str(tool), VIRTUAL_ENV=str(venv))
    env.pop("FINN_ENV_APPLIED", None)
    result = subprocess.run(
        [
            "bash",
            "-eu",
            "-c",
            'source "$1"; test "${PATH%%:*}" = "$VIRTUAL_ENV/bin"',
            "probe",
            str(ROOT / "docker/finn-toolchain.sh"),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_direct_exec_environment_includes_selected_tool_libraries(host, tmp_path, monkeypatch):
    root = tmp_path / "tools"
    for directory in ("Vivado/2024.2", "Vitis/2024.2", "Vitis_HLS/2024.2"):
        (root / directory).mkdir(parents=True)
    monkeypatch.setenv("FINN_XILINX_PATH", str(root))
    monkeypatch.setenv("FINN_XILINX_VERSION", "2024.2")
    monkeypatch.setenv("FINN_RUNTIMES", "xrt")
    env = config.container_environment(config.resolve_host("build", "mirror"))
    assert env["HLS_PATH"] == str(root / "Vitis_HLS/2024.2")
    assert env["XILINX_XRT"] == "/opt/xilinx/xrt"
    assert str(root / "Vivado/2024.2/lib/lnx64.o") in env["LD_LIBRARY_PATH"].split(":")
    assert "/opt/xilinx/xrt/lib" in env["LD_LIBRARY_PATH"].split(":")


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


def test_launcher_selects_its_checkout_over_a_stale_native_environment(host, monkeypatch):
    monkeypatch.setenv("FINN_ROOT", "/a/different/checkout")
    result = subprocess.run(
        [str(ROOT / "docker/run"), "--print", "--", "true"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "workspace=" + str(ROOT) in result.stdout


def test_release_source_omits_supplied_runtime_archives(tmp_path):
    source = tmp_path / "source"
    packages = source / "docker/packages"
    packages.mkdir(parents=True)
    (packages / "private.deb").write_text("supplied private runtime")
    (packages / "private.rpm").write_text("supplied private runtime")
    (packages / "README.md").write_text("package instructions")
    (source / "compiler.py").write_text("compiler source")
    (source / "compiler.py").chmod(0o600)
    (source / ".gitignore").write_text("generated.onnx\n")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "-f", "."], check=True)
    (source / "generated.onnx").write_text("ignored model download")
    (source / "untracked.txt").write_text("not part of the source snapshot")
    (source / "compiler.py").write_text("edited tracked source")
    destination = tmp_path / "prepared"
    archive = tmp_path / "source.tar"
    subprocess.run(
        [sys.executable, str(ROOT / "docker/source.py"), str(source), str(archive)], check=True
    )
    subprocess.run(
        ["sh", str(ROOT / "docker/copy-source.sh"), str(archive), str(destination)], check=True
    )
    assert (destination / "compiler.py").read_text() == "edited tracked source"
    assert (destination / "compiler.py").stat().st_mode & 0o004
    assert (destination / "docker/packages/README.md").is_file()
    assert not list(destination.rglob("*.deb"))
    assert not list(destination.rglob("*.rpm"))
    assert not (destination / "generated.onnx").exists()
    assert not (destination / "untracked.txt").exists()
