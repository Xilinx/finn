# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Locate upstream XSI sources and writable, installation-specific binaries."""
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sysconfig
from functools import lru_cache
from pathlib import Path

from finn import resources
from finn.xsi._build_config import LINK_ARGS, get_build_paths


@lru_cache(maxsize=32)
def _compiler_version(path, modified, size):
    try:
        return subprocess.check_output([path, "--version"], text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def source_dir() -> Path:
    """Return the sources embedded in a release or the selected checkout."""
    return Path(os.environ["FINN_ROOT"]) / "finn_xsi"


def build_dir() -> Path:
    """Separate Python ABIs, source changes and selected Vivado installations."""
    source = source_dir()
    vivado = Path(os.environ.get("XILINX_VIVADO", "/missing-vivado")).resolve()
    includes, compiler, flags = get_build_paths()
    # pybind11's headers are identified by its version below. Their location is
    # the venv's, which differs between otherwise identical checkouts.
    try:
        import pybind11  # noqa: PLC0415

        venv_includes = {pybind11.get_include(), pybind11.get_include(user=True)}
    except ImportError:
        venv_includes = set()
    includes = [path for path in includes if path not in venv_includes]
    executable = shutil.which(compiler)
    compiler_info = {"command": compiler}
    if executable:
        executable = Path(executable).resolve()
        stat = executable.stat()
        compiler_info.update(
            path=str(executable),
            version=_compiler_version(str(executable), stat.st_mtime_ns, stat.st_size),
        )
    try:
        pybind11_version = importlib.metadata.version("pybind11")
    except importlib.metadata.PackageNotFoundError:
        pybind11_version = "missing"
    inputs = {
        "vivado": str(vivado),
        "abi": sysconfig.get_config_var("SOABI"),
        "includes": includes,
        "compiler": compiler_info,
        "flags": flags,
        "link_flags": LINK_ARGS,
        "pybind11": pybind11_version,
    }
    # Sources by name and content, so copies of a checkout share one build.
    for name, path in [
        *(("finn_xsi/" + path.name, path) for path in source.glob("*.[ch]pp")),
        ("xsi.h", vivado / "data/xsim/include/xsi.h"),
    ]:
        if path.is_file():
            inputs[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:20]
    return root() / key


def root() -> Path:
    """Where finn_xsi builds are kept: FINN_XSI_PATH, by default ``$FINN_HOME/xsi``."""
    value = os.environ.get("FINN_XSI_PATH")
    return Path(value).expanduser() if value else resources.home() / "xsi"
