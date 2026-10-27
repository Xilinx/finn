# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Locate upstream XSI sources and writable, installation-specific binaries."""
import hashlib
import json
import os
import sysconfig
from pathlib import Path

from finn import resources


def source_dir() -> Path:
    """Return the sources embedded in a release or the selected checkout."""
    return Path(os.environ["FINN_ROOT"]) / "finn_xsi"


def build_dir() -> Path:
    """Separate Python ABIs, source changes and selected Vivado installations."""
    source = source_dir()
    vivado = Path(os.environ.get("XILINX_VIVADO", "/missing-vivado")).resolve()
    inputs = {"vivado": str(vivado), "abi": sysconfig.get_config_var("SOABI")}
    for path in [*source.glob("*.cpp"), *source.glob("*.hpp"), vivado / "data/xsim/include/xsi.h"]:
        if path.is_file():
            inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:20]
    return resources.home() / "xsi" / key
