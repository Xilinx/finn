############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ##########################################################################

"""
FINN: A Framework for Fast, Scalable Quantized Neural Network Inference

This package provides tools for building and deploying quantized neural networks
on FPGAs and other accelerators.
"""

import importlib.util
import os
import warnings
from pathlib import Path


def _setup_environment():
    """Configure FINN environment variables on import."""

    # 1. Determine FINN_ROOT (auto-detect if not set)
    finn_root = os.environ.get("FINN_ROOT")
    if not finn_root:
        try:
            finn_spec = importlib.util.find_spec("finn")
            if finn_spec and finn_spec.origin:
                finn_init_path = Path(finn_spec.origin).resolve()
                finn_root = str(finn_init_path.parent.parent.parent)
                os.environ["FINN_ROOT"] = finn_root
            else:
                raise RuntimeError("Could not find finn module spec")
        except Exception as e:
            warnings.warn(
                f"FINN_ROOT environment variable is not set and could not be inferred: {e}\n"
                "This may cause issues with certain FINN operations. "
                "Please set FINN_ROOT to the root directory of your FINN installation."
            )
            return

    # 2. Set FINN_DEPS_DIR (default to {FINN_ROOT}/deps if not set)
    if not os.environ.get("FINN_DEPS_DIR"):
        default_deps_dir = Path(finn_root) / "deps"
        os.environ["FINN_DEPS_DIR"] = str(default_deps_dir)
        if not default_deps_dir.exists():
            warnings.warn(
                f"FINN_DEPS_DIR set to {default_deps_dir}, but directory does not exist yet. "
                "Dependencies will need to be fetched before some operations can work. "
                "Run ./fetch-repos.sh or use the Docker container for full functionality."
            )

    # 3. Configure LD_LIBRARY_PATH for Xilinx tools
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    paths_to_add = []

    # Vivado libraries
    if vivado_path := os.environ.get("XILINX_VIVADO"):
        if (vivado_lib := Path(vivado_path) / "lib" / "lnx64.o").exists():
            paths_to_add.append(str(vivado_lib))
        if (system_lib := Path("/lib/x86_64-linux-gnu")).exists():
            paths_to_add.append(str(system_lib))

    # Vitis FPO libraries
    if vitis_path := os.environ.get("VITIS_PATH"):
        if (vitis_fpo := Path(vitis_path) / "lnx64" / "tools" / "fpo_v7_1").exists():
            paths_to_add.append(str(vitis_fpo))

    # Update LD_LIBRARY_PATH
    if paths_to_add:
        existing_paths = ld_library_path.split(":") if ld_library_path else []
        for path in paths_to_add:
            if path not in existing_paths:
                existing_paths.append(path)
        os.environ["LD_LIBRARY_PATH"] = ":".join(existing_paths)


# Configure environment on import
_setup_environment()

# Version information
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
