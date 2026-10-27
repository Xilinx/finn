# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Build settings shared by XSI compilation and cache identity."""
import os
import shutil
import sysconfig
from typing import List, Tuple

LINK_ARGS = ["-ldl", "-lrt"]


def get_build_paths() -> Tuple[List[str], str, List[str]]:
    """Get include paths and compiler for building the extension.

    Returns:
        Tuple of (include_dirs, compiler, extra_compile_args)
    """
    include_dirs = []

    # Get Python include directory
    python_include = sysconfig.get_path("include")
    if python_include:
        include_dirs.append(python_include)

    # Try to get pybind11 include directory
    try:
        import pybind11  # noqa: PLC0415

        pybind11_include = pybind11.get_include()
        include_dirs.append(pybind11_include)

        # Also get the user-specific include
        pybind11_user_include = pybind11.get_include(user=True)
        if pybind11_user_include != pybind11_include:
            include_dirs.append(pybind11_user_include)
    except ImportError:
        # Will be caught later in prerequisites check
        pass

    # Get Xilinx Vivado include directory
    xilinx_vivado = os.environ.get("XILINX_VIVADO")
    if xilinx_vivado:
        xilinx_include = os.path.join(xilinx_vivado, "data", "xsim", "include")
        if os.path.exists(xilinx_include):
            include_dirs.append(xilinx_include)

    # Determine compiler
    compiler = "g++"
    if not shutil.which("g++") and shutil.which("clang++"):
        compiler = "clang++"

    # Compile flags
    extra_compile_args = ["--std=c++17", "-Wall", "-O3", "-shared", "-fPIC"]

    return include_dirs, compiler, extra_compile_args
