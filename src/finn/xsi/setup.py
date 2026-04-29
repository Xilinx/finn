############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ##########################################################################
"""Setup script for FINN XSI (RTL simulation) support.

This script builds and configures the finn_xsi C++ extension module
required for RTL simulation in FINN.

Usage:
    python -m finn.xsi.setup [options]

Options:
    --force    Force rebuild even if already built
    --clean    Clean build artifacts
    --check    Only check if build is needed
"""

import argparse
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import List, Tuple


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
    # flake8: noqa
    try:
        import pybind11

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


def check_prerequisites() -> List[str]:
    """Check if required tools are available."""
    errors = []

    # Check for C++ compiler
    if not shutil.which("g++") and not shutil.which("clang++"):
        errors.append("No C++ compiler found. Please install g++ or clang++.")

    # Check for Xilinx tools
    if not shutil.which("vivado"):
        errors.append("'vivado' not found. Ensure Xilinx tools are in PATH.")

    # Check Xilinx Vivado environment variable
    xilinx_vivado = os.environ.get("XILINX_VIVADO")
    if not xilinx_vivado:
        errors.append("XILINX_VIVADO environment variable not set. Please source Vivado settings.")
    elif not os.path.exists(os.path.join(xilinx_vivado, "data", "xsim", "include")):
        errors.append(f"Xilinx XSim headers not found at {xilinx_vivado}/data/xsim/include")

    # Check for pybind11
    try:
        import pybind11
    except ImportError:
        errors.append("pybind11 not found. Please install it with: pip install pybind11")

    return errors


def build_xsi(force: bool = False, verbose: bool = True) -> bool:
    """Build the finn_xsi extension module using direct g++ compilation.

    Args:
        force: Force rebuild even if already built
        verbose: Print build output

    Returns:
        bool: True if build successful
    """
    finn_root = Path(os.environ["FINN_ROOT"])
    xsi_path = finn_root / "finn_xsi"

    if not xsi_path.exists():
        print(f"Error: finn_xsi source not found at {xsi_path}")
        return False

    # Check if already built
    if not force:
        xsi_so = xsi_path / "xsi.so"
        if xsi_so.exists():
            # Try importing to see if it works
            sys.path.insert(0, str(xsi_path))
            try:
                import xsi

                sys.path.pop(0)
                if verbose:
                    print("xsi.so is already built and working.")
                return True
            except ImportError:
                sys.path.pop(0)
                if verbose:
                    print("xsi.so exists but failed to import, rebuilding...")
        # else: Need to build

    if verbose:
        print(f"Building finn_xsi in {xsi_path}...")

    # Get build configuration
    include_dirs, compiler, compile_args = get_build_paths()

    # Source files
    source_files = ["xsi_bind.cpp", "xsi_finn.cpp"]

    # Build command
    cmd = [compiler] + compile_args

    # Add include directories
    for inc_dir in include_dirs:
        cmd.extend(["-I", inc_dir])

    # Output file
    cmd.extend(["-o", "xsi.so"])

    # Source files
    cmd.extend(source_files)

    # Link libraries
    cmd.extend(["-ldl", "-lrt"])

    if verbose:
        print(f"Build command: {' '.join(cmd)}")

    # Run the compilation
    result = subprocess.run(cmd, cwd=xsi_path, capture_output=True, text=True)

    if result.returncode != 0:
        print("Build failed!")
        if result.stderr:
            print("Error output:", result.stderr)
        if result.stdout:
            print("Build output:", result.stdout)
        print("\nBuild command was:")
        print(" ".join(cmd))
        print("\nCommon issues:")
        print("  - Ensure Xilinx Vivado is properly sourced")
        print("  - Check that pybind11 is installed in your Python environment")
        print("  - Verify C++ compiler is installed")
        return False

    if verbose and result.stdout:
        print(result.stdout)

    if verbose:
        print("Build completed successfully.")
    return True


def verify_installation() -> bool:
    """Verify that finn_xsi can be imported and works."""
    finn_root = Path(os.environ["FINN_ROOT"])
    xsi_path = finn_root / "finn_xsi"

    # Check if xsi.so exists
    xsi_so = xsi_path / "xsi.so"
    if not xsi_so.exists():
        print(f"\n✗ Compiled extension xsi.so not found at {xsi_so}")
        return False

    # Temporarily add to path
    sys.path.insert(0, str(xsi_path))

    try:
        # Import the compiled C++ extension
        import xsi

        print("\n✓ xsi C++ extension module imports successfully")

        # Import the Python package
        import finn_xsi.adapter

        print("✓ finn_xsi.adapter imports successfully")

        # Check for basic functionality
        if hasattr(finn_xsi.adapter, "rtlsim_multi_io"):
            print("✓ RTL simulation functions available")

        return True

    except ImportError as e:
        print(f"\n✗ Failed to import modules: {e}")
        return False
    finally:
        sys.path.pop(0)


def clean_build() -> bool:
    """Clean build artifacts."""
    finn_root = Path(os.environ["FINN_ROOT"])
    xsi_path = finn_root / "finn_xsi"

    print(f"Cleaning build artifacts in {xsi_path}...")

    # Remove xsi.so if it exists
    xsi_so = xsi_path / "xsi.so"
    if xsi_so.exists():
        try:
            xsi_so.unlink()
            print("Removed xsi.so")
            return True
        except Exception as e:
            print(f"Failed to remove xsi.so: {e}")
            return False
    else:
        print("No artifacts to clean.")
        return True


def main() -> int:
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup FINN XSI (RTL simulation) support")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if already built")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts and exit")
    parser.add_argument("--check", action="store_true", help="Only check prerequisites")
    parser.add_argument("--quiet", action="store_true", help="Suppress build output")

    args = parser.parse_args()

    # Clean and exit if requested
    if args.clean:
        if clean_build():
            print("Clean completed successfully.")
            return 0
        else:
            print("Clean failed.")
            return 1

    # Check prerequisites
    if not args.quiet:
        print("Checking prerequisites...")
    errors = check_prerequisites()

    if errors:
        print("Prerequisite check failed:")
        for error in errors:
            print(f"  ✗ {error}")
        print("Please resolve these issues and try again.")
        return 1

    if not args.quiet:
        print("✓ All prerequisites satisfied")

    if args.check:
        return 0

    # Build finn_xsi
    if not args.quiet:
        print("Building finn_xsi extension...")
    if not build_xsi(force=args.force, verbose=not args.quiet):
        print("Build failed. Please check the error messages above.")
        return 1

    # Verify installation
    verification_result = verify_installation() if not args.quiet else True

    if verification_result:
        if not args.quiet:
            print("\nFINN XSI setup completed successfully!")
        return 0
    else:
        print("\nSetup completed but verification failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
