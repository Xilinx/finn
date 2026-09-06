############################################################################
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

import pytest

import os
import re
from pathlib import Path

import finn.util.basic as basic
from finn.util.basic import make_build_dir, robust_rmtree

# representative tool used for the resolution-order scenarios below.
SAMPLE_TOOL = "vivado"


def _make_executable(path):
    """Create a runnable stand-in for a Xilinx tool at ``path``."""
    path = Path(path)
    path.write_text("#!/bin/bash\n")
    path.chmod(0o755)


@pytest.fixture
def tool_dir():
    build_dir = make_build_dir(prefix="test_resolve_xilinx_tool_")
    try:
        yield build_dir
    finally:
        robust_rmtree(build_dir)


@pytest.mark.util
def test_resolve_no_override_returns_bare_name(tool_dir, monkeypatch):
    monkeypatch.delenv(basic._XILINX_TOOL_DIR_ENV, raising=False)
    _make_executable(os.path.join(tool_dir, SAMPLE_TOOL))
    monkeypatch.setenv("PATH", tool_dir + os.pathsep + os.environ.get("PATH", ""))
    assert basic.resolve_xilinx_tool(SAMPLE_TOOL) == SAMPLE_TOOL


@pytest.mark.util
def test_resolve_dir_override_joined_with_tool_name(tool_dir, monkeypatch):
    tool_path = os.path.join(tool_dir, SAMPLE_TOOL)
    _make_executable(tool_path)
    monkeypatch.setenv(basic._XILINX_TOOL_DIR_ENV, tool_dir)
    assert basic.resolve_xilinx_tool(SAMPLE_TOOL) == tool_path


@pytest.mark.util
def test_resolve_override_missing_raises_with_override_in_message(tool_dir, monkeypatch):
    # override points at a directory with no matching shim, so resolution fails.
    monkeypatch.setenv(basic._XILINX_TOOL_DIR_ENV, tool_dir)
    with pytest.raises(FileNotFoundError, match=re.escape(basic._XILINX_TOOL_DIR_ENV)):
        basic.resolve_xilinx_tool(SAMPLE_TOOL)
