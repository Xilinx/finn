# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the SLASH linker invocation used by the SLASH_ALVEO shell flow."""

import pytest

from pathlib import Path

import finn.transformation.fpgadataflow.alveo_build as alveo_build
from finn.transformation.fpgadataflow.alveo_build import (
    _slash_link_argv,
    _slash_link_command,
)


@pytest.mark.util
def test_slash_link_argv_resolves_slashkit_via_shim(monkeypatch):
    shim_path = "/ci/shims/slashkit"
    resolved = []

    def fake_resolve(tool_name):
        resolved.append(tool_name)
        return shim_path

    monkeypatch.setattr(alveo_build, "resolve_xilinx_tool", fake_resolve)
    cmd = _slash_link_argv(Path("config.cfg"), Path("finn.vbin"), [], True)
    assert resolved == ["slashkit"]
    assert cmd[0] == shim_path


@pytest.mark.util
def test_slash_link_command_hw_platform_and_flags():
    config = Path("config.cfg")
    vbin = Path("finn.vbin")
    component = Path("ip/component.xml")
    cmd = _slash_link_command("slashkit", config, vbin, [component], True)
    assert cmd[0] == "slashkit"
    assert cmd[1] == "link"
    assert cmd[cmd.index("--config") + 1] == str(config)
    assert cmd[cmd.index("--platform") + 1] == "hw"
    assert cmd[cmd.index("--out") + 1] == str(vbin)
    assert cmd[cmd.index("--kernels") + 1 :] == [str(component)]


@pytest.mark.util
def test_slash_link_command_simulation_toggles_platform():
    cmd = _slash_link_command("slashkit", "config.cfg", "finn.vbin", [], False)
    assert cmd[cmd.index("--platform") + 1] == "sim"


@pytest.mark.util
def test_slash_link_command_appends_all_kernels():
    kernels = [Path("k0/component.xml"), Path("k1/component.xml")]
    cmd = _slash_link_command("slashkit", "config.cfg", "finn.vbin", kernels, True)
    assert cmd[cmd.index("--kernels") + 1 :] == [str(k) for k in kernels]
