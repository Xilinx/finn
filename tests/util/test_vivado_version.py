# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from finn.util.basic import get_vivado_version, require_vivado_version
from finn.util.launcher import ToolUnavailable


def make_vivado(directory, release=None):
    """Make a Vivado installation tree in ``directory``, with a data/version.sh
    naming ``release``, if given."""
    (directory / "data").mkdir(parents=True)
    if release is not None:
        (directory / "data" / "version.sh").write_text(
            f"XILINX_VERSION_DEFAULT={release}\nXILINX_VERSION_VITIS={release}\n"
        )
    return directory


@pytest.mark.util
def test_vivado_version_from_path_or_version_sh(tmp_path, monkeypatch):
    monkeypatch.setenv("XILINX_VIVADO", "/tools/Xilinx/Vivado/2024.2")
    assert get_vivado_version() == (2024, 2)
    # The path wins over version.sh.
    versioned = make_vivado(tmp_path / "2024.1" / "Vivado", "2025.2")
    monkeypatch.setenv("XILINX_VIVADO", str(versioned))
    assert get_vivado_version() == (2024, 1)
    unversioned = make_vivado(tmp_path / "current" / "Vivado", "2025.2")
    monkeypatch.setenv("XILINX_VIVADO", str(unversioned))
    assert get_vivado_version() == (2025, 2)
    assert require_vivado_version() == (2025, 2)


@pytest.mark.util
def test_unknown_vivado_version(tmp_path, monkeypatch):
    monkeypatch.delenv("XILINX_VIVADO", raising=False)
    assert get_vivado_version() is None
    with pytest.raises(ToolUnavailable, match="XILINX_VIVADO is not set"):
        require_vivado_version()
    unversioned = make_vivado(tmp_path / "current" / "Vivado")
    monkeypatch.setenv("XILINX_VIVADO", str(unversioned))
    assert get_vivado_version() is None
    with pytest.raises(ToolUnavailable, match="cannot read the Vivado release") as unknown:
        require_vivado_version()
    assert str(unversioned / "data" / "version.sh") in str(unknown.value)
