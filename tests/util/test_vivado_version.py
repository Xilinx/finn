# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from finn.util.basic import require_vivado_version
from finn.util.launcher import ToolUnavailable


@pytest.mark.util
def test_require_vivado_version(monkeypatch):
    monkeypatch.setenv("XILINX_VIVADO", "/tools/Xilinx/2024.2/Vivado")
    assert require_vivado_version() == (2024, 2)
    monkeypatch.setenv("XILINX_VIVADO", "/opt/Xilinx/Vivado/current")
    with pytest.raises(ToolUnavailable, match="does not name the Vivado release"):
        require_vivado_version()
    monkeypatch.delenv("XILINX_VIVADO")
    with pytest.raises(ToolUnavailable, match="XILINX_VIVADO is not set"):
        require_vivado_version()
