# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import logging

from finn.core.rtlsim_exec import _debug_stage


def test_debug_stage_uses_logging(caplog, capsys):
    with caplog.at_level(logging.DEBUG, logger="finn.core.rtlsim_exec"):
        _debug_stage("preparing IO")

    assert "rtlsim_exec: preparing IO" in caplog.messages
    assert capsys.readouterr().out == ""
