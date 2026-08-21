# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from finn.util.basic import (
    get_liveness_threshold_cycles,
    get_rtlsim_timeout_error_message,
    get_watchdog_timeout_cycles,
)


@pytest.mark.util
def test_liveness_threshold_default_and_estimate(monkeypatch):
    monkeypatch.delenv("LIVENESS_THRESHOLD", raising=False)

    assert get_liveness_threshold_cycles() == 10000
    assert get_watchdog_timeout_cycles() == 10000
    assert get_watchdog_timeout_cycles(5000) == 10000
    assert get_watchdog_timeout_cycles(20000) == 20000


@pytest.mark.util
def test_liveness_threshold_env_can_only_raise_estimate(monkeypatch):
    monkeypatch.setenv("LIVENESS_THRESHOLD", "5000")
    assert get_watchdog_timeout_cycles(12000) == 12000

    monkeypatch.setenv("LIVENESS_THRESHOLD", "20000")
    assert get_watchdog_timeout_cycles(12000) == 20000


@pytest.mark.util
def test_rtlsim_timeout_error_message_is_actionable():
    message = get_rtlsim_timeout_error_message(20000, 12000)

    assert "timed out after 20000 cycles" in message
    assert "derived estimate: 12000" in message
    assert "set LIVENESS_THRESHOLD to a higher value" in message
