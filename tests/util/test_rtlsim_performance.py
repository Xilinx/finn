# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.builder.build_dataflow_steps import _annotate_rtlsim_performance


def test_rtlsim_performance_defaults_to_multi_frame_measurement():
    cfg = DataflowBuildConfig(output_dir=".", synth_clk_period_ns=5.0, generate_outputs=[])
    assert cfg.rtlsim_batch_size == 2


def test_rtlsim_performance_rejects_single_frame_interval():
    stats = {
        "cycles": 45_244,
        "latency_cycles": 45_243,
        "interval_cycles": 26_331,
        "N": 1,
        "TIMEOUT": 0,
        "UNFINISHED_INS": 0,
        "UNFINISHED_OUTS": 0,
    }

    result = _annotate_rtlsim_performance(stats, batch_size=1, clock_period_ns=5.0)

    assert result["throughput[images/s]"] == pytest.approx(1.0e9 / (5.0 * 45_244))
    assert result["interval_is_steady_state"] is False
    assert result["fps_from_interval"] is None
    assert result["stable_throughput_valid"] is False
    assert result["stable_throughput[images/s]"] is None


def test_rtlsim_performance_uses_completed_frame_span():
    stats = {
        "cycles": 331,
        "latency_cycles": 100,
        "interval_cycles": 110,
        "interval_valid": 1,
        "completed_output_frames": 3,
        "steady_state_frames": 2,
        "steady_state_cycles": 230,
        "N": 3,
        "TIMEOUT": 0,
        "UNFINISHED_INS": 0,
        "UNFINISHED_OUTS": 0,
    }

    result = _annotate_rtlsim_performance(stats, batch_size=3, clock_period_ns=10.0)

    assert result["interval_is_steady_state"] is True
    assert result["fps_from_interval"] == pytest.approx(1.0e9 / (10.0 * 110))
    assert result["stable_throughput_valid"] is True
    assert result["stable_throughput[images/s]"] == pytest.approx(
        2.0e9 / (10.0 * 230)
    )


def test_rtlsim_performance_corrects_legacy_multi_frame_numerator():
    stats = {
        "cycles": 300,
        "latency_cycles": 100,
        "interval_cycles": 200,
        "N": 2,
        "TIMEOUT": 0,
        "UNFINISHED_INS": 0,
        "UNFINISHED_OUTS": 0,
    }

    result = _annotate_rtlsim_performance(stats, batch_size=2, clock_period_ns=10.0)

    assert result["stable_throughput_valid"] is True
    assert result["stable_throughput[images/s]"] == pytest.approx(1.0e9 / (10.0 * 200))


def test_rtlsim_performance_rejects_incomplete_run():
    stats = {
        "cycles": 300,
        "latency_cycles": 100,
        "interval_cycles": 200,
        "interval_valid": 1,
        "completed_output_frames": 2,
        "steady_state_frames": 1,
        "steady_state_cycles": 200,
        "N": 2,
        "TIMEOUT": 0,
        "UNFINISHED_INS": 1,
        "UNFINISHED_OUTS": 0,
    }

    result = _annotate_rtlsim_performance(stats, batch_size=2, clock_period_ns=10.0)

    assert result["interval_is_steady_state"] is False
    assert result["fps_from_interval"] is None
    assert result["stable_throughput_valid"] is False
    assert result["stable_throughput[images/s]"] is None
