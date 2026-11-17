# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.core.rtlsim_exec import _output_frame_sizes
from finn.util.rtlsim import annotate_rtlsim_performance, summarize_frame_completions

pytestmark = pytest.mark.util


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

    result = annotate_rtlsim_performance(stats, batch_size=1, clock_period_ns=5.0)

    assert result["throughput[images/s]"] == pytest.approx(1.0e9 / (5.0 * 45_244))
    assert result["interval_is_steady_state"] is False
    assert result["fps_from_interval"] is None
    assert result["stable_throughput_valid"] is False
    assert result["stable_throughput[images/s]"] is None


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

    result = annotate_rtlsim_performance(stats, batch_size=2, clock_period_ns=10.0)

    assert result["interval_is_steady_state"] is False
    assert result["fps_from_interval"] is None
    assert result["stable_throughput_valid"] is False
    assert result["stable_throughput[images/s]"] is None


# Python XSI (MLO) path: frame completion cycles -> C++-style statistics


def test_frame_completions_single_frame_has_latency_only():
    stats = summarize_frame_completions({"out0": [100]}, cycles=101, batch_size=1)
    result = annotate_rtlsim_performance(stats, batch_size=1, clock_period_ns=5.0)

    assert result["latency_cycles"] == 100
    assert result["completed_output_frames"] == 1
    assert result["interval_valid"] == 0
    assert result["steady_state_frames"] == 0
    assert result["fps_from_interval"] is None
    assert result["stable_throughput[images/s]"] is None


def test_frame_completions_throughput_from_first_to_last_completion():
    stats = summarize_frame_completions({"out0": [100, 210, 330]}, cycles=331, batch_size=3)
    result = annotate_rtlsim_performance(stats, batch_size=3, clock_period_ns=10.0)

    assert result["latency_cycles"] == 100
    assert result["interval_cycles"] == 120
    assert result["steady_state_frames"] == 2
    assert result["steady_state_cycles"] == 230
    assert result["fps_from_interval"] == pytest.approx(1.0e9 / (10.0 * 120))
    assert result["stable_throughput_valid"] is True
    assert result["stable_throughput[images/s]"] == pytest.approx(2.0e9 / (10.0 * 230))


def test_frame_completions_use_slowest_output_stream():
    stats = summarize_frame_completions(
        {"fast": [90, 180, 270], "slow": [100, 205, 325]}, cycles=326, batch_size=3
    )

    assert stats["latency_cycles"] == 100
    assert stats["interval_cycles"] == 120
    assert stats["completed_output_frames"] == 3
    assert stats["steady_state_cycles"] == 225


def test_output_frame_sizes_from_batch():
    assert _output_frame_sizes(24, batchsize=3) == 8
    assert _output_frame_sizes({"out0": 24, "out1": 12}, batchsize=3) == {"out0": 8, "out1": 4}
    with pytest.raises(AssertionError, match="whole frames"):
        _output_frame_sizes({"out0": 10}, batchsize=3)
