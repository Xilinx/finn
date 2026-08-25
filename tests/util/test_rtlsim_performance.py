# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.util.rtlsim import (
    annotate_rtlsim_estimate_comparison,
    annotate_rtlsim_mlo_capacity,
    annotate_rtlsim_performance,
)


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


def test_rtlsim_estimate_comparison_uses_raw_multi_frame_interval():
    stats = {
        "cycles": 910_067,
        "latency_cycles": 737_260,
        "interval_cycles": 172_806,
        "interval_valid": 1,
        "completed_output_frames": 2,
        "steady_state_frames": 1,
        "steady_state_cycles": 172_806,
        "TIMEOUT": 0,
        "UNFINISHED_INS": 0,
        "UNFINISHED_OUTS": 0,
    }
    result = annotate_rtlsim_performance(stats, batch_size=2, clock_period_ns=10.0)
    result = annotate_rtlsim_estimate_comparison(result, estimated_interval_cycles=172_800)

    assert result["estimated_interval_cycles"] == 172_800
    assert result["estimate_vs_rtlsim_cycles"] == -6
    assert result["estimate_vs_rtlsim_cycles[%]"] == pytest.approx(-0.00347210166)
    assert result["estimate_vs_rtlsim_cycles_abs[%]"] == pytest.approx(0.00347210166)
    assert result["estimate_vs_rtlsim_throughput[%]"] == pytest.approx(0.00347222222)


def test_mlo_capacity_scales_body_pass_interval_before_estimate_comparison():
    stats = {
        "cycles": 518_776,
        "latency_cycles": 291_831,
        "interval_cycles": 126_082,
        "interval_valid": 1,
        "completed_output_frames": 3,
        "steady_state_frames": 2,
        "steady_state_cycles": 226_944,
        "TIMEOUT": 0,
        "UNFINISHED_INS": 0,
        "UNFINISHED_OUTS": 0,
    }
    result = annotate_rtlsim_performance(stats, batch_size=3, clock_period_ns=5.0)
    result = annotate_rtlsim_mlo_capacity(result, loop_iterations=12, overhead_per_iteration=40)
    result = annotate_rtlsim_estimate_comparison(result, estimated_interval_cycles=1_359_912)

    assert result["body_interval_cycles"] == 126_082
    assert result["body_average_interval_cycles"] == 113_472
    assert result["effective_interval_cycles"] == 1_362_144
    assert result["interval_cycles"] == 1_362_144
    assert result["effective_fps_from_interval"] == pytest.approx(200_000_000.0 / 1_362_144)
    assert result["fps_from_interval"] == pytest.approx(200_000_000.0 / 1_362_144)
    assert result["stable_throughput[images/s]"] == pytest.approx(200_000_000.0 / 1_362_144)
    assert result["estimate_comparison_interval_source"] == "effective_interval_cycles"
    assert result["estimate_vs_rtlsim_cycles"] == -2_232
    assert result["estimate_vs_rtlsim_cycles_abs[%]"] == pytest.approx(100.0 * 2_232 / 1_362_144)


def test_mlo_capacity_rejects_two_frame_warmup_interval():
    stats = {
        "cycles": 392_694,
        "latency_cycles": 291_831,
        "interval_cycles": 100_862,
        "interval_valid": 1,
        "completed_output_frames": 2,
        "steady_state_frames": 1,
        "steady_state_cycles": 100_862,
        "TIMEOUT": 0,
        "UNFINISHED_INS": 0,
        "UNFINISHED_OUTS": 0,
    }
    result = annotate_rtlsim_performance(stats, batch_size=2, clock_period_ns=5.0)
    result = annotate_rtlsim_mlo_capacity(result, loop_iterations=12)
    result = annotate_rtlsim_estimate_comparison(result, estimated_interval_cycles=1_500_000)

    assert result["effective_interval_is_steady_state"] is False
    assert result["effective_interval_cycles"] is None
    assert result["body_interval_cycles"] == 100_862
    assert result["estimate_comparison_interval_source"] == "effective_interval_cycles"
    assert result["estimate_vs_rtlsim_cycles"] is None
    assert result["fps_from_interval"] is None
    assert result["stable_throughput_valid"] is False
    assert result["stable_throughput[images/s]"] is None


def test_mlo_capacity_rejects_unbalanced_four_frame_phase_average():
    stats = {
        "cycles": 619_638,
        "latency_cycles": 291_831,
        "interval_cycles": 100_862,
        "interval_valid": 1,
        "completed_output_frames": 4,
        "steady_state_frames": 3,
        "steady_state_cycles": 327_806,
        "TIMEOUT": 0,
        "UNFINISHED_INS": 0,
        "UNFINISHED_OUTS": 0,
    }
    result = annotate_rtlsim_performance(stats, batch_size=4, clock_period_ns=5.0)
    result = annotate_rtlsim_mlo_capacity(result, loop_iterations=12)

    assert result["body_average_interval_cycles"] == pytest.approx(327_806 / 3)
    assert result["effective_interval_is_steady_state"] is False
    assert result["effective_interval_cycles"] is None


def test_rtlsim_estimate_comparison_rejects_single_frame_error():
    stats = {
        "cycles": 45_244,
        "latency_cycles": 45_243,
        "interval_cycles": 26_331,
        "TIMEOUT": 0,
        "UNFINISHED_INS": 0,
        "UNFINISHED_OUTS": 0,
    }
    result = annotate_rtlsim_performance(stats, batch_size=1, clock_period_ns=5.0)
    result = annotate_rtlsim_estimate_comparison(result, estimated_interval_cycles=25_000)

    assert result["estimated_interval_cycles"] == 25_000
    assert result["estimate_vs_rtlsim_cycles"] is None
    assert result["estimate_vs_rtlsim_cycles[%]"] is None
    assert result["estimate_vs_rtlsim_throughput[%]"] is None
