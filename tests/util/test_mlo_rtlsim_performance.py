# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from finn.core.rtlsim_exec import _output_frame_sizes
from finn.util.mlo_sim import _resolve_mlo_weight_datfile
from finn.util.rtlsim_performance import (
    annotate_rtlsim_completion_throughput,
    summarize_output_frame_completions,
)


def test_single_frame_has_no_steady_state_interval():
    stats = summarize_output_frame_completions({"out0": [100]}, total_cycles=101)
    result = annotate_rtlsim_completion_throughput(stats, clk_ns=5.0)

    assert result["latency_cycles"] == 100
    assert result["completed_output_frames"] == 1
    assert result["interval_valid"] == 0
    assert result["steady_state_frames"] == 0
    assert result["steady_state_cycles"] == 0
    assert result["fps_from_interval"] is None
    assert result["stable_throughput_valid"] is False
    assert result["stable_throughput[images/s]"] is None


def test_multi_frame_throughput_uses_first_to_last_completion_span():
    stats = summarize_output_frame_completions({"out0": [100, 210, 330]}, total_cycles=331)
    result = annotate_rtlsim_completion_throughput(stats, clk_ns=10.0)

    assert result["latency_cycles"] == 100
    assert result["interval_cycles"] == 120
    assert result["completed_output_frames"] == 3
    assert result["steady_state_frames"] == 2
    assert result["steady_state_cycles"] == 230
    assert result["fps_from_interval"] == pytest.approx(1.0e9 / (10.0 * 120))
    assert result["stable_throughput_valid"] is True
    assert result["stable_throughput[images/s]"] == pytest.approx(2.0e9 / (10.0 * 230))


def test_multi_output_summary_uses_slowest_complete_result():
    stats = summarize_output_frame_completions(
        {
            "fast": [90, 180, 270],
            "slow": [100, 205, 325],
        },
        total_cycles=326,
    )

    assert stats["latency_cycles"] == 100
    assert stats["interval_cycles"] == 120
    assert stats["completed_output_frames"] == 3
    assert stats["steady_state_frames"] == 2
    assert stats["steady_state_cycles"] == 225


def test_output_frame_sizes_are_derived_from_batch():
    assert _output_frame_sizes(24, batchsize=3) == 8
    assert _output_frame_sizes({"out0": 24, "out1": 12}, batchsize=3) == {
        "out0": 8,
        "out1": 4,
    }


def test_output_frame_sizes_reject_partial_frames():
    with pytest.raises(AssertionError, match="whole frames"):
        _output_frame_sizes({"out0": 10}, batchsize=3)


def test_mlo_weight_image_uses_specialized_mvau_type(tmp_path):
    hls_weights = tmp_path / "memblock_MVAU_hls_id_4.dat"
    hls_weights.write_text("00\n")

    assert _resolve_mlo_weight_datfile(tmp_path, "MVAU_hls", 4) == hls_weights
    with pytest.raises(FileNotFoundError, match="MVAU_rtl"):
        _resolve_mlo_weight_datfile(tmp_path, "MVAU_rtl", 4)
