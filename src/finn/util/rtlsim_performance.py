# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for deriving throughput from RTLSIM frame completion times."""


def summarize_output_frame_completions(completion_cycles, total_cycles):
    """Summarize per-stream frame completion cycles conservatively.

    A complete multi-output frame is available only when every output stream
    has completed it. The reported latency and spans therefore use the slowest
    output stream for each measurement.
    """

    assert completion_cycles, "At least one output stream is required"
    completed_frames = min(len(cycles) for cycles in completion_cycles.values())
    latency_cycles = 0
    interval_cycles = 0
    steady_state_cycles = 0
    if completed_frames > 0:
        latency_cycles = max(cycles[0] for cycles in completion_cycles.values())
    if completed_frames > 1:
        interval_cycles = max(
            cycles[completed_frames - 1] - cycles[completed_frames - 2]
            for cycles in completion_cycles.values()
        )
        steady_state_cycles = max(
            cycles[completed_frames - 1] - cycles[0] for cycles in completion_cycles.values()
        )

    interval_valid = completed_frames > 1 and interval_cycles > 0
    return {
        "cycles": int(total_cycles),
        "latency_cycles": int(latency_cycles),
        "interval_cycles": int(interval_cycles),
        "completed_output_frames": int(completed_frames),
        "interval_valid": int(interval_valid),
        "steady_state_frames": int(max(0, completed_frames - 1)),
        "steady_state_cycles": int(steady_state_cycles),
        "output_frame_completion_cycles": completion_cycles,
    }


def annotate_rtlsim_completion_throughput(stats, clk_ns):
    """Add sustained-throughput metrics to frame completion statistics."""

    clk_ns = float(clk_ns)
    assert clk_ns > 0.0, "RTLSIM clock period must be greater than zero"
    interval_cycles = int(stats["interval_cycles"])
    interval_valid = bool(stats["interval_valid"])
    steady_state_frames = int(stats["steady_state_frames"])
    steady_state_cycles = int(stats["steady_state_cycles"])
    stable_valid = interval_valid and steady_state_frames > 0 and steady_state_cycles > 0
    stats["interval_is_steady_state"] = interval_valid
    stats["fps_from_interval"] = 1.0e9 / (clk_ns * interval_cycles) if interval_valid else None
    stats["stable_throughput_valid"] = stable_valid
    stats["stable_throughput[images/s]"] = (
        steady_state_frames * 1.0e9 / (clk_ns * steady_state_cycles) if stable_valid else None
    )
    return stats
