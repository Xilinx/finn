# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for deriving throughput from RTLSIM frame completion times."""

import numpy as np
from qonnx.util.basic import gen_finn_dt_tensor


def summarize_output_frame_completions(completion_cycles, total_cycles):
    """Summarize per-stream frame completion cycles conservatively.

    A complete multi-output frame is available only when every output stream
    has completed it. Compute that timestamp for each frame first, then derive
    latency and spans from the aggregate sequence. This remains correct when
    the slowest output stream changes between frames.
    """

    assert completion_cycles, "At least one output stream is required"
    completed_frames = min(len(cycles) for cycles in completion_cycles.values())
    aggregate_completion_cycles = [
        max(cycles[frame] for cycles in completion_cycles.values())
        for frame in range(completed_frames)
    ]
    latency_cycles = 0
    interval_cycles = 0
    steady_state_cycles = 0
    if completed_frames > 0:
        latency_cycles = aggregate_completion_cycles[0]
    if completed_frames > 1:
        interval_cycles = aggregate_completion_cycles[-1] - aggregate_completion_cycles[-2]
        steady_state_cycles = aggregate_completion_cycles[-1] - aggregate_completion_cycles[0]

    interval_valid = completed_frames > 1 and interval_cycles > 0
    return {
        "cycles": int(total_cycles),
        "latency_cycles": int(latency_cycles),
        "interval_cycles": int(interval_cycles),
        "completed_output_frames": int(completed_frames),
        "interval_valid": int(interval_valid),
        "steady_state_frames": int(max(0, completed_frames - 1)),
        "steady_state_cycles": int(steady_state_cycles),
        "aggregate_output_frame_completion_cycles": aggregate_completion_cycles,
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


def throughput_test_rtlsim(
    model,
    clk_ns,
    batchsize=100,
    pre_hook=None,
    collect_performance=False,
):
    """Measure a stitched design with the Python XSI simulation engine.

    The Python engine is required for MLO designs because ``pre_hook`` installs
    the ideal AXI-MM models used by external loop weights. When
    ``collect_performance`` is enabled, sustained throughput is derived from
    completed output-frame timestamps rather than pipeline-fill latency.
    """

    # Keep this import local: rtlsim_exec imports finn.util.rtlsim, which loads
    # the FINN XSI adapter, and that adapter imports this module's summary helper.
    from finn.core.rtlsim_exec import rtlsim_exec  # noqa: PLC0415

    assert model.get_metadata_prop("exec_mode") == "rtlsim"
    assert batchsize > 0, "rtlsim batch size must be greater than zero"
    assert clk_ns > 0.0, "RTLSIM clock period must be greater than zero"

    ctx = model.make_empty_exec_context()
    input_bytes = 0
    for input_info in model.graph.input:
        input_name = input_info.name
        input_shape = list(model.get_tensor_shape(input_name))
        input_shape[0] = batchsize
        input_datatype = model.get_tensor_datatype(input_name)
        ctx[input_name] = gen_finn_dt_tensor(input_datatype, input_shape)
        input_bytes += (np.prod(input_shape) * input_datatype.bitwidth()) / 8

    output_bytes = 0
    for output_info in model.graph.output:
        output_shape = list(model.get_tensor_shape(output_info.name))
        output_shape[0] = batchsize
        output_datatype = model.get_tensor_datatype(output_info.name)
        output_bytes += (np.prod(output_shape) * output_datatype.bitwidth()) / 8

    completion_stats = rtlsim_exec(
        model,
        ctx,
        pre_hook=pre_hook,
        collect_performance=collect_performance,
    )
    cycles = int(model.get_metadata_prop("cycles_rtlsim"))
    runtime_s = cycles * clk_ns * 1.0e-9
    stats = {
        "cycles": cycles,
        "runtime[ms]": runtime_s * 1000.0,
        "throughput[images/s]": batchsize / runtime_s,
        "DRAM_in_bandwidth[MB/s]": input_bytes * 1.0e-6 / runtime_s,
        "DRAM_out_bandwidth[MB/s]": output_bytes * 1.0e-6 / runtime_s,
        "fclk[mhz]": 1000.0 / clk_ns,
        "N": batchsize,
    }
    if collect_performance:
        stats.update(completion_stats)
        annotate_rtlsim_completion_throughput(stats, clk_ns)
    return stats
