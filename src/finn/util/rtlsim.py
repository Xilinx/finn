# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# This module contains helpers for RTL simulation, including MLO prehook setup
# and performance metrics annotation.

import numpy as np
from qonnx.custom_op.registry import getCustomOp
from typing import Callable

from finn import xsi

SimEngine = xsi.SimEngine if xsi.is_available() else None


def annotate_rtlsim_performance(rtlsim_stats, batch_size, clock_period_ns):
    """Add latency and throughput metrics to raw XSI simulation statistics.

    Overall throughput includes pipeline fill and is available for any completed
    run. Steady-state throughput requires at least two completed output frames;
    one frame provides latency only and cannot define an output-to-output rate.

    Args:
        rtlsim_stats: Dictionary of raw statistics from XSI simulation
        batch_size: Number of frames simulated
        clock_period_ns: Clock period in nanoseconds

    Returns:
        Updated rtlsim_stats dictionary with computed metrics
    """
    batch_size = int(batch_size)
    clock_period_ns = float(clock_period_ns)
    cycles = int(rtlsim_stats["cycles"])
    latency_cycles = int(rtlsim_stats["latency_cycles"])
    assert batch_size > 0, "rtlsim batch size must be >0"
    assert cycles > 0, "rtlsim cycle count must be >0"
    assert clock_period_ns > 0.0, "rtlsim clock period must be >0"

    runtime_s = cycles * clock_period_ns * 1.0e-9
    rtlsim_stats["runtime[ms]"] = runtime_s * 1000.0
    rtlsim_stats["throughput[images/s]"] = batch_size / runtime_s
    rtlsim_stats["fclk[mhz]"] = 1000.0 / clock_period_ns

    timeout = int(rtlsim_stats.get("TIMEOUT", 1))
    unfinished_inputs = int(rtlsim_stats.get("UNFINISHED_INS", 1))
    unfinished_outputs = int(rtlsim_stats.get("UNFINISHED_OUTS", 1))
    run_complete = timeout == 0 and unfinished_inputs == 0 and unfinished_outputs == 0
    completed_frames = int(
        rtlsim_stats.get("completed_output_frames", batch_size if run_complete else 0)
    )
    run_complete = run_complete and completed_frames >= batch_size

    interval_cycles = int(rtlsim_stats.get("interval_cycles", 0))
    xsi_interval_valid = bool(
        int(rtlsim_stats.get("interval_valid", completed_frames >= 2 and interval_cycles > 0))
    )
    interval_valid = (
        run_complete and completed_frames >= 2 and interval_cycles > 0 and xsi_interval_valid
    )
    rtlsim_stats["interval_is_steady_state"] = interval_valid
    rtlsim_stats["fps_from_interval"] = (
        1.0e9 / (clock_period_ns * interval_cycles) if interval_valid else None
    )

    # New XSI results report the exact span and frame count between the first
    # and last completed outputs. Fall back to legacy results by removing the
    # first (pipeline-fill) frame from both the count and elapsed cycles.
    steady_state_frames = int(rtlsim_stats.get("steady_state_frames", max(0, batch_size - 1)))
    steady_state_cycles = int(
        rtlsim_stats.get("steady_state_cycles", max(0, cycles - latency_cycles))
    )
    stable_valid = (
        run_complete
        and completed_frames >= 2
        and steady_state_frames > 0
        and steady_state_cycles > 0
    )
    rtlsim_stats["stable_throughput_valid"] = stable_valid
    rtlsim_stats["stable_throughput[images/s]"] = (
        steady_state_frames * 1.0e9 / (clock_period_ns * steady_state_cycles)
        if stable_valid
        else None
    )
    return rtlsim_stats


def annotate_rtlsim_mlo_capacity(rtlsim_stats, loop_iterations, overhead_per_iteration=0):
    """Convert a FINNLoop body pass rate into external-frame service capacity.

    FINNLoop may keep several frames in flight, so adjacent top-level outputs
    can be a burst and their raw spacing is not the long-run image rate. Every
    completed external frame nevertheless consumes ``loop_iterations`` body
    passes. A multi-frame body RTLSIM therefore gives the work-normalized
    external interval without confusing output burst spacing with capacity.
    Body output intervals can alternate as reorder buffers move through their
    ping-pong phases. Use an odd number of at least three body frames so the
    first-to-last span contains an even number of intervals, then use that
    periodic average rather than either member of the alternating pair.
    """

    loop_iterations = int(loop_iterations)
    overhead_per_iteration = int(overhead_per_iteration)
    assert loop_iterations > 0, "FINNLoop iteration count must be >0"
    assert overhead_per_iteration >= 0, "FINNLoop iteration overhead must be >=0"

    interval_valid = bool(rtlsim_stats.get("interval_is_steady_state", False))
    completed_frames = int(rtlsim_stats.get("completed_output_frames", 0))
    body_interval_cycles = int(rtlsim_stats.get("interval_cycles", 0))
    steady_state_frames = int(rtlsim_stats.get("steady_state_frames", 0))
    body_steady_state_cycles = int(rtlsim_stats.get("steady_state_cycles", 0))
    body_average_interval_cycles = (
        body_steady_state_cycles / steady_state_frames if steady_state_frames > 0 else None
    )
    if (
        body_average_interval_cycles is not None
        and float(body_average_interval_cycles).is_integer()
    ):
        body_average_interval_cycles = int(body_average_interval_cycles)
    rtlsim_stats["measurement_scope"] = "finnloop_body_capacity"
    rtlsim_stats["loop_iterations"] = loop_iterations
    rtlsim_stats["loop_iteration_overhead_cycles"] = overhead_per_iteration
    rtlsim_stats["body_interval_cycles"] = body_interval_cycles
    rtlsim_stats["body_interval_is_steady_state"] = interval_valid
    rtlsim_stats["body_fps_from_interval"] = rtlsim_stats.get("fps_from_interval")
    rtlsim_stats["body_cycles"] = rtlsim_stats.get("cycles")
    rtlsim_stats["body_latency_cycles"] = rtlsim_stats.get("latency_cycles")
    rtlsim_stats["body_throughput[images/s]"] = rtlsim_stats.get("throughput[images/s]")
    rtlsim_stats["body_stable_throughput_valid"] = rtlsim_stats.get("stable_throughput_valid")
    rtlsim_stats["body_stable_throughput[images/s]"] = rtlsim_stats.get(
        "stable_throughput[images/s]"
    )
    rtlsim_stats["body_average_interval_cycles"] = body_average_interval_cycles

    if (
        not interval_valid
        or completed_frames < 3
        or completed_frames % 2 == 0
        or body_interval_cycles <= 0
        or steady_state_frames <= 0
        or body_steady_state_cycles <= 0
    ):
        rtlsim_stats["effective_interval_cycles"] = None
        rtlsim_stats["effective_interval_is_steady_state"] = False
        rtlsim_stats["effective_fps_from_interval"] = None
        rtlsim_stats["interval_is_steady_state"] = False
        rtlsim_stats["fps_from_interval"] = None
        rtlsim_stats["throughput[images/s]"] = None
        rtlsim_stats["stable_throughput_valid"] = False
        rtlsim_stats["stable_throughput[images/s]"] = None
        return rtlsim_stats

    effective_interval_cycles = loop_iterations * (
        body_average_interval_cycles + overhead_per_iteration
    )
    if float(effective_interval_cycles).is_integer():
        effective_interval_cycles = int(effective_interval_cycles)
    fclk_mhz = float(rtlsim_stats["fclk[mhz]"])
    effective_fps = fclk_mhz * 1.0e6 / effective_interval_cycles
    rtlsim_stats["effective_interval_cycles"] = effective_interval_cycles
    rtlsim_stats["effective_interval_is_steady_state"] = True
    rtlsim_stats["effective_fps_from_interval"] = effective_fps
    # Keep standard network-throughput fields useful to existing report
    # consumers while retaining every raw body quantity under a body_* key.
    rtlsim_stats["interval_cycles"] = effective_interval_cycles
    rtlsim_stats["interval_is_steady_state"] = True
    rtlsim_stats["fps_from_interval"] = effective_fps
    rtlsim_stats["throughput[images/s]"] = effective_fps
    effective_steady_state_cycles = effective_interval_cycles * steady_state_frames
    rtlsim_stats["body_steady_state_cycles"] = body_steady_state_cycles
    rtlsim_stats["steady_state_cycles"] = effective_steady_state_cycles
    rtlsim_stats["stable_throughput_valid"] = True
    rtlsim_stats["stable_throughput[images/s]"] = (
        steady_state_frames * fclk_mhz * 1.0e6 / effective_steady_state_cycles
    )
    return rtlsim_stats


def annotate_rtlsim_estimate_comparison(rtlsim_stats, estimated_interval_cycles):
    """Compare an analytical initiation interval against a valid RTLSIM interval.

    Ordinary streaming graphs use the raw output-to-output interval. Rolled
    MLO graphs use ``effective_interval_cycles`` when body RTLSIM has converted
    pass rate into work-normalized external-frame capacity. Invalid or
    single-frame RTLSIM runs retain the estimate but do not report an error.
    """

    estimated_interval_cycles = int(estimated_interval_cycles)
    assert estimated_interval_cycles > 0, "estimated interval must be >0"
    rtlsim_stats["estimated_interval_cycles"] = estimated_interval_cycles

    clock_period_ns = float(rtlsim_stats["fclk[mhz]"])
    clock_period_ns = 1000.0 / clock_period_ns
    estimated_fps = 1.0e9 / (clock_period_ns * estimated_interval_cycles)
    rtlsim_stats["estimated_throughput[images/s]"] = estimated_fps

    is_mlo_capacity = rtlsim_stats.get("measurement_scope") == "finnloop_body_capacity"
    effective_interval = rtlsim_stats.get("effective_interval_cycles")
    if is_mlo_capacity:
        rtlsim_stats["estimate_comparison_interval_source"] = "effective_interval_cycles"
        comparison_valid = bool(
            effective_interval is not None
            and rtlsim_stats.get("effective_interval_is_steady_state", False)
        )
        measured_interval_cycles = float(effective_interval or 0)
        measured_fps = float(rtlsim_stats.get("effective_fps_from_interval") or 0)
    else:
        measured_interval_cycles = float(rtlsim_stats.get("interval_cycles", 0))
        measured_fps_value = rtlsim_stats.get("fps_from_interval")
        measured_fps = float(measured_fps_value) if measured_fps_value is not None else 0.0
        comparison_valid = bool(rtlsim_stats.get("interval_is_steady_state", False))
        rtlsim_stats["estimate_comparison_interval_source"] = "interval_cycles"

    if not comparison_valid:
        rtlsim_stats["estimate_vs_rtlsim_cycles"] = None
        rtlsim_stats["estimate_vs_rtlsim_cycles[%]"] = None
        rtlsim_stats["estimate_vs_rtlsim_cycles_abs[%]"] = None
        rtlsim_stats["estimate_vs_rtlsim_throughput[%]"] = None
        return rtlsim_stats

    cycle_delta = estimated_interval_cycles - measured_interval_cycles
    if float(cycle_delta).is_integer():
        cycle_delta = int(cycle_delta)
    cycle_error_pct = 100.0 * cycle_delta / measured_interval_cycles
    throughput_error_pct = 100.0 * (estimated_fps - measured_fps) / measured_fps
    rtlsim_stats["estimate_vs_rtlsim_cycles"] = cycle_delta
    rtlsim_stats["estimate_vs_rtlsim_cycles[%]"] = cycle_error_pct
    rtlsim_stats["estimate_vs_rtlsim_cycles_abs[%]"] = abs(cycle_error_pct)
    rtlsim_stats["estimate_vs_rtlsim_throughput[%]"] = throughput_error_pct
    return rtlsim_stats


def dat_file_to_numpy_array(file_path):
    byte_values = []

    with open(file_path, "r") as file:
        for line in file:
            hex_string = line.strip()
            for i in range(len(hex_string) - 2, -1, -2):
                byte = hex_string[i : i + 2]
                byte_values.append(int(byte, 16))
            if len(hex_string) % 2 == 1:  # Dealing when we have a leftover nibble
                byte_values.append(int(hex_string[-1], 16))
    byte_array = np.array(byte_values, dtype=np.uint8)

    return byte_array


def mlo_prehook_func_factory(node) -> Callable[[SimEngine], None]:
    """Factory that will construct a prehook function to
    setup the axi memory mapped interfaces for MLO validation.
    """

    # Get the FINNLoop
    finnloop_op = getCustomOp(node)

    finnloop_body = finnloop_op.get_nodeattr("body")

    mvau_mlo_weights = {}
    extern_idx = 0
    for idx, lb_inp in enumerate(finnloop_body.graph.input):
        downstream = finnloop_body.find_consumer(lb_inp.name)
        if downstream.op_type.startswith("MVAU"):
            mvau_mlo_weights[idx] = {}
            mvau_mlo_weights[idx]["name"] = lb_inp.name
            code_gen_dir = finnloop_op.get_nodeattr("code_gen_dir_ipgen")
            datfile = f"{code_gen_dir}/memblock_MVAU_rtl_id_{idx}.dat"
            # memblock.dat already holds the per-layer weights padded to LAYER_OFFS
            weight_bytes = dat_file_to_numpy_array(datfile)
            mvau_mlo_weights[idx]["value"] = weight_bytes
            mvau_mlo_weights[idx]["extern_idx"] = extern_idx
            mvau_mlo_weights[idx]["extern_name"] = f"m_axi_MVAU_id_{idx}"
            mvau_mlo_weights[idx]["offset"] = getCustomOp(downstream).get_nodeattr("address_offset")
            extern_idx += 1

    def mlo_rtlsim_prehook(sim):
        sim.aximm_queue("m_axi_intermediate_frame")
        for name, intf in mvau_mlo_weights.items():
            sim.aximm_ro_image(intf["extern_name"], intf["offset"], intf["value"].flatten())

    return mlo_rtlsim_prehook
