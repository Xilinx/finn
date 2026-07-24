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

from pathlib import Path
from typing import Callable

import numpy as np
from qonnx.custom_op.registry import getCustomOp

from finn import xsi

SimEngine = xsi.SimEngine if xsi.is_available() else None


def _has_aximm_read_bus(sim: SimEngine, name: str) -> bool:
    """Return True when the AXI-MM read channel is present on the sim top."""

    required = ("arready", "arvalid", "araddr", "arlen", "arsize", "rready", "rvalid", "rdata")
    return all(sim.get_bus_port(name, suffix) is not None for suffix in required)


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


def _resolve_mlo_weight_datfile(code_gen_dir, op_type, input_index):
    """Return the generated external-weight image for an MLO MVAU input."""

    datfile = Path(code_gen_dir) / f"memblock_{op_type}_id_{input_index}.dat"
    if not datfile.is_file():
        raise FileNotFoundError(
            f"Missing {op_type} MLO weight image for loop input {input_index}: {datfile}"
        )
    return datfile


def mlo_prehook_func_factory(node) -> Callable[[SimEngine], None]:
    """Factory that will construct a prehook function to
    setup the axi memory mapped interfaces for MLO validation.
    """

    # Get the FINNLoop
    finnloop_op = getCustomOp(node)

    finnloop_body = finnloop_op.get_nodeattr("body")

    mvau_hbm_weights = {}
    extern_idx = 0
    for idx, lb_inp in enumerate(finnloop_body.graph.input):
        downstream = finnloop_body.find_consumer(lb_inp.name)
        if downstream.op_type.startswith("MVAU"):
            mvau_hbm_weights[idx] = {}
            mvau_hbm_weights[idx]["name"] = lb_inp.name
            code_gen_dir = finnloop_op.get_nodeattr("code_gen_dir_ipgen")
            datfile = _resolve_mlo_weight_datfile(code_gen_dir, downstream.op_type, idx)
            # memblock.dat holds the per-layer weights back-to-back, byte-aligned
            # per IWSIMD group. fetch_weights.sv places layer i at i*LAYER_OFFS,
            # where LAYER_OFFS rounds the layer size up to the AXI bus width, so
            # pad each layer to that boundary to reproduce the external memory image.
            weight_bytes = dat_file_to_numpy_array(datfile)
            iteration = finnloop_op.get_nodeattr("iteration")
            layer_bytes = len(weight_bytes) // iteration
            axi_bytes = 32  # AXI bus width (256 bits), matches RTL LAYER_OFFS
            layer_offs = (layer_bytes + axi_bytes - 1) & ~(axi_bytes - 1)
            if layer_offs != layer_bytes:
                padded = np.zeros(iteration * layer_offs, dtype=weight_bytes.dtype)
                for it in range(iteration):
                    padded[it * layer_offs : it * layer_offs + layer_bytes] = weight_bytes[
                        it * layer_bytes : (it + 1) * layer_bytes
                    ]
                weight_bytes = padded
            mvau_hbm_weights[idx]["value"] = weight_bytes
            mvau_hbm_weights[idx]["extern_idx"] = extern_idx
            mvau_hbm_weights[idx]["extern_name"] = f"m_axi_MVAU_id_{idx}"
            extern_idx += 1

    def mlo_rtlsim_prehook(sim):
        if _has_aximm_read_bus(sim, "m_axi_hbm"):
            sim.aximm_queue("m_axi_hbm")
        for name, intf in mvau_hbm_weights.items():
            if _has_aximm_read_bus(sim, intf["extern_name"]):
                sim.aximm_ro_image(intf["extern_name"], 0, intf["value"].flatten())

    return mlo_rtlsim_prehook
