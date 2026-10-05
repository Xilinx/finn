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
import os
import re
from qonnx.custom_op.registry import getCustomOp
from typing import Any, Callable, Dict, Tuple

from finn import xsi

SimEngine = xsi.SimEngine if xsi.is_available() else None


def parse_fifo_log(path: str) -> Dict[str, Any]:
    """Parse one FIFO gauge log, as written by finn-rtllib/fifo/hdl/fifo_gauge.sv.

    The plain and verbose formats are told apart by the log's own header line,
    so callers do not have to know which one the simulation was configured for.

    Args:
        path: Path to a single ``<fifo name>.log`` file.

    Returns:
        A dict with the log's contents:

        * path: Path of log.
        * verbose: False: <data_in>, True: <data>, <direction>, <cycle>.
        * name: the SystemVerilog name of the FIFO instance.
        * txns: List of(data:int, direction:Optional[int], cycle:Optional[int])
        * cycles, maxfill, in, out: the gauge's summary (bottom line).
    """

    def is_verbose(header_line: str) -> bool:
        """Determine if FIFO logs are in verbose mode based on header."""
        headers = {"# data": False, "# data dir cycle": True}
        if header_line not in headers.keys():
            raise ValueError(
                "%s: header is %r, expected one of %s" % (path, header_line, headers.keys())
            )
        return headers[header_line]

    def summary(text: str) -> Tuple[str, Dict[str, int]]:
        """Parse last line of the logs, the summary line."""
        # Format regex
        summary_regex = re.compile(
            r"^# \[(?P<name>.+) @(?P<time>\d+)\] Cycles: (?P<cycles>\d+); "
            r"MaxFill: (?P<maxfill>\d+); Transactions: in=(?P<n_in>\d+) out=(?P<n_out>\d+)$"
        )

        # Match & validate
        sum_match = summary_regex.match(text)
        if sum_match is None:
            raise ValueError("%s: last line %r is not a gauge summary line" % (path, text))

        # Transform to int dictionary
        summary_dict = sum_match.groupdict()
        fifo_name = summary_dict["name"]
        fifo_values = {k: int(v) for k, v in summary_dict.items() if k != "name"}

        return fifo_name, fifo_values

    def data(line: str, verbose: bool, lineno: int, path: str):
        """Parse a line of the data."""
        # Map Verbose -> Regex
        data_regex = (
            re.compile(r"^(?P<data>[0-9a-fA-FxXzZ]+)$")
            if not verbose
            else re.compile(r"^(?P<data>[0-9a-fA-FxXzZ]+) (?P<dir>[01]) (?P<cycle>\d+)$")
        )

        # Match & Validate
        m = data_regex.match(line)
        if m is None:
            data_syntax = {False: "<hex>", True: "<hex> <0|1> <cycle>"}
            raise ValueError(
                "%s:%d: %r does not match %r" % (path, lineno, line, data_syntax[verbose])
            )

        # Handle X/Z Values
        data = None if re.search(r"[xXzZ]", m.group("data")) else int(m.group("data"), 16)
        direction = None if not verbose else int(m.group("dir"))
        cycle = None if not verbose else int(m.group("cycle"))
        return (data, direction, cycle)

    # Buffer FIFO Log
    with open(path) as f:
        lines = f.read().splitlines()
    if len(lines) < 2:
        raise ValueError("%s: expected at least a header and a summary line" % path)

    # Parse
    verbose = is_verbose(lines[0])
    fifo_name, fifo_values = summary(lines[-1])
    txns = [data(line, verbose, lineno, path) for lineno, line in enumerate(lines[1:-1], start=2)]

    # Turn to dictionary
    return {
        "path": path,
        "verbose": verbose,
        "name": fifo_name,
        "txns": txns,
        "cycles": fifo_values["cycles"],
        "maxfill": fifo_values["maxfill"],
        "in": fifo_values["n_in"],
        "out": fifo_values["n_out"],
    }


def read_fifo_log_snapshot(log_dir: str) -> Dict[str, Dict[str, Any]]:
    """Parse every FIFO gauge log in one snapshot directory."
    Args - log_dir: str path
    Return - Map FIFO Name -> parse_fifo_log() dictionary.
    """
    if not os.path.isdir(log_dir):
        raise ValueError("no FIFO log directory at " + log_dir)
    logs = {}
    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith(".log"):
            continue
        path = os.path.join(log_dir, fname)
        if os.path.getsize(path) == 0:
            continue
        logs[os.path.splitext(fname)[0]] = parse_fifo_log(path)
    return logs


def fifo_log_is_consistent(log: Dict[str, Any]) -> None:
    """Validate number of transactions = reported summary.
    Args:
        log: parse_fifo_log() dict
    Raises:
        ValueError: on the first inconsistency found.
    """
    path = log["path"]

    def fail(msg, *args):
        raise ValueError("%s: %s" % (path, msg % args))

    # <direction> 0 means input, 1 means output
    ins = [t for t in log["txns"] if t[1] in [0, None]]
    outs = [t for t in log["txns"] if t[1] == 1]

    if len(ins) != log["in"]:
        fail("%d input lines but summary says in=%d", len(ins), log["in"])
    if log["maxfill"] > log["in"]:
        fail("MaxFill %d exceeds the %d words written", log["maxfill"], log["in"])

    if not log["verbose"]:
        # a plain log records inputs only, so every body line must be one
        if outs or len(log["txns"]) != log["in"]:
            fail("body has lines that are not inputs")
        return

    if len(outs) != log["out"]:
        fail("%d output lines but summary says out=%d", len(outs), log["out"])
    for i, out in enumerate(outs):
        if out[0] != ins[i][0]:
            fail("output #%d is %#x, but input #%d was %#x", i, out[0], i, ins[i][0])
        if out[2] <= ins[i][2]:
            fail("output #%d at cycle %d precedes its input", i, out[2])
    cycles = [t[2] for t in log["txns"]]
    if cycles != sorted(cycles):
        fail("cycle column is not monotonic")
    if cycles and log["cycles"] < max(cycles):
        fail("a transaction is logged after the last cycle")


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
