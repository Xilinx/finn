# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Synthesis check that the RTL memories of finn-rtllib map to memory primitives.

A memory that Vivado does not recognize as such is built from registers: it
simulates correctly and costs many times the logic, which no functional test
notices. Each case synthesizes one memory out of context at a size that has to
map to RAM.
"""

import pytest

import os

from finn.util.basic import make_build_dir
from finn.util.test import assert_memory_inferred
from finn.util.vivado import synth_ooc_utilization

PART = "xczu7ev-ffvc1156-2-e"  # UltraScale+ with URAM


def rtllib(*path):
    return os.path.join(os.environ["FINN_ROOT"], "finn-rtllib", *path)


# input_gen takes array parameters, which synth_design cannot override: a
# 32 x 32 transposition of 32-bit elements buffers the whole map.
INPUT_GEN_WRAPPER = """
module input_gen_top #(parameter RAM_STYLE = "auto")(
	input	logic  clk, rst,
	input	logic [31:0]  idat, input logic  ivld, output logic  irdy,
	output	logic [31:0]  odat, output logic  ovld, output logic [1:0]  olst, odone,
	input	logic  ordy
);
	input_gen #(
		.DATA_WIDTH(32), .FM_SIZE(1024), .D(2), .DIMS('{32, 32}), .COEFS('{1, 32}),
		.RAM_STYLE(RAM_STYLE)
	) impl (.*);
endmodule
"""

# (name, top, sources, generics, ram styles, memory bits)
CASES = [
    (
        "memstream",
        "memstream",
        [rtllib("memstream", "hdl", "memstream.sv")],
        {"SETS": 1, "DEPTH": 1024, "WIDTH": 32},
        ["auto", "block", "distributed", "ultra"],
        1024 * 32,
    ),
    (
        "fifo",
        "fifo",
        [rtllib("fifo", "hdl", "fifo.sv")],
        {"DEPTH": 1024, "DATA_WIDTH": 32},
        ["auto", "block", "distributed", "ultra"],
        1024 * 32,
    ),
    (
        "elasticmem",
        "elasticmem",
        [rtllib("inner_shuffle", "elasticmem.sv"), rtllib("fifo", "hdl", "fifo.sv")],
        {"WIDTH": 32, "DEPTH": 1024},
        ["auto", "block", "distributed"],
        1024 * 32,
    ),
    (
        "input_gen",
        "input_gen_top",
        [rtllib("input_gen", "hdl", "input_gen.sv")],
        {},
        ["auto", "block", "distributed"],
        1024 * 32,
    ),
    (
        "thresholding",
        "thresholding",
        [rtllib("thresholding", "hdl", "thresholding.sv")],
        # runtime-writable, so the stage memories cannot fold into logic
        {"K": 16, "N": 255, "C": 64, "PE": 4, "USE_CONFIG": 1, "DEPTH_TRIGGER_BRAM": 512},
        ["block", "distributed"],
        64 * 255 * 16,
    ),
    (
        "dynamic_load",
        "dynamic_load",
        [rtllib("dynload", "hdl", "dynamic_load.sv")],
        {"PE": 4, "SIMD": 4, "WEIGHT_WIDTH": 8, "MH": 64, "MW": 64, "N_REPS": 1},
        ["block", "distributed"],
        64 * 64 * 8,
    ),
    (
        "local_weight_buffer",
        "local_weight_buffer",
        [rtllib("fetch_weights", "local_weight_buffer.sv")],
        {"PE": 4, "SIMD": 4, "WEIGHT_WIDTH": 8, "MH": 64, "MW": 64, "N_REPS": 1},
        ["block", "distributed"],
        64 * 64 * 8,
    ),
]

PARAMS = [(c, s) for c in CASES for s in c[4]]


@pytest.mark.parametrize("case,ram_style", PARAMS, ids=[f"{c[0]}-{s}" for c, s in PARAMS])
@pytest.mark.vivado
@pytest.mark.slow
def test_rtl_memory_inference(case, ram_style):
    name, top, sources, generics, _, bits = case
    build_dir = make_build_dir(f"test_rtl_memory_inference_{name}_{ram_style}_")
    sources = list(sources)
    if top == "input_gen_top":
        wrapper = os.path.join(build_dir, "input_gen_top.sv")
        with open(wrapper, "w") as f:
            f.write(INPUT_GEN_WRAPPER)
        sources.append(wrapper)
    generics = dict(generics)
    if name == "thresholding":
        # the depth trigger selects the storage; ram_style names what to expect
        pass
    else:
        generics["RAM_STYLE"] = ram_style
    util = synth_ooc_utilization(top, sources, PART, build_dir, generics)
    assert_memory_inferred(util, ram_style, bits)
