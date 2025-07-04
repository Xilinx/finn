#############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @brief	rtlsim_multi_io interface adapter for FINN XSI
# @author	Yaman Umuroglu <yaman.umuroglu@amd.com>
#############################################################################

import errno
import os
import os.path
from finn_xsi.sim_engine import SimEngine
from typing import Optional

from finn.util.basic import launch_process_helper


def locate_glbl() -> Optional[str]:
    """
    Tries to determine the glbl.v file path from environment variables.
    Returns None if it cannot be found.
    """
    # Get GLBL from the Vitis environment variable
    vivado_path = os.environ.get("XILINX_VIVADO")
    if vivado_path:
        glbl_path = os.path.join(vivado_path, "data", "verilog", "src", "glbl.v")
        if os.path.isfile(glbl_path):
            return glbl_path
    return None


def compile_sim_obj(top_module_name, source_list, sim_out_dir, debug=False):
    # create a .prj file with the source files
    with open(sim_out_dir + "/rtlsim.prj", "w") as f:
        glbl = locate_glbl()
        if glbl is not None:
            f.write(f"verilog work {glbl}\n")

        # extract (unique, by using a set) verilog headers for inclusion
        verilog_headers = {os.path.dirname(x) for x in source_list if x.endswith(".vh")}
        verilog_header_incl_str = " ".join(["--include " + x for x in verilog_headers])

        for src_line in source_list:
            if src_line.endswith(".v"):
                f.write(f"verilog work {verilog_header_incl_str} {src_line}\n")
            elif src_line.endswith(".vhd"):
                # note that Verilog header incls are not added for VHDL
                f.write(f"vhdl2008 work {src_line}\n")
            elif src_line.endswith(".sv"):
                f.write(f"sv work {verilog_header_incl_str} {src_line}\n")
            elif src_line.endswith(".vh"):
                # skip adding Verilog headers directly (see verilog_header_incl_str)
                continue
            else:
                raise Exception(f"Unknown extension for .prj file sources: {src_line}")

    # now call xelab to generate the .so for the design to be simulated
    # list of libs for xelab retrieved from Vitis HLS cosim cmdline
    # the particular lib version used depends on the Vivado/Vitis version being used
    # but putting in multiple (nonpresent) versions seems to pose no problem as long
    # as the correct one is also in there. at least this is how Vitis HLS cosim is
    # handling it.
    # TODO make this an optional param instead of hardcoding
    xelab_libs = [
        "smartconnect_v1_0",
        "axi_protocol_checker_v1_1_12",
        "axi_protocol_checker_v1_1_13",
        "axis_protocol_checker_v1_1_11",
        "axis_protocol_checker_v1_1_12",
        "xil_defaultlib",
        "unisims_ver",
        "xpm",
        "floating_point_v7_1_16",
        "floating_point_v7_0_21",
        "floating_point_v7_1_18",
        "floating_point_v7_1_15",
        "floating_point_v7_1_19",
    ]

    cmd_xelab = [
        "xelab",
        "work." + top_module_name,
        "-relax",
        "-prj",
        "rtlsim.prj",
        "-dll",
        "-s",
        top_module_name,
    ]
    # Add debug flag if debug is enabled
    if debug:
        cmd_xelab.append("-debug")
        cmd_xelab.append("all")
    for lib in xelab_libs:
        cmd_xelab.append("-L")
        cmd_xelab.append(lib)

    if locate_glbl() is not None:
        cmd_xelab.insert(1, "work.glbl")

    launch_process_helper(cmd_xelab, cwd=sim_out_dir)
    out_so_relative_path = "xsim.dir/%s/xsimk.so" % top_module_name
    out_so_full_path = sim_out_dir + "/" + out_so_relative_path

    if not os.path.isfile(out_so_full_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), out_so_full_path)

    return (sim_out_dir, out_so_relative_path)


def get_simkernel_so():
    vivado_path = os.environ.get("XILINX_VIVADO")
    # xsi kernel lib name depends on Vivado version (renamed in 2024.2)
    try:
        if vivado_path:
            if float(vivado_path.split("/")[-1]) > 2024.1:
                simkernel_so = "libxv_simulator_kernel.so"
            else:
                simkernel_so = "librdi_simulator_kernel.so"
        else:
            simkernel_so = "librdi_simulator_kernel.so"
    except Exception:
        # fallback/default
        simkernel_so = "librdi_simulator_kernel.so"
    return simkernel_so


def load_sim_obj(sim_out_dir, out_so_relative_path, tracefile=None, simkernel_so=None):
    if simkernel_so is None:
        simkernel_so = get_simkernel_so()
    oldcwd = os.getcwd()
    os.chdir(sim_out_dir)
    sim = SimEngine(simkernel_so, out_so_relative_path, "finnxsi_rtlsim.log", tracefile)
    if tracefile:
        sim.top.trace_all()
    os.chdir(oldcwd)
    return sim


def reset_rtlsim(
    sim, rst_name="ap_rst_n", active_low=True, clk_name="ap_clk", clk2x_name="ap_clk2x", n_cycles=16
):
    sim.do_reset()
    sim.run()


def close_rtlsim(sim):
    del sim


def rtlsim_multi_io(
    sim,
    io_dict,
    num_out_values,
    sname="_V_V",
    liveness_threshold=10000,
):
    if len(io_dict["outputs"]) > 1:
        assert isinstance(
            num_out_values, dict
        ), "num_out_values must be dict for multiple output streams"
    else:
        # num_out_values is provided as integer (indicating the expected
        # outputs from the single output stream) - make into dict
        oname = list(io_dict["outputs"].keys())[0]
        num_out_values = {oname: num_out_values}
    # FINN XSI expects hex strings, while rtlsim_multi_io uses
    # lists of arbitrary-precision integers, so need to convert
    # inputs and outputs to appropriate format
    # TODO: refactor components&data packing to directly generate and consume
    # hex strings instead of arb-prec Python integers
    for inp in io_dict["inputs"]:
        arbprec_int_input = io_dict["inputs"][inp]
        hexstring_input = map(lambda var: f"{var:0x}", arbprec_int_input)
        stream_name = inp + sname
        sim.stream_input(stream_name, hexstring_input)

    hex_output_streams = {}
    for out in io_dict["outputs"]:
        stream_name = out + sname
        hex_output_streams[out] = sim.collect_output(
            stream_name,
            num_out_values[out],
            watchdog=sim.create_watchdog(f"{stream_name} timeout", liveness_threshold),
        )

    start_ticks = sim.ticks
    ret = sim.run()
    if len(ret) > 0:
        assert False, f"RTL simulation watchdogs {str(ret)} timed out. Check rtlsim_trace if any."
    end_ticks = sim.ticks
    for out in io_dict["outputs"]:
        io_dict["outputs"][out] = list(map(lambda var: int(var, base=16), hex_output_streams[out]))

    return end_ticks - start_ticks
