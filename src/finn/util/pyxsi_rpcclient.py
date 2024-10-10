# Copyright (C) 2024, Advanced Micro Devices, Inc.
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
# * Neither the name of pyxsi nor the names of its
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

import os
import pyxsi_utils
import subprocess
import xmlrpc.client
from time import sleep

from finn.util.basic import get_finn_root, get_vivado_root


def compile_sim_obj(top_module_name, source_list, sim_out_dir):
    # compile_sim_obj does not require special envvar settings and is safe to call
    # directly without any RPC
    ret = pyxsi_utils.compile_sim_obj(top_module_name, source_list, sim_out_dir)
    return ret


def load_sim_obj(sim_out_dir, out_so_relative_path, tracefile=None, is_toplevel_verilog=True):
    # launch a pyxsi RPC server
    proc_env = os.environ.copy()
    proc_env["LD_LIBRARY_PATH"] = get_vivado_root() + "/lib/lnx64.o"
    logfile_wr_fd = open(sim_out_dir + "/pyxsi_rpcserver.log", "w")
    logfile_rd_fd = open(sim_out_dir + "/pyxsi_rpcserver.log", "r")
    command = ["python", "-u", get_finn_root() + "/src/finn/util/pyxsi_rpcserver.py"]
    proc = subprocess.Popen(
        command,
        bufsize=1,
        env=proc_env,
        stdout=logfile_wr_fd,
        stderr=logfile_wr_fd,
        universal_newlines=True,
    )
    rpc_port = 8000
    # TODO sleep to ensure RPC server has started before trying to read its port number from stdout
    # bit hacky - is there a better way of communicating the open port number back to the client?
    line = logfile_rd_fd.readline()
    retries = 10
    while line == "" and retries > 0:
        sleep(0.1)
        line = logfile_rd_fd.readline()
        retries -= 1
    if "pyxsi RPC server is now running on" in line:
        rpc_port = int(line.split(" on ")[1])
        logfile_rd_fd.close()
    else:
        assert False, f"Unexpected output from pyxsi RPC server: {line}"
    rpc_proxy = xmlrpc.client.ServerProxy(f"http://localhost:{rpc_port}", allow_none=True)
    sim_id = rpc_proxy.load_sim_obj(
        sim_out_dir, out_so_relative_path, tracefile, is_toplevel_verilog
    )
    # return all relevant information for rtlsim
    handle = (sim_id, rpc_proxy, rpc_port, proc)
    return handle


def close_rtlsim(handle):
    (sim_id, rpc_proxy, _, proc) = handle
    rpc_proxy.close_rtlsim(sim_id)
    proc.terminate()


def find_signal(handle, signal_name):
    (sim_id, rpc_proxy, _, _) = handle
    ret = rpc_proxy.find_signal(sim_id, signal_name)
    return ret


def read_signal(handle, signal_name):
    (sim_id, rpc_proxy, _, _) = handle
    signal_value_str = rpc_proxy.read_signal(sim_id, signal_name)
    signal_value = int(signal_value_str)
    return signal_value


def write_signal(handle, signal_name, signal_value):
    (sim_id, rpc_proxy, _, _) = handle
    signal_value_str = str(signal_value)
    rpc_proxy.write_signal(sim_id, signal_name, signal_value_str)


def reset_rtlsim(handle, rst_name="ap_rst_n", active_low=True, clk_name="ap_clk"):
    (sim_id, rpc_proxy, _, _) = handle
    rpc_proxy.reset_rtlsim(sim_id, rst_name, active_low, clk_name)


def toggle_clk(handle, clk_name="ap_clk"):
    (sim_id, rpc_proxy, _, _) = handle
    rpc_proxy.toggle_clk(sim_id, clk_name)


def toggle_neg_edge(handle, clk_name="ap_clk"):
    (sim_id, rpc_proxy, _, _) = handle
    rpc_proxy.toggle_neg_edge(sim_id, clk_name)


def toggle_pos_edge(handle, clk_name="ap_clk"):
    (sim_id, rpc_proxy, _, _) = handle
    rpc_proxy.toggle_pos_edge(sim_id, clk_name)


def rtlsim_multi_io(
    handle,
    io_dict,
    num_out_values,
    sname="_V_V_",
    liveness_threshold=10000,
    hook_preclk=None,
    hook_postclk=None,
):
    for outp in io_dict["outputs"]:
        write_signal(handle, outp + sname + "TREADY", 1)

    # observe if output is completely calculated
    # total_cycle_count will contain the number of cycles the calculation ran
    output_done = False
    total_cycle_count = 0
    output_count = 0
    old_output_count = 0

    # avoid infinite looping of simulation by aborting when there is no change in
    # output values after 100 cycles
    no_change_count = 0

    while not (output_done):
        signals_to_write = {}
        if hook_preclk:
            hook_preclk(handle)
        # Toggle falling edge to arrive at a delta cycle before the rising edge
        toggle_neg_edge(handle)

        # examine signals, decide how to act based on that but don't update yet
        # so only read_signal access in this block, no _write_signal
        for inp in io_dict["inputs"]:
            inputs = io_dict["inputs"][inp]
            signal_name = inp + sname
            if (
                read_signal(handle, signal_name + "TREADY") == 1
                and read_signal(handle, signal_name + "TVALID") == 1
            ):
                inputs = inputs[1:]
            io_dict["inputs"][inp] = inputs

        for outp in io_dict["outputs"]:
            outputs = io_dict["outputs"][outp]
            signal_name = outp + sname
            if (
                read_signal(handle, signal_name + "TREADY") == 1
                and read_signal(handle, signal_name + "TVALID") == 1
            ):
                outputs = outputs + [read_signal(handle, signal_name + "TDATA")]
                output_count += 1
            io_dict["outputs"][outp] = outputs

        # update signals based on decisions in previous block, but don't examine anything
        # so only write_signal access in this block, no read_signal
        for inp in io_dict["inputs"]:
            inputs = io_dict["inputs"][inp]
            signal_name = inp + sname
            signals_to_write[signal_name + "TVALID"] = 1 if len(inputs) > 0 else 0
            signals_to_write[signal_name + "TDATA"] = inputs[0] if len(inputs) > 0 else 0

        # Toggle rising edge to arrive at a delta cycle before the falling edge
        toggle_pos_edge(handle)

        for k, v in signals_to_write.items():
            write_signal(handle, k, v)

        if hook_postclk:
            hook_postclk(handle)

        total_cycle_count = total_cycle_count + 1

        if output_count == old_output_count:
            no_change_count = no_change_count + 1
        else:
            no_change_count = 0
            old_output_count = output_count

        # check if all expected output words received
        if output_count == num_out_values:
            output_done = True

        # end sim on timeout
        if no_change_count == liveness_threshold:
            raise Exception(
                "Error in simulation! Takes too long to produce output. "
                "Consider setting the liveness_threshold parameter to a "
                "larger value."
            )

    return total_cycle_count
