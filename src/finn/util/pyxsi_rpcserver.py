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
from xmlrpc.server import SimpleXMLRPCRequestHandler, SimpleXMLRPCServer

# since simulation with XSI requires a certain LD_LIBRARY_PATH setting
# which breaks other applications, we launch the simulation in its
# own executable with this env.var. setting, and use xmlrpc to access it

try:
    ldlp = os.environ["LD_LIBRARY_PATH"]
    if not ("Vivado" in ldlp):
        assert False, "Must be launched with LD_LIBRARY_PATH=$(XILINX_VIVADO)/lib/lnx64.o"
except KeyError:
    assert False, "Must be launched with LD_LIBRARY_PATH=$(XILINX_VIVADO)/lib/lnx64.o"


class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ("/RPC2",)


# we need to do some conversions while pyxsi calls are going through xmlrpc:
# * sim objs become strings (stored in the sim_id_to_obj dict until done)
# * signal values become strings
#   (converted back and forth to Python integers)

sim_id_to_obj = {}


def compile_sim_obj(top_module_name, source_list, sim_out_dir):
    ret = pyxsi_utils.compile_sim_obj(top_module_name, source_list, sim_out_dir)
    return ret


def load_sim_obj(sim_out_dir, out_so_relative_path, tracefile=None, is_toplevel_verilog=True):
    ret_sim_obj = pyxsi_utils.load_sim_obj(
        sim_out_dir, out_so_relative_path, tracefile, is_toplevel_verilog
    )
    ret_sim_id = str(id(ret_sim_obj))
    sim_id_to_obj[ret_sim_id] = ret_sim_obj
    return ret_sim_id


def find_signal(sim_id, signal_name):
    sim = sim_id_to_obj[sim_id]
    return pyxsi_utils._find_signal(sim, signal_name)


def read_signal(sim_id, signal_name):
    sim = sim_id_to_obj[sim_id]
    signal_value = pyxsi_utils._read_signal(sim, signal_name)
    signal_value_str = str(signal_value)
    return signal_value_str


def write_signal(sim_id, signal_name, signal_value_str):
    sim = sim_id_to_obj[sim_id]
    signal_value = int(signal_value_str)
    pyxsi_utils._write_signal(sim, signal_name, signal_value)


def reset_rtlsim(sim_id, rst_name, active_low, clk_name):
    sim = sim_id_to_obj[sim_id]
    pyxsi_utils.reset_rtlsim(sim, rst_name, active_low, clk_name)


def toggle_clk(sim_id, clk_name):
    sim = sim_id_to_obj[sim_id]
    pyxsi_utils.toggle_clk(sim, clk_name)


def toggle_neg_edge(sim_id, clk_name):
    sim = sim_id_to_obj[sim_id]
    pyxsi_utils.toggle_neg_edge(sim, clk_name)


def toggle_pos_edge(sim_id, clk_name):
    sim = sim_id_to_obj[sim_id]
    pyxsi_utils.toggle_pos_edge(sim, clk_name)


# ask to create server on port 0 to find an available port
with SimpleXMLRPCServer(("localhost", 0), requestHandler=RequestHandler, allow_none=True) as server:
    port = server.server_address[1]
    server.register_introspection_functions()
    server.register_function(compile_sim_obj)
    server.register_function(load_sim_obj)
    server.register_function(find_signal)
    server.register_function(read_signal)
    server.register_function(write_signal)
    server.register_function(reset_rtlsim)
    server.register_function(toggle_clk)
    server.register_function(toggle_neg_edge)
    server.register_function(toggle_pos_edge)
    print(f"pyxsi RPC server is now running on {port}")
    server.serve_forever()
