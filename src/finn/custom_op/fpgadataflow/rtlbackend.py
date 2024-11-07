# Copyright (C) 2023, Advanced Micro Devices, Inc.
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

from abc import ABC, abstractmethod

from finn.util import pyxsi_rpcclient
from finn.util.basic import get_rtlsim_trace_depth, make_build_dir

try:
    from pyverilator import PyVerilator
except ModuleNotFoundError:
    PyVerilator = None


class RTLBackend(ABC):
    """RTLBackend class all custom ops that correspond to a module in finn-rtllib
    are using functionality of. Contains different functions every RTL
    custom node should have. Some as abstract methods, these have to be filled
    when writing a new RTL custom op node."""

    def get_nodeattr_types(self):
        return {
            # attribute to save top module name - not user configurable
            "gen_top_module": ("s", False, ""),
        }

    @abstractmethod
    def generate_hdl(self, model, fpgapart, clk):
        pass

    def prepare_rtlsim(self):
        """Creates a Verilator emulation library for the RTL code generated
        for this node, sets the rtlsim_so attribute to its path and returns
        a PyVerilator wrapper around it."""

        if PyVerilator is None:
            raise ImportError("Installation of PyVerilator is required.")

        verilog_paths = self.get_verilog_paths()
        rtlsim_backend = self.get_nodeattr("rtlsim_backend")
        if rtlsim_backend == "pyverilator":
            if PyVerilator is None:
                raise ImportError("Installation of PyVerilator is required.")
            verilog_files = self.get_rtl_file_list(abspath=False)

            # build the Verilator emu library
            sim = PyVerilator.build(
                verilog_files,
                build_dir=make_build_dir("pyverilator_" + self.onnx_node.name + "_"),
                verilog_path=verilog_paths,
                trace_depth=get_rtlsim_trace_depth(),
                top_module_name=self.get_nodeattr("gen_top_module"),
            )
            # save generated lib filename in attribute
            self.set_nodeattr("rtlsim_so", sim.lib._name)
        elif rtlsim_backend == "pyxsi":
            verilog_files = self.get_rtl_file_list(abspath=True)
            single_src_dir = make_build_dir("rtlsim_" + self.onnx_node.name + "_")
            ret = pyxsi_rpcclient.compile_sim_obj(
                self.get_verilog_top_module_name(), verilog_files, single_src_dir
            )
            # save generated lib filename in attribute
            self.set_nodeattr("rtlsim_so", ret[0] + "/" + ret[1])
            # TODO return val of this function is never used
            # refactor s.t. it does not return anything at all,
            # consistently between pyverilator and pyxsi
            sim = None
        else:
            assert False, "Unknown rtlsim_backend"
        return sim

    def get_verilog_paths(self):
        """Returns path to code gen directory. Can be overwritten to
        return additional paths to relevant verilog files"""
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        return [code_gen_dir]

    @abstractmethod
    def get_rtl_file_list(self, abspath=False):
        """Returns list of rtl files. Needs to be filled by each node."""
        pass

    @abstractmethod
    def code_generation_ipi(self):
        pass

    def code_generation_ipgen(self, model, fpgapart, clk):
        self.generate_hdl(model, fpgapart, clk)

    # TODO: Implement alternative
    def hls_sname(self):
        """Get the naming convention used by Vitis HLS for stream signals
        Example: the TDATA for a stream called "out" would be out_V_TDATA.
        """
        return "V"
