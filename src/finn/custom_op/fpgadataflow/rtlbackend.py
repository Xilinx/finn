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

import os
import subprocess
from abc import ABC, abstractmethod

from finn.custom_op.fpgadataflow import templates


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

    @abstractmethod
    def prepare_rtlsim(self):
        pass

    def code_generation_ipi(self):
        """Constructs and returns the TCL for node instantiation in Vivado IPI."""
        vlnv = self.get_nodeattr("ip_vlnv")
        cmd = ["create_bd_cell -type ip -vlnv %s %s" % (vlnv, self.onnx_node.name)]
        return cmd

    def code_generation_pack_ip(self, fpgapart):
        """Pack RTL as IP"""
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        # prepare the IP packaging tcl template
        template = templates.ip_package_tcl
        self.code_gen_dict.clear()
        self.code_gen_dict["$TOPNAME$"] = [self.get_nodeattr("gen_top_module")]
        self.code_gen_dict["$PART$"] = [fpgapart]
        # note: setting the root dir as absolute can cause path problems
        # the ipgen script will be invoked from the sources dir so root_dir=. is OK
        self.code_gen_dict["$VERILOG_DIR$"] = ["."]
        for key in self.code_gen_dict:
            # transform list into long string separated by '\n'
            code_gen_line = "\n".join(self.code_gen_dict[key])
            template = template.replace(key, code_gen_line)
        f = open(os.path.join(code_gen_dir, "package_ip.tcl"), "w")
        f.write(template)
        f.close()
        # create a shell script and call Vivado to invoke the IP pkg script
        make_project_sh = code_gen_dir + "/make_ip.sh"
        working_dir = os.environ["PWD"]
        with open(make_project_sh, "w") as f:
            f.write("#!/bin/bash \n")
            f.write("cd {}\n".format(code_gen_dir))
            f.write("vivado -mode batch -source package_ip.tcl\n")
            f.write("cd {}\n".format(working_dir))
        bash_command = ["bash", make_project_sh]
        process_compile = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
        process_compile.communicate()
        # set ipgen_path and ip_path to point to the new packaged IP
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)
        vlnv = "xilinx.com:hls:%s:1.0" % (self.get_nodeattr("gen_top_module"))
        self.set_nodeattr("ip_vlnv", vlnv)
        self.code_gen_dict.clear()

    def code_generation_ipgen(self, model, fpgapart, clk):
        self.generate_hdl(model, fpgapart, clk)
        self.code_generation_pack_ip(fpgapart)

    # TODO: Implement alternative
    def hls_sname(self):
        """Get the naming convention used by Vitis HLS for stream signals
        Example: the TDATA for a stream called "out" would be out_V_TDATA.
        """
        return "V"
