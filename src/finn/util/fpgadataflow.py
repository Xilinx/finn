# Copyright (c) 2020, Xilinx
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

try:
    from pyverilator import PyVerilator
except ModuleNotFoundError:
    PyVerilator = None
from finn.util.basic import get_by_name, make_build_dir, get_rtlsim_trace_depth


class IPGenBuilder:
    """Builds the bash script to generate IP blocks using Vivado HLS."""

    def __init__(self):
        self.tcl_script = ""
        self.ipgen_path = ""
        self.code_gen_dir = ""
        self.ipgen_script = ""

    def append_tcl(self, tcl_script):
        """Sets member variable "tcl_script" to given tcl script."""
        self.tcl_script = tcl_script

    def set_ipgen_path(self, path):
        """Sets member variable ipgen_path to given path."""
        self.ipgen_path = path

    def build(self, code_gen_dir):
        """Builds the bash script with given parameters and saves it in given folder.
        To guarantee the generation in the correct folder the bash script contains a
        cd command."""
        self.code_gen_dir = code_gen_dir
        self.ipgen_script = str(self.code_gen_dir) + "/ipgen.sh"
        working_dir = os.environ["PWD"]
        f = open(self.ipgen_script, "w")
        f.write("#!/bin/bash \n")
        f.write("cd {}\n".format(code_gen_dir))
        f.write("vivado_hls {}\n".format(self.tcl_script))
        f.write("cd {}\n".format(working_dir))
        f.close()
        bash_command = ["bash", self.ipgen_script]
        process_compile = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
        process_compile.communicate()


def pyverilate_stitched_ip(model):
    "Given a model with stitched IP, return a PyVerilator sim object."
    if PyVerilator is None:
        raise ImportError("Installation of PyVerilator is required.")

    vivado_stitch_proj_dir = model.get_metadata_prop("vivado_stitch_proj")
    with open(vivado_stitch_proj_dir + "/all_verilog_srcs.txt", "r") as f:
        all_verilog_srcs = f.read().split()

    def file_to_dir(x):
        return os.path.dirname(os.path.realpath(x))

    all_verilog_dirs = list(map(file_to_dir, all_verilog_srcs))
    top_verilog = model.get_metadata_prop("wrapper_filename")
    build_dir = make_build_dir("pyverilator_ipstitched_")
    sim = PyVerilator.build(
        top_verilog,
        verilog_path=all_verilog_dirs,
        build_dir=build_dir,
        trace_depth=get_rtlsim_trace_depth(),
    )
    return sim


def pyverilate_get_liveness_threshold_cycles():
    """Return the number of no-output cycles rtlsim will wait before assuming
    the simulation is not finishing and throwing an exception."""

    return int(os.getenv("LIVENESS_THRESHOLD", 10000))


def is_fpgadataflow_node(node):
    """Returns True if given node is fpgadataflow node. Otherwise False."""
    is_node = False
    if node is not None:
        if node.domain == "finn":
            n_backend = get_by_name(node.attribute, "backend")
            if n_backend is not None:
                backend_value = n_backend.s.decode("UTF-8")
                if backend_value == "fpgadataflow":
                    is_node = True

    return is_node
