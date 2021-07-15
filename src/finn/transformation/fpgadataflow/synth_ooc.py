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
from shutil import copy2

from finn.transformation.base import Transformation
from finn.util.basic import make_build_dir
from finn.util.vivado import out_of_context_synth


class SynthOutOfContext(Transformation):
    """Run out-of-context Vivado synthesis on a stitched IP design."""

    def __init__(self, part, clk_period_ns, clk_name="ap_clk"):
        super().__init__()
        self.part = part
        self.clk_period_ns = clk_period_ns
        self.clk_name = clk_name

    def apply(self, model):
        def file_to_basename(x):
            return os.path.basename(os.path.realpath(x))

        vivado_stitch_proj_dir = model.get_metadata_prop("vivado_stitch_proj")
        assert vivado_stitch_proj_dir is not None, "Need stitched IP to run."
        top_module_name = model.get_metadata_prop("wrapper_filename")
        top_module_name = file_to_basename(top_module_name).strip(".v")
        build_dir = make_build_dir("synth_out_of_context_")
        verilog_extensions = [".v", ".vh"]
        with open(vivado_stitch_proj_dir + "/all_verilog_srcs.txt", "r") as f:
            all_verilog_srcs = f.read().split()
        for file in all_verilog_srcs:
            if any([file.endswith(x) for x in verilog_extensions]):
                copy2(file, build_dir)
        ret = out_of_context_synth(
            build_dir, top_module_name, self.part, self.clk_name, self.clk_period_ns
        )
        model.set_metadata_prop("res_total_ooc_synth", str(ret))
        return (model, False)
