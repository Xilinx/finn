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

import math
import os
import shutil
from qonnx.util.basic import roundup_to_integer_multiple

from finn.custom_op.fpgadataflow.fmpadding import FMPadding
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend


class FMPadding_rtl(FMPadding, RTLBackend):
    """CustomOp wrapper for the finn-rtllib fmpadding_axi component
    Supports adjusting the padding amount and spatial feature sizes at
    runtime."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            # Enable reprogrammable implementation to change FM dimensions,
            # stride, or dilation during runtime
            "dynamic_mode": ("i", False, 0, {0, 1}),
        }
        my_attrs.update(FMPadding.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    def get_verilog_top_module_intf_names(self):
        # Overload default HLSCustomOp implementation to add axilite control IF
        intf_names = super().get_verilog_top_module_intf_names()
        if self.get_nodeattr("dynamic_mode"):
            intf_names["axilite"] = ["s_axilite"]
        return intf_names

    def get_template_values(self, ifm_dims, pads, chans, simd, idt):
        dimY, dimX = ifm_dims
        padT, padL, padB, padR = pads
        y_counter_bits = int(math.ceil(math.log2(padT + dimY + padB + 1)))
        x_counter_bits = int(math.ceil(math.log2(padL + dimX + padR + 1)))
        topname = self.get_verilog_top_module_name()
        stream_bits = idt.bitwidth() * simd
        stream_bits = int(roundup_to_integer_multiple(stream_bits, 8))
        code_gen_dict = {
            "XCOUNTER_BITS": int(x_counter_bits),
            "YCOUNTER_BITS": int(y_counter_bits),
            "NUM_CHANNELS": int(chans),
            "SIMD": int(simd),
            "ELEM_BITS": idt.bitwidth(),
            "TOP_MODULE_NAME": topname,
            "INIT_XON": int(padL),
            "INIT_XOFF": int(padL + dimX),
            "INIT_XEND": int(padL + dimX + padR - 1),
            "INIT_YON": int(padT),
            "INIT_YOFF": int(padT + dimY),
            "INIT_YEND": int(padT + dimY + padB - 1),
            "STREAM_BITS": int(stream_bits),
        }
        return code_gen_dict

    def get_dynamic_config(self, ifm_dims=None, pads=None):
        """Returns a configuration dict to re-configure FM dimension and
        padding amounts during runtime."""

        if ifm_dims is None:
            ifm_dims = self.get_nodeattr("ImgDim")
        if pads is None:
            pads = self.get_nodeattr("Padding")
        chans = self.get_nodeattr("NumChannels")
        simd = self.get_nodeattr("SIMD")
        idt = self.get_input_datatype()
        code_gen_dict = self.get_template_values(ifm_dims, pads, chans, simd, idt)
        config = {
            "XON": (0 * 4, (code_gen_dict["INIT_XON"])),
            "XOFF": (1 * 4, (code_gen_dict["INIT_XOFF"])),
            "XEND": (2 * 4, (code_gen_dict["INIT_XEND"])),
            "YON": (3 * 4, (code_gen_dict["INIT_YON"])),
            "YOFF": (4 * 4, (code_gen_dict["INIT_YOFF"])),
            "YEND": (5 * 4, (code_gen_dict["INIT_YEND"])),
        }
        return config

    def generate_hdl(self, model, fpgapart, clk):
        rtlsrc = os.environ["FINN_ROOT"] + "/finn-rtllib/fmpadding/hdl"
        template_path = rtlsrc + "/fmpadding_template.v"
        dims = self.get_nodeattr("ImgDim")
        pads = self.get_nodeattr("Padding")
        chans = self.get_nodeattr("NumChannels")
        simd = self.get_nodeattr("SIMD")
        idt = self.get_input_datatype()
        code_gen_dict = self.get_template_values(dims, pads, chans, simd, idt)
        # save top module name so we can refer to it after this node has been renamed
        # (e.g. by GiveUniqueNodeNames(prefix) during MakeZynqProject)
        self.set_nodeattr("gen_top_module", self.get_verilog_top_module_name())

        # apply code generation to templates
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        with open(template_path, "r") as f:
            template = f.read()
        for key_name in code_gen_dict:
            key = "$%s$" % key_name
            template = template.replace(key, str(code_gen_dict[key_name]))

        with open(
            os.path.join(code_gen_dir, self.get_verilog_top_module_name() + ".v"),
            "w",
        ) as f:
            f.write(template)

        sv_files = ["fmpadding_axi.sv", "fmpadding.sv", "axi2we.sv"]
        for sv_file in sv_files:
            shutil.copy(rtlsrc + "/" + sv_file, code_gen_dir)
        # set ipgen_path and ip_path so that HLS-Synth transformation
        # and stich_ip transformation do not complain
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def get_rtl_file_list(self, abspath=False):
        if abspath:
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen") + "/"
            rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/fmpadding/hdl/")
        else:
            code_gen_dir = ""
            rtllib_dir = ""

        verilog_files = [
            rtllib_dir + "fmpadding_axi.sv",
            rtllib_dir + "fmpadding.sv",
            rtllib_dir + "axi2we.sv",
            code_gen_dir + self.get_nodeattr("gen_top_module") + ".v",
        ]
        return verilog_files

    def code_generation_ipi(self):
        """Constructs and returns the TCL for node instantiation in Vivado IPI."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

        sourcefiles = [
            "fmpadding_axi.sv",
            "fmpadding.sv",
            "axi2we.sv",
            self.get_nodeattr("gen_top_module") + ".v",
        ]

        sourcefiles = [os.path.join(code_gen_dir, f) for f in sourcefiles]

        cmd = []
        for f in sourcefiles:
            cmd += ["add_files -norecurse %s" % (f)]
        cmd += [
            "create_bd_cell -type module -reference %s %s"
            % (self.get_nodeattr("gen_top_module"), self.onnx_node.name)
        ]
        return cmd

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        if mode == "cppsim":
            FMPadding.execute_node(self, context, graph)
        elif mode == "rtlsim":
            RTLBackend.execute_node(self, context, graph)
