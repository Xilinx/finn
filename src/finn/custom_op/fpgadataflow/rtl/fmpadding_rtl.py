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
import numpy as np
import os
import shutil
from qonnx.util.basic import roundup_to_integer_multiple

from finn.custom_op.fpgadataflow.fmpadding import FMPadding
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.util.basic import get_rtlsim_trace_depth, make_build_dir
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy

try:
    from pyverilator import PyVerilator
except ModuleNotFoundError:
    PyVerilator = None


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

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

        if mode == "cppsim":
            FMPadding.execute_node(self, context, graph)
        elif mode == "rtlsim":
            node = self.onnx_node
            exp_ishape = self.get_normal_input_shape()
            exp_oshape = self.get_normal_output_shape()
            folded_ishape = self.get_folded_input_shape()
            inp = context[node.input[0]]
            assert str(inp.dtype) == "float32", "Input datatype is not float32"
            assert (
                inp.shape == exp_ishape
            ), """Input shape doesn't
            match expected shape (1, ImgDim_h, ImgDim_w, NumChannels)."""
            export_idt = self.get_input_datatype()

            reshaped_input = inp.reshape(folded_ishape)
            np.save(os.path.join(code_gen_dir, "input_0.npy"), reshaped_input)

            sim = self.get_rtlsim()
            nbits = self.get_instream_width()
            rtlsim_inp = npy_to_rtlsim_input(
                "{}/input_0.npy".format(code_gen_dir), export_idt, nbits
            )
            super().reset_rtlsim(sim)
            super().toggle_clk(sim)
            rtlsim_output = self.rtlsim(sim, rtlsim_inp)
            odt = export_idt
            target_bits = odt.bitwidth()
            packed_bits = self.get_outstream_width()
            out_npy_path = "{}/output.npy".format(code_gen_dir)
            out_shape = self.get_folded_output_shape()
            rtlsim_output_to_npy(
                rtlsim_output, out_npy_path, odt, out_shape, packed_bits, target_bits
            )
            # load and reshape output
            output = np.load(out_npy_path)
            output = np.asarray([output], dtype=np.float32).reshape(*exp_oshape)
            context[node.output[0]] = output

            assert (
                context[node.output[0]].shape == exp_oshape
            ), """Output shape doesn't match expected shape
                (1, OutputDim_H, OutputDim_W, NumChannels)."""

        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )

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

    def prepare_rtlsim(self):
        """Creates a Verilator emulation library for the RTL code generated
        for this node, sets the rtlsim_so attribute to its path and returns
        a PyVerilator wrapper around it."""
        # Modified to use generated (System-)Verilog instead of HLS output products

        if PyVerilator is None:
            raise ImportError("Installation of PyVerilator is required.")

        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        verilog_paths = [code_gen_dir]
        verilog_files = [
            "fmpadding_axi.sv",
            "fmpadding.sv",
            "axi2we.sv",
            self.get_nodeattr("gen_top_module") + ".v",
        ]

        # build the Verilator emu library
        sim = PyVerilator.build(
            verilog_files,
            build_dir=make_build_dir("pyverilator_" + self.onnx_node.name + "_"),
            verilog_path=verilog_paths,
            trace_depth=get_rtlsim_trace_depth(),
            top_module_name=self.get_verilog_top_module_name(),
        )
        # save generated lib filename in attribute
        self.set_nodeattr("rtlsim_so", sim.lib._name)
        return sim

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
