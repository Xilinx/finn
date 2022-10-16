# Copyright (C) 2022, Advanced Micro Devices, Inc.
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
import warnings
from qonnx.core.datatype import DataType
from qonnx.util.basic import roundup_to_integer_multiple

from finn.custom_op.fpgadataflow.hlscustomop import HLSCustomOp
from finn.util.basic import get_rtlsim_trace_depth, make_build_dir
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy

try:
    from pyverilator import PyVerilator
except ModuleNotFoundError:
    PyVerilator = None


class FMPadding_rtl(HLSCustomOp):
    """CustomOp wrapper for the finn-rtllib fmpadding_axi component
    Supports adjusting the padding amount and spatial feature sizes at
    runtime."""

    def __init__(self, onnx_node):
        super().__init__(onnx_node)

    def get_nodeattr_types(self):
        my_attrs = {
            # spatial size of input images
            "ImgDim": ("ints", True, []),  # [H, W] = [Y, X]
            # total padding (per dimension) to apply
            "Padding": (
                "ints",
                True,
                [1, 1, 1, 1],
            ),  # [H_begin, W_begin, H_end, W_end] = [Y_begin, X_begin, Y_end, X_end]
            # number of channels in input image
            "NumChannels": ("i", True, 0),
            # SIMD Input parallelism
            "SIMD": ("i", False, 1),
            # FINN input datatype
            "inputDataType": ("s", True, ""),
            # controls distribution of padded pixels
            # in case of uneven padding -- see FMPadding fxn
            # in hlslib
            "PaddingStyle": ("i", False, 2, {2, 1}),
            # shape describing input vecs per execution
            "numInputVectors": ("i", False, 1),
            # Enable reprogrammable implementation to change FM dimensions,
            # stride, or dilation during runtime
            "dynamic_mode": ("i", False, 0, {0, 1}),
            # attribute to save top module name - not user configurable
            "gen_top_module": ("s", False, ""),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_padded_odim(self):
        "Return the padded spatial size of the output."
        idim_h, idim_w = self.get_nodeattr("ImgDim")
        pad = self.get_nodeattr("Padding")
        pad_h = pad[0] + pad[2]
        pad_w = pad[1] + pad[3]
        odim_h = idim_h + pad_h
        odim_w = idim_w + pad_w
        return [odim_h, odim_w]

    def get_exp_cycles(self):
        odim_h, odim_w = self.get_padded_odim()
        channels = self.get_nodeattr("NumChannels")
        simd = self.get_nodeattr("SIMD")
        batch_size = self.get_nodeattr("numInputVectors")
        exp_cycles = (channels / simd) * batch_size * odim_h * odim_w
        return int(exp_cycles)

    def get_normal_input_shape(self):
        idim_h, idim_w = self.get_nodeattr("ImgDim")
        num_ch = self.get_nodeattr("NumChannels")
        ishape = (1, idim_h, idim_w, num_ch)
        return ishape

    def get_normal_output_shape(self):
        odim_h, odim_w = self.get_padded_odim()
        num_ch = self.get_nodeattr("NumChannels")

        oshape = (1, odim_h, odim_w, num_ch)
        return oshape

    def get_folded_input_shape(self):
        normal_ishape = list(self.get_normal_input_shape())
        ifm_ch = self.get_nodeattr("NumChannels")
        simd = self.get_nodeattr("SIMD")
        assert ifm_ch % simd == 0, "SIMD must divide input channels"
        fold = int(normal_ishape[-1] / simd)
        folded_ishape = normal_ishape[:-1] + [fold, simd]
        return tuple(folded_ishape)

    def get_folded_output_shape(self):
        normal_oshape = list(self.get_normal_output_shape())
        ifm_ch = self.get_nodeattr("NumChannels")
        simd = self.get_nodeattr("SIMD")
        assert ifm_ch % simd == 0, "SIMD must divide input channels"
        fold = int(normal_oshape[-1] / simd)
        folded_oshape = normal_oshape[:-1] + [fold, simd]
        return tuple(folded_oshape)

    def make_shape_compatible_op(self, model):
        exp_ishape = self.get_normal_input_shape()
        oshape = self.get_normal_output_shape()
        ishape = tuple(model.get_tensor_shape(self.onnx_node.input[0]))
        assert ishape == exp_ishape, "Unexpect input shape for SameResize."
        return super().make_const_shape_op(oshape)

    def infer_node_datatype(self, model):
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            warn_str = "inputDataType changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype()),
                str(idt),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("inputDataType", idt.name)
        model.set_tensor_datatype(node.output[0], idt)

    def verify_node(self):
        pass

    def get_input_datatype(self):
        """Returns FINN DataType of input."""
        ret = DataType[self.get_nodeattr("inputDataType")]
        # the hlslib op always pads with zeros, so ensure that the DataType
        # is able to represent zeros
        assert ret.allowed(0), "FMPadding_Batch DataType must support zero"
        return ret

    def get_output_datatype(self):
        """Returns FINN DataType of output. (Same as input datatype)"""
        return self.get_input_datatype()

    def get_instream_width(self):
        ibits = self.get_input_datatype().bitwidth()
        simd = self.get_nodeattr("SIMD")
        return ibits * simd

    def get_outstream_width(self):
        obits = self.get_output_datatype().bitwidth()
        simd = self.get_nodeattr("SIMD")
        return obits * simd

    def get_number_output_values(self):
        folded_oshape = self.get_folded_output_shape()
        return np.prod(folded_oshape[:-1])

    def get_verilog_top_module_intf_names(self):
        # Overload default HLSCustomOp implementation to add axilite control IF
        intf_names = super().get_verilog_top_module_intf_names()
        if self.get_nodeattr("dynamic_mode"):
            intf_names["axilite"] = ["s_axilite"]
        return intf_names

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node
        exp_ishape = self.get_normal_input_shape()
        exp_oshape = self.get_normal_output_shape()
        folded_ishape = self.get_folded_input_shape()

        if mode == "cppsim":
            raise Exception(
                "cppsim not possible for FMPadding_rtl, please set exec_mode to rtlsim"
            )
        elif mode == "rtlsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )

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

    def get_template_values(self, ifm_dims, pads, chans, simd, idt):
        dimY, dimX = ifm_dims
        padT, padL, padB, padR = pads
        y_counter_bits = int(math.ceil(math.log2(padT + dimY + padB)))
        x_counter_bits = int(math.ceil(math.log2(padL + dimX + padR)))
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

    def get_dynamic_config(self, ifm_dims, pads):
        """Returns a configuration dict to re-configure FM dimension and
        padding amounts during runtime."""

        chans = self.get_nodeattr("NumChannels")
        simd = self.get_nodeattr("SIMD")
        idt = self.get_input_datatype()
        code_gen_dict = self.get_template_values(ifm_dims, pads, chans, simd, idt)
        config = {
            "XON": (0, (code_gen_dict["INIT_XON"])),
            "XOFF": (1, (code_gen_dict["INIT_XOFF"])),
            "XEND": (2, (code_gen_dict["INIT_XEND"])),
            "YON": (4, (code_gen_dict["INIT_YON"])),
            "YOFF": (5, (code_gen_dict["INIT_YOFF"])),
            "YEND": (6, (code_gen_dict["INIT_YEND"])),
        }
        return config

    def generate_hdl(self):
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

    def code_generation_ipgen(self, model, fpgapart, clk):
        """Normally: Generates C++ code and tcl script for IP generation.
        Here: Generates (System-)Verilog code for IP generation."""
        self.generate_hdl()

    def ipgen_singlenode_code(self):
        """Normally: Builds the bash script for IP generation."""
        pass

    def code_generation_cppsim(self, model):
        """Normally: Generates C++ code for simulation (cppsim)."""
        pass

    def compile_singlenode_code(self):
        pass

    def global_includes(self):
        pass

    def defines(self, var):
        pass

    def read_npy_data(self):
        pass

    def strm_decl(self):
        pass

    def docompute(self):
        pass

    def dataoutstrm(self):
        pass

    def save_as_npy(self):
        pass

    def blackboxfunction(self):
        pass

    def pragmas(self):
        pass
