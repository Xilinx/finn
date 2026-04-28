# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
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

import numpy as np
import os
import shutil
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.addclstoken import AddCLSToken
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.util.basic import get_rtlsim_trace_depth, make_build_dir
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy

try:
    from pyverilator import PyVerilator
except ModuleNotFoundError:
    PyVerilator = None


def _rtlsrc_dir():
    return os.environ["FINN_ROOT"] + "/finn-rtllib/addclstoken/hdl"


class AddCLSToken_rtl(AddCLSToken, RTLBackend):
    """RTL implementation of AddCLSToken."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(AddCLSToken.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    def _pack_value(self, value, dtype):
        bitwidth = dtype.bitwidth()
        if dtype == DataType["BIPOLAR"]:
            int_value = int((value + 1) // 2)
        else:
            if dtype.is_fixed_point():
                value = value / dtype.scale_factor()
            int_value = int(value)
            if int_value < 0:
                int_value += 1 << bitwidth
        return int_value & ((1 << bitwidth) - 1)

    def _pack_cls_token(self, model):
        dtype = self.get_input_datatype()
        bitwidth = dtype.bitwidth()
        num_channels = self.get_nodeattr("NumChannels")
        cls_token = model.get_initializer(self.onnx_node.input[1])
        if cls_token is None:
            raise Exception("AddCLSToken RTL generation requires a constant CLS token input.")

        cls_token = np.asarray(cls_token, dtype=np.float32)
        assert cls_token.shape == self.get_normal_input_shape(
            1
        ), "CLS token shape does not match AddCLSToken attributes."
        assert np.vectorize(dtype.allowed)(cls_token).all(), (
            "CLS token values cannot be represented with %s" % dtype.name
        )
        packed = 0
        for i, value in enumerate(cls_token.flatten()):
            packed |= self._pack_value(value, dtype) << (i * bitwidth)
        return "%d'h%x" % (num_channels * bitwidth, packed)

    def generate_hdl(self, model, fpgapart, clk):
        simd = self.get_nodeattr("SIMD")
        num_channels = self.get_nodeattr("NumChannels")
        assert num_channels % simd == 0, "SIMD must divide NumChannels"

        rtlsrc = _rtlsrc_dir()
        template_path = rtlsrc + "/addclstoken_template.v"
        with open(template_path, "r") as f:
            template = f.read()

        topname = self.get_verilog_top_module_name()
        self.set_nodeattr("gen_top_module", topname)

        elem_width = self.get_input_datatype().bitwidth()
        fold_width = elem_width * simd
        code_gen_dict = {
            "TOP_MODULE_NAME": topname,
            "NUM_TOKENS": self.get_nodeattr("NumTokens"),
            "NUM_CHANNELS": num_channels,
            "SIMD": simd,
            "ELEM_WIDTH": elem_width,
            "PAD_TOKENS": self.get_nodeattr("PadTokens"),
            "FOLD_WIDTH": fold_width,
            "CLS_WIDTH": num_channels * elem_width,
            "CLS_DATA": self._pack_cls_token(model),
        }

        for key, value in code_gen_dict.items():
            template = template.replace("$%s$" % key, str(value))

        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        with open(os.path.join(code_gen_dir, topname + ".v"), "w") as f:
            f.write(template)
        shutil.copy(rtlsrc + "/addclstoken.sv", code_gen_dir)

        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def prepare_rtlsim(self):
        if PyVerilator is None:
            raise ImportError("Installation of PyVerilator is required.")

        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        verilog_files = [
            "addclstoken.sv",
            self.get_nodeattr("gen_top_module") + ".v",
        ]
        sim = PyVerilator.build(
            verilog_files,
            build_dir=make_build_dir("pyverilator_" + self.onnx_node.name + "_"),
            verilog_path=[code_gen_dir],
            trace_depth=get_rtlsim_trace_depth(),
            top_module_name=self.get_nodeattr("gen_top_module"),
        )
        self.set_nodeattr("rtlsim_so", sim.lib._name)
        return sim

    def code_generation_ipi(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        sourcefiles = [
            "addclstoken.sv",
            self.get_nodeattr("gen_top_module") + ".v",
        ]
        sourcefiles = [os.path.join(code_gen_dir, f) for f in sourcefiles]

        cmd = []
        for f in sourcefiles:
            cmd += ["add_files -norecurse %s" % f]
        cmd += [
            "create_bd_cell -type module -reference %s %s"
            % (self.get_nodeattr("gen_top_module"), self.onnx_node.name)
        ]
        return cmd

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        if mode == "cppsim":
            AddCLSToken.execute_node(self, context, graph)
        elif mode == "rtlsim":
            node = self.onnx_node
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
            exp_ishape = self.get_normal_input_shape(0)
            exp_oshape = self.get_normal_output_shape()

            inp = context[node.input[0]]
            assert str(inp.dtype) == "float32", "Input datatype is not float32"
            assert inp.shape == exp_ishape, "Input shape does not match expected shape."

            folded_ishape = self.get_folded_input_shape(0)
            np.save(os.path.join(code_gen_dir, "input_0.npy"), inp.reshape(folded_ishape).copy())

            sim = self.get_rtlsim()
            export_idt = self.get_input_datatype()
            rtlsim_inp = npy_to_rtlsim_input(
                os.path.join(code_gen_dir, "input_0.npy"),
                export_idt,
                self.get_instream_width(),
            )
            self.reset_rtlsim(sim)
            self.toggle_clk(sim)
            rtlsim_output = self.rtlsim(sim, rtlsim_inp)

            odt = self.get_output_datatype()
            out_npy = rtlsim_output_to_npy(
                rtlsim_output,
                os.path.join(code_gen_dir, "output.npy"),
                odt,
                self.get_folded_output_shape(),
                self.get_outstream_width(),
                odt.bitwidth(),
            )
            context[node.output[0]] = np.asarray(out_npy, dtype=np.float32).reshape(exp_oshape)
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following values ("cppsim", "rtlsim")""".format(
                    mode
                )
            )
