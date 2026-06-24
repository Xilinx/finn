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

import os
import shutil

from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.custom_op.fpgadataflow.where import Where


def _rtlsrc_dir():
    return os.environ["FINN_ROOT"] + "/finn-rtllib/where/hdl"


class Where_rtl(Where, RTLBackend):
    """RTL implementation of Where."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(Where.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    def _shape_literal(self, shape):
        rtl_shape = self._rtl_shape(shape)
        return "'{ " + ", ".join(str(int(x)) for x in rtl_shape) + " }"

    def generate_hdl(self, model, fpgapart, clk):
        pe = self._output_stream_pe()
        out_shape = self.get_normal_output_shape()
        cond_shape = self.get_normal_input_shape(0)
        x_shape = self.get_normal_input_shape(1)
        y_shape = self.get_normal_input_shape(2)
        out_rtl_shape = self._rtl_shape(out_shape)
        cond_rtl_shape = self._rtl_shape(cond_shape)
        x_rtl_shape = self._rtl_shape(x_shape)
        y_rtl_shape = self._rtl_shape(y_shape)
        assert out_rtl_shape[-1] % pe == 0, "PE must divide the output innermost dimension"

        rtlsrc = _rtlsrc_dir()
        template_path = rtlsrc + "/where_template.v"
        with open(template_path, "r") as f:
            template = f.read()
        core_template_path = rtlsrc + "/where_core_template.sv"
        with open(core_template_path, "r") as f:
            core_template = f.read()

        topname = self.get_verilog_top_module_name()
        self.set_nodeattr("gen_top_module", topname)

        elem_width = self.get_input_datatype(1).bitwidth()
        cond_width = self.get_instream_width(0)
        x_width = self.get_instream_width(1)
        y_width = self.get_instream_width(2)
        out_width = self.get_outstream_width(0)
        code_gen_dict = {
            "TOP_MODULE_NAME": topname,
            "PE": pe,
            "DATA_WIDTH": elem_width,
            "NDIMS": len(out_rtl_shape),
            "COND_NDIMS": len(cond_rtl_shape),
            "X_NDIMS": len(x_rtl_shape),
            "Y_NDIMS": len(y_rtl_shape),
            "OUT_SHAPE": self._shape_literal(out_shape),
            "COND_SHAPE": self._shape_literal(cond_shape),
            "X_SHAPE": self._shape_literal(x_shape),
            "Y_SHAPE": self._shape_literal(y_shape),
            "COND_WIDTH": cond_width,
            "X_WIDTH": x_width,
            "Y_WIDTH": y_width,
            "OUT_WIDTH": out_width,
            "RAM_STYLE": '"{}"'.format(self.get_nodeattr("ram_style")),
        }

        for key, value in code_gen_dict.items():
            template = template.replace("$%s$" % key, str(value))
            core_template = core_template.replace("$%s$" % key, str(value))

        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        with open(os.path.join(code_gen_dir, topname + ".v"), "w") as f:
            f.write(template)
        with open(os.path.join(code_gen_dir, topname + "_core.sv"), "w") as f:
            f.write(core_template)
        for sv_file in ["input_gen.sv", "where.sv"]:
            shutil.copy(rtlsrc + "/" + sv_file, code_gen_dir)

        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def get_rtl_file_list(self, abspath=False):
        if abspath:
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen") + "/"
            rtllib_dir = _rtlsrc_dir() + "/"
        else:
            code_gen_dir = ""
            rtllib_dir = ""

        return [
            rtllib_dir + "input_gen.sv",
            rtllib_dir + "where.sv",
            code_gen_dir + self.get_nodeattr("gen_top_module") + "_core.sv",
            code_gen_dir + self.get_nodeattr("gen_top_module") + ".v",
        ]

    def code_generation_ipi(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        sourcefiles = self.get_rtl_file_list()
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
            Where.execute_node(self, context, graph)
        elif mode == "rtlsim":
            RTLBackend.execute_node(self, context, graph)
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following values ("cppsim", "rtlsim")""".format(
                    mode
                )
            )
