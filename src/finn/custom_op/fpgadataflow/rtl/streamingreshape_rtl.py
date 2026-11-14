# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""RTL implementation of StreamingReshape."""

import os

from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.custom_op.fpgadataflow.streamingreshape import StreamingReshape


def _rtlsrc_dir():
    return os.path.join(os.environ["FINN_ROOT"], "finn-rtllib", "passthru", "hdl")


class StreamingReshape_rtl(StreamingReshape, RTLBackend):
    """StreamingReshape as a wire (finn-rtllib passthru_axi): no logic, no
    registers, no latency."""

    def get_nodeattr_types(self):
        return StreamingReshape.get_nodeattr_types(self) | RTLBackend.get_nodeattr_types(self)

    def generate_hdl(self, model, fpgapart, clk):
        with open(os.path.join(_rtlsrc_dir(), "passthru_template.v"), "r") as f:
            template = f.read()

        topname = self.get_verilog_top_module_name()
        self.set_nodeattr("gen_top_module", topname)
        # The padded widths of both streams are equal: they carry the same PE
        # elements of the same datatype.
        code_gen_dict = {
            "TOP_MODULE_NAME": topname,
            "AXI_WIDTH": self.get_instream_width_padded(),
        }
        for key, value in code_gen_dict.items():
            template = template.replace("$%s$" % key, str(value))

        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        with open(os.path.join(code_gen_dir, topname + ".v"), "w") as f:
            f.write(template)
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def get_rtl_file_list(self, abspath=False):
        top = self.get_nodeattr("gen_top_module") + ".v"
        if abspath:
            return [
                os.path.join(_rtlsrc_dir(), "passthru_axi.sv"),
                os.path.join(self.get_nodeattr("code_gen_dir_ipgen"), top),
            ]
        return ["passthru_axi.sv", top]

    def code_generation_ipi(self):
        cmd = ["add_files -norecurse %s" % f for f in self.get_rtl_file_list(abspath=True)]
        cmd.append(
            "create_bd_cell -type module -reference %s %s"
            % (self.get_nodeattr("gen_top_module"), self.onnx_node.name)
        )
        return cmd

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        if mode == "cppsim":
            StreamingReshape.execute_node(self, context, graph)
        elif mode == "rtlsim":
            RTLBackend.execute_node(self, context, graph)
        else:
            raise ValueError('exec_mode must be either "cppsim" or "rtlsim"')
