# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import os

from finn.custom_op.fpgadataflow.crop import Crop
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend


def _rtlsrc_dir():
    return os.path.join(os.environ["FINN_ROOT"], "finn-rtllib", "crop", "hdl")


class Crop_rtl(Crop, RTLBackend):
    """RTL implementation of Crop using the finn-rtllib crop core."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        return Crop.get_nodeattr_types(self) | RTLBackend.get_nodeattr_types(self)

    def generate_hdl(self, model, fpgapart, clk):
        height, width = self.get_nodeattr("ImgDim")
        if height == 0:
            height = 1
        channels = self.get_nodeattr("NumChannels")
        simd = self.get_nodeattr("SIMD")
        assert channels % simd == 0, "SIMD must divide NumChannels"

        rtlsrc = _rtlsrc_dir()
        with open(os.path.join(rtlsrc, "crop_template.v"), "r") as f:
            template = f.read()

        topname = self.get_verilog_top_module_name()
        self.set_nodeattr("gen_top_module", topname)
        code_gen_dict = {
            "TOP_MODULE_NAME": topname,
            "H": height,
            "W": width,
            "CF": channels // simd,
            "FOLD_WIDTH": self.get_input_datatype().bitwidth() * simd,
            "CROP_N": self.get_nodeattr("CropNorth"),
            "CROP_E": self.get_nodeattr("CropEast"),
            "CROP_S": self.get_nodeattr("CropSouth"),
            "CROP_W": self.get_nodeattr("CropWest"),
        }
        for key, value in code_gen_dict.items():
            template = template.replace("$%s$" % key, str(value))

        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        with open(os.path.join(code_gen_dir, topname + ".v"), "w") as f:
            f.write(template)
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def get_rtl_file_list(self, abspath=False):
        if abspath:
            return [
                os.path.join(_rtlsrc_dir(), "crop.sv"),
                os.path.join(
                    self.get_nodeattr("code_gen_dir_ipgen"),
                    self.get_nodeattr("gen_top_module") + ".v",
                ),
            ]
        return ["crop.sv", self.get_nodeattr("gen_top_module") + ".v"]

    def code_generation_ipi(self):
        sourcefiles = self.get_rtl_file_list(abspath=True)
        source_target = "./ip/verilog/rtl_ops/%s" % self.onnx_node.name
        cmd = ["file mkdir %s" % source_target]
        for f in sourcefiles:
            cmd += ["add_files -copy_to %s -norecurse %s" % (source_target, f)]
        cmd += [
            "create_bd_cell -type module -reference %s %s"
            % (self.get_nodeattr("gen_top_module"), self.onnx_node.name)
        ]
        return cmd

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        if mode == "cppsim":
            Crop.execute_node(self, context, graph)
        elif mode == "rtlsim":
            RTLBackend.execute_node(self, context, graph)
        else:
            raise ValueError('exec_mode must be either "cppsim" or "rtlsim"')
