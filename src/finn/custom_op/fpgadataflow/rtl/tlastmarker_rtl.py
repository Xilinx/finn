# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import os

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend


def _rtlsrc_dir():
    return os.path.join(os.environ["FINN_ROOT"], "finn-rtllib", "lastmarker", "hdl")


class TLastMarker_rtl(HWCustomOp, RTLBackend):
    """RTL implementation of TLastMarker using a finn-rtllib counter core.

    Direction="out" inserts TLAST every NumIters beats.
    Direction="in" strips incoming TLAST (pure wire-through).
    """

    def get_nodeattr_types(self):
        my_attrs = {
            "NumIters": ("i", True, 0),
            "DynIters": ("i", False, 0),
            "Direction": ("s", False, "out", {"out", "in"}),
            "StreamWidth": ("i", True, 0),
            "ElemWidth": ("i", True, 0),
            "Protocol": ("s", False, "external", {"external", "internal"}),
        }
        my_attrs.update(HWCustomOp.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        if mode == "cppsim":
            # TLastMarker is transparent in node-by-node execution
            i_name = self.onnx_node.input[0]
            o_name = self.onnx_node.output[0]
            context[o_name] = context[i_name]
        elif mode == "rtlsim":
            RTLBackend.execute_node(self, context, graph)
        else:
            raise ValueError('exec_mode must be "cppsim" or "rtlsim"')

    def generate_hdl(self, model, fpgapart, clk):
        if self.get_nodeattr("DynIters") == 1:
            raise Exception("DynIters=1 is not supported for TLastMarker_rtl")

        rtlsrc = _rtlsrc_dir()
        with open(os.path.join(rtlsrc, "lastmarker_template.v"), "r") as f:
            template = f.read()

        topname = self.get_verilog_top_module_name()
        self.set_nodeattr("gen_top_module", topname)

        direction = self.get_nodeattr("Direction")
        params = {
            "TOP_MODULE_NAME": topname,
            "DATA_WIDTH": str(self.get_nodeattr("StreamWidth")),
            "DIRECTION_OUT": "1" if direction == "out" else "0",
            "NUM_ITERS": str(self.get_nodeattr("NumIters")),
        }
        for key, value in params.items():
            template = template.replace("$%s$" % key, value)

        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        with open(os.path.join(code_gen_dir, topname + ".v"), "w") as f:
            f.write(template)
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def get_rtl_file_list(self, abspath=False):
        if abspath:
            return [
                os.path.join(_rtlsrc_dir(), "lastmarker_axi.sv"),
                os.path.join(
                    self.get_nodeattr("code_gen_dir_ipgen"),
                    self.get_nodeattr("gen_top_module") + ".v",
                ),
            ]
        return ["lastmarker_axi.sv", self.get_nodeattr("gen_top_module") + ".v"]

    def code_generation_ipi(self):
        cmd = []
        for f in self.get_rtl_file_list(abspath=True):
            cmd.append("add_files -norecurse %s" % f)
        cmd.append(
            "create_bd_cell -type module -reference %s %s"
            % (self.get_nodeattr("gen_top_module"), self.onnx_node.name)
        )
        return cmd

    def make_shape_compatible_op(self, model):
        pass

    def infer_node_datatype(self, model):
        pass

    def get_input_datatype(self, ind=0):
        raise Exception("get_input_datatype not implemented for TLastMarker")

    def get_output_datatype(self, ind=0):
        raise Exception("get_output_datatype not implemented for TLastMarker")

    def get_normal_input_shape(self, ind=0):
        raise Exception("get_normal_input_shape not implemented for TLastMarker")

    def get_normal_output_shape(self, ind=0):
        raise Exception("get_normal_output_shape not implemented for TLastMarker")

    def get_number_output_values(self):
        return self.get_nodeattr("NumIters")

    def get_folded_input_shape(self, ind=0):
        stream_width = self.get_nodeattr("StreamWidth")
        elem_width = self.get_nodeattr("ElemWidth")
        n_packed_elems = stream_width // elem_width
        n_iters = self.get_nodeattr("NumIters")
        return (1, n_iters, n_packed_elems)

    def get_folded_output_shape(self, ind=0):
        return self.get_folded_input_shape()

    def get_instream_width(self, ind=0):
        return self.get_nodeattr("StreamWidth")

    def get_outstream_width(self, ind=0):
        return self.get_nodeattr("StreamWidth")

    def get_verilog_top_module_intf_names(self):
        intf_names = super().get_verilog_top_module_intf_names()
        stream_width = self.get_nodeattr("StreamWidth")
        intf_names["s_axis"] = [("in0_V", stream_width)]
        intf_names["m_axis"] = [("out0_V", stream_width)]
        return intf_names
