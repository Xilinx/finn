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
import numpy as np
import os
import shutil
import warnings

from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.custom_op.fpgadataflow.streamingfifo import StreamingFIFO


class StreamingFIFO_rtl(StreamingFIFO, RTLBackend):
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            # Toggle between rtl or IPI implementation
            # rtl - use the rtl generated IP during stitching
            # vivado - use the AXI Infrastructure FIFO
            "impl_style": ("s", False, "rtl", {"rtl", "vivado"}),
        }
        my_attrs.update(StreamingFIFO.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))

        return my_attrs

    def get_adjusted_depth(self):
        impl = self.get_nodeattr("impl_style")
        depth = self.get_nodeattr("depth")
        if impl == "vivado":
            old_depth = depth
            # round up depth to nearest power-of-2
            # Vivado FIFO impl may fail otherwise
            depth = (1 << (depth - 1).bit_length()) if impl == "vivado" else depth
            if old_depth != depth:
                warnings.warn(
                    "%s: rounding-up FIFO depth from %d to %d for impl_style=vivado"
                    % (self.onnx_node.name, old_depth, depth)
                )

        return depth

    def get_verilog_top_module_intf_names(self):
        ret = super().get_verilog_top_module_intf_names()
        is_rtl = self.get_nodeattr("impl_style") == "rtl"
        is_depth_monitor = self.get_nodeattr("depth_monitor") == 1
        if is_rtl and is_depth_monitor:
            ret["ap_none"] = ["maxcount"]
        return ret

    def is_sim_fifo_gauge(self):
        # special case: a StreamingFIFO layer with impl_style=rtl
        # depth_monitor=1 is implemented using a Verilog infite
        # queue sim instead of Q_srl
        is_rtl = self.get_nodeattr("impl_style") == "rtl"
        is_depth_monitor = self.get_nodeattr("depth_monitor") == 1
        return is_depth_monitor and is_rtl

    def generate_hdl(self, model, fpgapart, clk):
        rtlsrc = os.environ["FINN_ROOT"] + "/finn-rtllib/fifo/hdl"
        template_path = rtlsrc + "/fifo_template.v"

        # save top module name so we can refer to it after this node has been renamed
        # (e.g. by GiveUniqueNodeNames(prefix) during MakeZynqProject)
        topname = self.get_verilog_top_module_name()
        self.set_nodeattr("gen_top_module", topname)

        code_gen_dict = {}
        code_gen_dict["$TOP_MODULE_NAME$"] = topname
        # make instream width a multiple of 8 for axi interface
        in_width = self.get_instream_width_padded()

        gauge = self.is_sim_fifo_gauge()
        if gauge:
            count_width = 32
            code_gen_dict["$FIFO_CORE$"] = "sim_fifo_gauge"
            depth = 0
        else:
            count_width = int(self.get_nodeattr("depth")).bit_length()
            code_gen_dict["$FIFO_CORE$"] = "q_srl"
            depth = int(self.get_nodeattr("depth"))
        code_gen_dict["$COUNT_WIDTH$"] = f"{count_width}"
        code_gen_dict["$COUNT_RANGE$"] = "[{}:0]".format(count_width - 1)
        code_gen_dict["$IN_RANGE$"] = "[{}:0]".format(in_width - 1)
        code_gen_dict["$OUT_RANGE$"] = "[{}:0]".format(in_width - 1)
        code_gen_dict["$WIDTH$"] = str(in_width)
        code_gen_dict["$DEPTH$"] = str(depth)
        # apply code generation to templates
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        with open(template_path, "r") as f:
            template = f.read()
        for key_name in code_gen_dict:
            key = "%s" % key_name
            template = template.replace(key, str(code_gen_dict[key_name]))
        with open(
            os.path.join(code_gen_dir, self.get_verilog_top_module_name() + ".v"),
            "w",
        ) as f:
            f.write(template)

        if self.is_sim_fifo_gauge():
            shutil.copy(rtlsrc + "/fifo_gauge.sv", code_gen_dir)
        else:
            shutil.copy(rtlsrc + "/Q_srl.v", code_gen_dir)
        # set ipgen_path and ip_path so that HLS-Synth transformation
        # and stich_ip transformation do not complain
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def code_generation_ipi(self):
        impl_style = self.get_nodeattr("impl_style")
        if impl_style == "rtl":
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

            sourcefiles = [
                "fifo_gauge.sv" if self.is_sim_fifo_gauge() else "Q_srl.v",
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
        elif impl_style == "vivado":
            cmd = []
            node_name = self.onnx_node.name
            depth = self.get_adjusted_depth()
            ram_style = self.get_nodeattr("ram_style")
            # create a hierarchy for this layer, with the same port names
            clk_name = self.get_verilog_top_module_intf_names()["clk"][0]
            rst_name = self.get_verilog_top_module_intf_names()["rst"][0]
            dout_name = self.get_verilog_top_module_intf_names()["m_axis"][0][0]
            din_name = self.get_verilog_top_module_intf_names()["s_axis"][0][0]
            cmd.append("create_bd_cell -type hier %s" % node_name)
            cmd.append("create_bd_pin -dir I -type clk /%s/%s" % (node_name, clk_name))
            cmd.append("create_bd_pin -dir I -type rst /%s/%s" % (node_name, rst_name))
            cmd.append(
                "create_bd_intf_pin -mode Master "
                "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s" % (node_name, dout_name)
            )
            cmd.append(
                "create_bd_intf_pin -mode Slave "
                "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s" % (node_name, din_name)
            )
            # instantiate and configure DWC
            cmd.append(
                "create_bd_cell -type ip "
                "-vlnv xilinx.com:ip:axis_data_fifo:2.0 /%s/fifo" % node_name
            )
            cmd.append(
                "set_property -dict [list CONFIG.FIFO_DEPTH {%d}] "
                "[get_bd_cells /%s/fifo]" % (depth, node_name)
            )
            cmd.append(
                "set_property -dict [list CONFIG.FIFO_MEMORY_TYPE {%s}] "
                "[get_bd_cells /%s/fifo]" % (ram_style, node_name)
            )
            cmd.append(
                "set_property -dict [list CONFIG.TDATA_NUM_BYTES {%d}] "
                "[get_bd_cells /%s/fifo]" % (np.ceil(self.get_outstream_width() / 8), node_name)
            )
            cmd.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/fifo/M_AXIS] "
                "[get_bd_intf_pins %s/%s]" % (node_name, node_name, dout_name)
            )
            cmd.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/fifo/S_AXIS] "
                "[get_bd_intf_pins %s/%s]" % (node_name, node_name, din_name)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] "
                "[get_bd_pins %s/fifo/s_axis_aresetn]" % (node_name, rst_name, node_name)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] "
                "[get_bd_pins %s/fifo/s_axis_aclk]" % (node_name, clk_name, node_name)
            )
            return cmd
        else:
            raise Exception(
                "FIFO implementation style %s not supported, please use rtl or vivado" % impl_style
            )

    def get_rtl_file_list(self, abspath=False):
        if abspath:
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen") + "/"
            rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/fifo/hdl/")
        else:
            code_gen_dir = ""
            rtllib_dir = ""

        verilog_files = [
            rtllib_dir + "Q_srl.v",
            code_gen_dir + self.get_nodeattr("gen_top_module") + ".v",
        ]
        return verilog_files

    def prepare_rtlsim(self):
        assert self.get_nodeattr("impl_style") != "vivado", (
            "StreamingFIFO impl_style "
            "cannot be vivado for rtlsim. Only impl_style=rtl supported."
        )
        return super().prepare_rtlsim()

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        if mode == "cppsim":
            StreamingFIFO.execute_node(self, context, graph)
        elif mode == "rtlsim":
            RTLBackend.execute_node(self, context, graph)
