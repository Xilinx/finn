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
import os

from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.custom_op.fpgadataflow.streamingfifo import StreamingFIFO
from finn.util.basic import fifo_rtl_files, is_versal


class StreamingFIFO_rtl(StreamingFIFO, RTLBackend):
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            # single-valued, kept so that existing folding configs still load
            "impl_style": ("s", False, "rtl", {"rtl"}),
        }
        my_attrs.update(StreamingFIFO.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))

        return my_attrs

    def get_verilog_top_module_intf_names(self):
        ret = super().get_verilog_top_module_intf_names()
        if self.get_nodeattr("depth_monitor") == 1:
            ret["ap_none"] = ["maxcount"]
        return ret

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

        depth = int(self.get_nodeattr("depth"))
        # fifo.sv will not elaborate below DEPTH 2; catch it here rather than in a
        # Vivado elaboration log
        assert depth >= 2, (
            "%s: depth %d cannot be built, fifo.sv requires 2 or above. A FIFO this "
            "shallow should have been removed by RemoveShallowFIFOs." % (self.onnx_node.name, depth)
        )
        # fifo.sv may implement more capacity than requested. shift is never
        # smaller then four stages, the memory path rounds up to whole primitives.
        count_width = depth.bit_length() + 1
        # fifo.sv's RAM_STYLE_EFF ladder is the decision, so the request goes to it
        # untouched; resolve_ram_style() only predicts what it will pick, for the
        # estimators and the build report
        self.set_nodeattr("ram_style_resolved", self.resolve_ram_style())
        self.set_nodeattr("is_versal", int(is_versal(fpgapart)))
        code_gen_dict["$COUNT_WIDTH$"] = f"{count_width}"
        code_gen_dict["$COUNT_RANGE$"] = "[{}:0]".format(count_width - 1)
        code_gen_dict["$IN_RANGE$"] = "[{}:0]".format(in_width - 1)
        code_gen_dict["$OUT_RANGE$"] = "[{}:0]".format(in_width - 1)
        code_gen_dict["$WIDTH$"] = str(in_width)
        code_gen_dict["$DEPTH$"] = str(depth)
        code_gen_dict["$RAM_STYLE$"] = self.get_nodeattr("ram_style")
        code_gen_dict["$DATA_LOGFILE$"] = self.get_nodeattr("debug_log_path")
        # apply code generation to templates
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        with open(template_path, "r") as f:
            template = f.read()
        for key_name, value in code_gen_dict.items():
            key = "%s" % key_name
            template = template.replace(key, str(value))
        with open(
            os.path.join(code_gen_dir, self.get_verilog_top_module_name() + ".v"),
            "w",
        ) as f:
            f.write(template)

        # set ipgen_path and ip_path so that HLS-Synth transformation
        # and stich_ip transformation do not complain
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def code_generation_ipi(self):
        cmd = ["add_files -norecurse %s" % f for f in self.get_rtl_file_list(abspath=True)]
        cmd += [
            "create_bd_cell -type module -reference %s %s"
            % (self.get_nodeattr("gen_top_module"), self.onnx_node.name)
        ]
        return cmd

    def get_rtl_file_list(self, abspath=False):
        """The shared FIFO sources plus the per-node wrapper generate_hdl() wrote."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen") + "/" if abspath else ""
        return [
            code_gen_dir + self.get_nodeattr("gen_top_module") + ".v"
        ] + fifo_rtl_files(abspath, gauge=True)

    def execute_node(self, context, graph):
        # a FIFO only passes data through, so it is never simulated on its own:
        # pin the passthrough ahead of RTLBackend's rtlsim implementation
        StreamingFIFO.execute_node(self, context, graph)
