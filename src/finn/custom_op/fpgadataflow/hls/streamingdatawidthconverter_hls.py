# Copyright (C) 2023, Advanced Micro Devices, Inc.
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

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.streamingdatawidthconverter import (
    StreamingDataWidthConverter,
)

# does not do anything at the ONNX node-by-node level, and input-output
# tensor shapes are the same. performs data width conversion at the rtlsim level


class StreamingDataWidthConverter_hls(StreamingDataWidthConverter, HLSBackend):
    """Class that corresponds to finn-hlslib StreamingDataWidthConverter_Batch
    function."""

    def get_iowidth_lcm(self):
        iwidth = self.get_nodeattr("inWidth")
        owidth = self.get_nodeattr("outWidth")
        return int(np.lcm(iwidth, owidth))

    def needs_lcm(self):
        iwidth = self.get_nodeattr("inWidth")
        owidth = self.get_nodeattr("outWidth")
        return max(iwidth, owidth) % min(iwidth, owidth) != 0

    def _needs_element_width_split(self):
        return self.needs_lcm() and self.get_iowidth_lcm() > 8191

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(StreamingDataWidthConverter.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "dwc.hpp"']

    def defines(self, var):
        inWidth = self.get_nodeattr("inWidth")
        outWidth = self.get_nodeattr("outWidth")
        self.code_gen_dict["$DEFINES$"] = [
            "static constexpr unsigned InWidth = %d;" % inWidth,
            "static constexpr unsigned OutWidth = %d;" % outWidth,
        ]
        if self._needs_element_width_split():
            elemWidth = self.get_input_datatype().bitwidth()
            assert inWidth % elemWidth == 0, "DWC input width must be element-aligned"
            assert outWidth % elemWidth == 0, "DWC output width must be element-aligned"
            self.code_gen_dict["$DEFINES$"].append(
                "static constexpr unsigned IntermediateWidth = %d;" % elemWidth
            )
        elif self.needs_lcm():
            lcmWidth = self.get_iowidth_lcm()
            self.code_gen_dict["$DEFINES$"].append(
                "static constexpr unsigned LCMWidth = %d;" % lcmWidth
            )

    def strm_decl(self):
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> in0_V ("in0_V");'.format(self.get_instream_width())
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> out0_V ("out0_V");'.format(self.get_outstream_width())
        )

    def docompute(self):
        op = "StreamingDataWidthConverter"
        if self._needs_element_width_split():
            self.code_gen_dict["$DOCOMPUTE$"] = [
                'hls::stream<ap_uint<IntermediateWidth>> intermediate ("intermediate");',
                "%s<InWidth, IntermediateWidth>(in0_V, intermediate);" % op,
                "%s<IntermediateWidth, OutWidth>(intermediate, out0_V);" % op,
            ]
        elif self.needs_lcm():
            self.code_gen_dict["$DOCOMPUTE$"] = [
                'hls::stream<ap_uint<{}>> intermediate ("intermediate");'.format(
                    self.get_iowidth_lcm()
                ),
                "%s<InWidth, LCMWidth>(in0_V, intermediate);" % op,
                "%s<LCMWidth, OutWidth>(intermediate, out0_V);" % op,
            ]
        else:
            self.code_gen_dict["$DOCOMPUTE$"] = ["%s<InWidth, OutWidth>(in0_V, out0_V);" % op]

    def blackboxfunction(self):
        in_packed_bits = self.get_instream_width()
        in_packed_hls_type = "ap_uint<%d>" % in_packed_bits
        out_packed_bits = self.get_outstream_width()
        out_packed_hls_type = "ap_uint<%d>" % out_packed_bits
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            "void %s(hls::stream<%s > &in0_V, hls::stream<%s > &out0_V)"
            % (
                self.onnx_node.name,
                in_packed_hls_type,
                out_packed_hls_type,
            )
        ]

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"] = ["#pragma HLS INTERFACE axis port=in0_V"]
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=out0_V")
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE ap_ctrl_none port=return")
        if self.needs_lcm():
            self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS DATAFLOW disable_start_propagation")

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        if mode == "cppsim":
            exp_shape = self.get_normal_input_shape()
            output = context[self.onnx_node.input[0]]
            output = np.asarray([output], dtype=np.float32).reshape(*exp_shape)
            context[self.onnx_node.output[0]] = output

        elif mode == "rtlsim":
            HLSBackend.execute_node(self, context, graph)

    def ipgen_extra_directives(self):
        "Return a list of extra tcl directives for HLS synthesis."

        return ["config_compile -pipeline_style flp"]
