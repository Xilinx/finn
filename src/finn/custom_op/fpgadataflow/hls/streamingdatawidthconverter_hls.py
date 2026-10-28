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
from finn.util.data_packing import hls_datatype_str

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
        self.code_gen_dict["$DEFINES$"] = []

    def docompute(self):
        elem_hls_type = hls_datatype_str(self.get_input_datatype())
        n_in = self.get_folded_input_shape()[-1]
        n_out = self.get_folded_output_shape()[-1]
        op = "StreamingDWC"
        if self._needs_element_width_split():
            self.code_gen_dict["$DOCOMPUTE$"] = [
                'hls::stream<tensor<%s,1>> intermediate ("intermediate");' % elem_hls_type,
                "%s<%s,%d,1>(in0_V, intermediate);" % (op, elem_hls_type, n_in),
                "%s<%s,1,%d>(intermediate, out0_V);" % (op, elem_hls_type, n_out),
            ]
        elif self.needs_lcm():
            elem_bits = self.get_input_datatype().bitwidth()
            lcm_elems = self.get_iowidth_lcm() // elem_bits
            self.code_gen_dict["$DOCOMPUTE$"] = [
                'hls::stream<tensor<%s,%d>> intermediate ("intermediate");'
                % (elem_hls_type, lcm_elems),
                "%s<%s,%d,%d>(in0_V, intermediate);" % (op, elem_hls_type, n_in, lcm_elems),
                "%s<%s,%d,%d>(intermediate, out0_V);" % (op, elem_hls_type, lcm_elems, n_out),
            ]
        else:
            self.code_gen_dict["$DOCOMPUTE$"] = [
                "%s<%s,%d,%d>(in0_V, out0_V);" % (op, elem_hls_type, n_in, n_out)
            ]

    def blackboxfunction(self):
        elem_hls_type = hls_datatype_str(self.get_input_datatype())
        n_in = self.get_folded_input_shape()[-1]
        n_out = self.get_folded_output_shape()[-1]
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            "void %s(hls::stream<tensor<%s,%d>> &in0_V, hls::stream<tensor<%s,%d>> &out0_V)"
            % (self.onnx_node.name, elem_hls_type, n_in, elem_hls_type, n_out)
        ]

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"] = [
            "#pragma HLS INTERFACE axis port=in0_V",
            "#pragma HLS INTERFACE axis port=out0_V",
            "#pragma HLS INTERFACE ap_ctrl_none port=return",
            "#pragma HLS aggregate variable=in0_V compact=bit",
            "#pragma HLS aggregate variable=out0_V compact=bit",
        ]
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
