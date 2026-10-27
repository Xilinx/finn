# Copyright (c) 2021, Xilinx
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

from finn.custom_op.fpgadataflow.concat import StreamingConcat
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend


class StreamingConcat_hls(StreamingConcat, HLSBackend):
    """Streaming concatenation node with dynamically generated HLS.
    Only supports concatenating along the last axis."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(StreamingConcat.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def execute_node(self, context, graph):
        HLSBackend.execute_node(self, context, graph)

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "concat.hpp"']

    def defines(self, var):
        self.code_gen_dict["$DEFINES$"] = ["#define SIMD {}".format(self.get_nodeattr("SIMD"))]

    def docompute(self):
        self.code_gen_dict["$DOCOMPUTE$"] = []
        n_inputs = self.get_n_inputs()
        input_folds = [str(self.get_folded_input_shape(i)[-2]) for i in range(n_inputs)]
        in_streams = []
        for i in range(n_inputs):
            in_streams.append("in%d_V" % i)
        in_stream_names = ", ".join(in_streams)
        in_stream_folds = ", ".join(input_folds)
        comp_call = "StreamingConcat<{}>(out0_V, {});".format(in_stream_folds, in_stream_names)
        self.code_gen_dict["$DOCOMPUTE$"] = [comp_call]

    def blackboxfunction(self):
        n_inputs = self.get_n_inputs()
        in_streams = []
        for i in range(n_inputs):
            input_elem_hls_type = self.get_input_datatype(i).get_hls_datatype_str()
            in_streams.append(
                "hls::stream<hls::vector<%s, SIMD>> &in%d_V" % (input_elem_hls_type, i)
            )
        in_streams = ", ".join(in_streams)
        out_stream = "hls::stream<hls::vector<%s, SIMD>> &out0_V" % (
            self.get_output_datatype().get_hls_datatype_str(),
        )
        blackbox_hls = "void %s(%s, %s)" % (self.onnx_node.name, in_streams, out_stream)
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [blackbox_hls]

    def pragmas(self):
        n_inputs = self.get_n_inputs()
        pragmas = []
        for i in range(n_inputs):
            pragmas.append("#pragma HLS INTERFACE axis port=in%d_V" % i)
        self.code_gen_dict["$PRAGMAS$"] = pragmas
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=out0_V")
        for i in range(n_inputs):
            pragmas.append("#pragma HLS aggregate variable=in%d_V compact=bit" % i)
        pragmas.append("#pragma HLS aggregate variable=out0_V compact=bit")
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE ap_ctrl_none port=return")
