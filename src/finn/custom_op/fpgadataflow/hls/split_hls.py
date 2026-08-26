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

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.split import StreamingSplit


class StreamingSplit_hls(StreamingSplit, HLSBackend):
    """Streaming split node with dynamically generated HLS.
    Only supports splitting along the last axis."""

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(StreamingSplit.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def execute_node(self, context, graph):
        HLSBackend.execute_node(self, context, graph)

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "split.hpp"']

    def defines(self, var):
        self.code_gen_dict["$DEFINES$"] = []

    def docompute(self):
        self.code_gen_dict["$DOCOMPUTE$"] = []
        n_outputs = self.get_n_outputs()
        output_folds = [str(self.get_folded_output_shape(i)[-2]) for i in range(n_outputs)]
        out_streams = []
        for i in range(self.get_n_outputs()):
            out_streams.append("out%d_V" % i)
        out_stream_names = ", ".join(out_streams)
        out_stream_folds = ", ".join(output_folds)
        comp_call = "StreamingSplit<%s>(in0_V, %s);" % (out_stream_folds, out_stream_names)
        self.code_gen_dict["$DOCOMPUTE$"] = [comp_call]

    def blackboxfunction(self):
        input_elem_hls_type = self.get_input_datatype().get_hls_datatype_str()
        simd = self.get_nodeattr("SIMD")
        in_stream = "hls::stream<hls::vector<%s, %d>> &in0_V" % (input_elem_hls_type, simd)
        out_streams = []
        for i in range(self.get_n_outputs()):
            out_streams.append(
                "hls::stream<hls::vector<%s, %d>> &out%d_V" % (input_elem_hls_type, simd, i)
            )
        out_streams = ", ".join(out_streams)
        blackbox_hls = "void %s(%s, %s)" % (self.onnx_node.name, in_stream, out_streams)
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [blackbox_hls]

    def pragmas(self):
        pragmas = []
        pragmas.append("#pragma HLS INTERFACE axis port=in0_V")
        for i in range(self.get_n_outputs()):
            pragmas.append("#pragma HLS INTERFACE axis port=out%d_V" % i)
        pragmas.append("#pragma HLS INTERFACE ap_ctrl_none port=return")
        pragmas.append("#pragma HLS aggregate variable=in0_V compact=bit")
        for i in range(self.get_n_outputs()):
            pragmas.append("#pragma HLS aggregate variable=out%d_V compact=bit" % i)
        self.code_gen_dict["$PRAGMAS$"] = pragmas

    def timeout_condition(self):
        condition = []
        for i in range(self.get_n_outputs()):
            condition.append("out{}_V.empty()".format(i))
        condition = " && ".join(condition)
        self.code_gen_dict["$TIMEOUT_CONDITION$"] = [condition]

    def timeout_read_stream(self):
        read_stream_command = []
        for i in range(self.get_n_outputs()):
            read_stream_command.append(
                """if(!out%d_V.empty()){
                   strm%d << out%d_V.read();
                   }"""
                % (i, i, i)
            )
        self.code_gen_dict["$TIMEOUT_READ_STREAM$"] = read_stream_command
