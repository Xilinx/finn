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

from finn.custom_op.fpgadataflow.duplicatestreams import DuplicateStreams
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend


class DuplicateStreams_hls(DuplicateStreams, HLSBackend):
    """Class that corresponds to finn-hlslib function of the same name."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(DuplicateStreams.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def verify_node(self):
        info_messages = []
        # verify that "backend" is set to "fpgadataflow"
        backend_value = self.get_nodeattr("backend")
        if backend_value == "fpgadataflow":
            info_messages.append("Attribute backend is set correctly")
        else:
            info_messages.append('Attribute backend should be set to "fpgadataflow"')

        # verify that all necessary attributes exist
        try:
            self.get_nodeattr("code_gen_dir_cppsim")
            self.get_nodeattr("executable_path")
            self.get_nodeattr("NumChannels")
            self.get_nodeattr("PE")
            self.get_nodeattr("NumOutputStreams")
            self.get_nodeattr("inputDataType")
            info_messages.append("All necessary attributes exist")
        except Exception:
            info_messages.append("""The required GlobalAccPool_Batch attributes do not exist.""")

        return info_messages

    def execute_node(self, context, graph):
        HLSBackend.execute_node(self, context, graph)

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "dup.hpp"']

    def defines(self, var):
        self.code_gen_dict["$DEFINES$"] = []

    def docompute(self):
        self.code_gen_dict["$DOCOMPUTE$"] = []
        n_outputs = self.get_nodeattr("NumOutputStreams")
        out_streams = []
        for i in range(n_outputs):
            out_streams.append("out%d_V" % i)
        out_stream_names = ", ".join(out_streams)
        comp_call = f"StreamingDup(in0_V, {out_stream_names});"
        self.code_gen_dict["$DOCOMPUTE$"] = [comp_call]

    def blackboxfunction(self):
        input_elem_hls_type = self.get_input_datatype().get_hls_datatype_str()
        pe = self.get_nodeattr("PE")
        in_stream = "hls::stream<hls::vector<%s, %d>> &in0_V" % (input_elem_hls_type, pe)
        out_streams = []
        n_outputs = self.get_nodeattr("NumOutputStreams")
        for i in range(n_outputs):
            out_streams.append(
                "hls::stream<hls::vector<%s, %d>> &out%d_V" % (input_elem_hls_type, pe, i)
            )
        out_streams = ", ".join(out_streams)
        blackbox_hls = "void %s(%s, %s)" % (self.onnx_node.name, in_stream, out_streams)
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [blackbox_hls]

    def pragmas(self):
        pragmas = []
        pragmas.append("#pragma HLS dataflow disable_start_propagation")
        pragmas.append("#pragma HLS INTERFACE axis port=in0_V")
        n_outputs = self.get_nodeattr("NumOutputStreams")
        for i in range(n_outputs):
            pragmas.append("#pragma HLS INTERFACE axis port=out%d_V" % i)
        pragmas.append("#pragma HLS INTERFACE ap_ctrl_none port=return")
        pragmas.append("#pragma HLS aggregate variable=in0_V compact=bit")
        for i in range(n_outputs):
            pragmas.append("#pragma HLS aggregate variable=out%d_V compact=bit" % i)
        self.code_gen_dict["$PRAGMAS$"] = pragmas

    def timeout_condition(self):
        condition = []
        n_outputs = self.get_nodeattr("NumOutputStreams")
        for i in range(n_outputs):
            condition.append("out{}_V.empty()".format(i))
        condition = " && ".join(condition)
        self.code_gen_dict["$TIMEOUT_CONDITION$"] = [condition]

    def timeout_read_stream(self):
        read_stream_command = []
        n_outputs = self.get_nodeattr("NumOutputStreams")
        for i in range(n_outputs):
            read_stream_command.append(
                """if(!out%d_V.empty()){
                   strm%d << out%d_V.read();
                   }"""
                % (i, i, i)
            )
        self.code_gen_dict["$TIMEOUT_READ_STREAM$"] = read_stream_command
