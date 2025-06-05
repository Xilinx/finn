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

    def generate_params(self, model, path):
        n_outputs = self.get_num_output_streams()
        inp_streams = []
        commands = []
        o_stream_w = self.get_outstream_width()
        i_stream_w = self.get_instream_width()
        in_stream = "hls::stream<ap_uint<%d> > &in0_V" % (i_stream_w)
        inp_streams.append(in_stream)
        commands.append("ap_uint<%d> e = in0_V.read();" % i_stream_w)
        iters = self.get_number_output_values()["out0"]
        for i in range(n_outputs):
            out_stream = "hls::stream<ap_uint<%d> > &out%d_V" % (o_stream_w, i)
            inp_streams.append(out_stream)
            cmd = "out%d_V.write(e);" % i
            commands.append(cmd)

        impl_hls_code = []
        impl_hls_code.append("void DuplicateStreamsCustom(")
        impl_hls_code.append(",".join(inp_streams))
        impl_hls_code.append(") {")
        impl_hls_code.append("for(unsigned int i = 0; i < %d; i++) {" % iters)
        impl_hls_code.append("#pragma HLS PIPELINE II=1")
        impl_hls_code.append("\n".join(commands))
        impl_hls_code.append("}")
        impl_hls_code.append("}")
        impl_hls_code = "\n".join(impl_hls_code)

        impl_filename = "{}/duplicate_impl.hpp".format(path)
        f_impl = open(impl_filename, "w")
        f_impl.write(impl_hls_code)
        f_impl.close()

    def execute_node(self, context, graph):
        HLSBackend.execute_node(self, context, graph)

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "duplicate_impl.hpp"']

    def defines(self, var):
        self.code_gen_dict["$DEFINES$"] = []

    def strm_decl(self):
        n_outputs = self.get_num_output_streams()
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> in0_V ("in0_V");'.format(self.get_instream_width())
        )
        for i in range(n_outputs):
            out_name = "out%d_V" % i
            self.code_gen_dict["$STREAMDECLARATIONS$"].append(
                'hls::stream<ap_uint<%d>> %s ("%s");'
                % (self.get_outstream_width(), out_name, out_name)
            )

    def docompute(self):
        n_outputs = self.get_num_output_streams()
        ostreams = []
        for i in range(n_outputs):
            ostreams.append("out%d_V" % i)
        dc = "DuplicateStreamsCustom(in0_V, %s);" % (",".join(ostreams),)
        self.code_gen_dict["$DOCOMPUTE$"] = [dc]

    def blackboxfunction(self):
        n_outputs = self.get_num_output_streams()
        inp_streams = []
        o_stream_w = self.get_outstream_width()
        i_stream_w = self.get_instream_width()
        in_stream = "hls::stream<ap_uint<%d> > &in0_V" % i_stream_w
        inp_streams.append(in_stream)
        for i in range(n_outputs):
            out_stream = "hls::stream<ap_uint<%d> > &out%d_V" % (
                o_stream_w,
                i,
            )
            inp_streams.append(out_stream)

        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            """void {}({})""".format(
                self.onnx_node.name,
                ",".join(inp_streams),
            )
        ]

    def pragmas(self):
        n_outputs = self.get_num_output_streams()
        self.code_gen_dict["$PRAGMAS$"] = ["#pragma HLS INTERFACE axis port=in0_V"]
        for i in range(n_outputs):
            self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=out%d_V" % i)
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE ap_ctrl_none port=return")
