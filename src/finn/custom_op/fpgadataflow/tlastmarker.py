# Copyright (c) 2020, Xilinx
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

from finn.custom_op.fpgadataflow import HLSCustomOp
from finn.util.basic import roundup_to_integer_multiple


class TLastMarker(HLSCustomOp):
    """Class that corresponds to the TLastMarker node that needs to be
    inserted at the end of the model for rtlsim with stitched IP.
    It marks the end of the current image/input sample."""

    def __init__(self, onnx_node):
        super().__init__(onnx_node)

    def get_nodeattr_types(self):
        my_attrs = {
            "NumIters": ("i", True, 0),
            # width of input-output data streams, in bits
            "StreamWidth": ("i", True, 0),
            # width of individual element in stream, in bits
            "ElemWidth": ("i", True, 0),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def execute_node(self, context, graph):
        # TLastMarker's behavior is only visible when doing
        # rtlsim with stitched IP, since it marks the end
        # of the current image/input sample. when executing
        # inside FINN as a single node, this is not visible.
        # so here we simply return the input as output
        i_name = self.onnx_node.input[0]
        o_name = self.onnx_node.output[0]
        i_tensor = context[i_name]
        context[o_name] = i_tensor

    def make_shape_compatible_op(self, model):
        # not supported for shape inference
        pass

    def infer_node_datatype(self, model):
        # not supported for datatype inference
        pass

    def verify_node(self):
        # TODO implement verify_node for TLastMarker
        pass

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "ap_axi_sdata.h"']

    def defines(self, var):
        stream_width = self.get_nodeattr("StreamWidth")
        # output stream must have TLAST, so we use this stream data type:
        # qdma_axis<stream_data_width,0,0,0 >
        out_stream_dtype = "qdma_axis<%d,0,0,0>" % stream_width
        self.code_gen_dict["$DEFINES$"] = [
            "#define StreamWidth %d" % stream_width,
            "#define OutDType %s" % out_stream_dtype,
            "#define NumItersPerImg %d" % self.get_nodeattr("NumIters"),
        ]

    def read_npy_data(self):
        self.code_gen_dict["$READNPYDATA$"] = []

    def docompute(self):
        self.code_gen_dict["$DOCOMPUTE$"] = [
            "unsigned int n = 1;",
            "OutDType t;",
            "t.set_keep(-1);",
            "io_section: { // start of cycle accurate region",
            "#pragma HLS protocol fixed",
            "// do a first read from stream before we decide on numIters",
            "// giving software a chance to set up the numIters prior to startup",
            "t.set_data(in0.read());",
            "n = (numIters == 0 ? NumItersPerImg : numIters);",
            "t.set_last(n==1);",
            "out.write(t);",
            "} // end of cycle accurate region",
            "// do one less iteration than spec since we already did one",
            "for(unsigned int i=1; i<n; i++) {",
            "#pragma HLS PIPELINE II=1",
            "t.set_data(in0.read());",
            "t.set_last(i==(n-1));",
            "out.write(t);",
            "}",
        ]

    def dataoutstrm(self):
        self.code_gen_dict["$DATAOUTSTREAM$"] = []

    def save_as_npy(self):
        self.code_gen_dict["$SAVEASCNPY$"] = []

    def blackboxfunction(self):
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            """void %s(hls::stream<ap_uint<StreamWidth> > &in0,
                hls::stream<OutDType> &out, unsigned int numIters)"""
            % self.onnx_node.name
        ]

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"] = ["#pragma HLS INTERFACE axis port=in0"]
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=out")
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE s_axilite port=numIters bundle=control"
        )
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE ap_ctrl_none port=return"
        )

    def get_number_output_values(self):
        return self.get_nodeattr("NumIters")

    def get_folded_input_shape(self):
        stream_width = self.get_nodeattr("StreamWidth")
        elem_width = self.get_nodeattr("ElemWidth")
        n_packed_elems = stream_width // elem_width
        n_iters = self.get_nodeattr("NumIters")
        return (1, n_iters, n_packed_elems)

    def get_folded_output_shape(self):
        return self.get_folded_input_shape()

    def get_instream_width(self, axi_strm_padding=False):
        stream_width = self.get_nodeattr("StreamWidth")
        if axi_strm_padding is True:
            stream_width = roundup_to_integer_multiple(stream_width, 8)
        return stream_width

    def get_outstream_width(self, axi_strm_padding=False):
        stream_width = self.get_nodeattr("StreamWidth")
        if axi_strm_padding is True:
            stream_width = roundup_to_integer_multiple(stream_width, 8)
        return stream_width

    def strm_decl(self):
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> in0 ("in0");'.format(self.get_instream_width())
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<OutDType> out ("out");'
        )
