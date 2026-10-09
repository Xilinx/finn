# Copyright (c) 2020-2022, Xilinx, Inc.
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
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class TLastMarker_hls(HWCustomOp, HLSBackend):
    """Node that adds/removes AXI stream TLAST signals where needed. Its behavior
    is transparent in node-by-node execution, only visible in IP-stitched rtlsim or
    actual hardware.
    This node  may be needed at the end of the network to signal a DMA write
    (needed by the FINN PYNQ shell) or at the beginning to remove the end-of-burst
    from DMA read."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            # number of (static) iterations until TLAST=1 is generated for Direction=out
            "NumIters": ("i", True, 0),
            # whether static or dynamic (from AXI lite) number of iterations are used
            "DynIters": ("i", False, 0),
            # direction: whether to insert or remove TLAST
            "Direction": ("s", False, "out", {"out", "in"}),
            # width of input-output data streams, in bits
            "StreamWidth": ("i", True, 0),
            # width of individual element in stream, in bits
            "ElemWidth": ("i", True, 0),
            # Protocol: external or internal
            # Vitis docs recommend using qdma_axis for external, ap_axiu for internal
            "Protocol": ("s", False, "external", {"external", "internal"}),
        }
        my_attrs.update(HWCustomOp.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
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

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "last_marker.hpp"']

    def defines(self, var):
        stream_width = self.get_nodeattr("StreamWidth")
        direction = self.get_nodeattr("Direction")
        protocol = self.get_nodeattr("Protocol")
        # output stream must have TLAST, so we use this stream data type:
        # qdma_axis<stream_data_width,0,0,0 >
        if direction == "out":
            if protocol == "external":
                out_stream_dtype = "qdma_axis<%d,0,0,0>" % stream_width
            elif protocol == "internal":
                out_stream_dtype = "ap_axiu<%d,0,0,0>" % stream_width
            else:
                raise Exception("Unrecognized Protocol in TLastMarker")
            in_stream_dtype = "ap_uint<%d>" % stream_width
        elif direction == "in":
            out_stream_dtype = "ap_uint<%d>" % stream_width
            if protocol == "external":
                in_stream_dtype = "qdma_axis<%d,0,0,0>" % stream_width
            elif protocol == "internal":
                in_stream_dtype = "ap_axiu<%d,0,0,0>" % stream_width
            else:
                raise Exception("Unrecognized Protocol in TLastMarker")
        else:
            raise Exception("Unrecognized Direction in TLastMarker")

        self.code_gen_dict["$DEFINES$"] = [
            "#define OutDType %s" % out_stream_dtype,
            "#define InDType %s" % in_stream_dtype,
        ]

    def read_npy_data(self):
        self.code_gen_dict["$READNPYDATA$"] = []

    def docompute(self):
        if self.get_nodeattr("DynIters") == 1:
            raise Exception("DynIters=1 is not supported for TLastMarker_hls")
        direction = self.get_nodeattr("Direction")
        num_iters = self.get_nodeattr("NumIters")

        if direction == "in":
            self.code_gen_dict["$DOCOMPUTE$"] = [
                "TLastMarker_In<InDType>(in0_V, out0_V);"
            ]
        else:
            self.code_gen_dict["$DOCOMPUTE$"] = [
                "TLastMarker_Out<%d, OutDType>(in0_V, out0_V);" % num_iters
            ]

    def dataoutstrm(self):
        self.code_gen_dict["$DATAOUTSTREAM$"] = []

    def blackboxfunction(self):
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            """void %s(hls::stream<InDType> &in0_V,
            hls::stream<OutDType> &out0_V)"""
            % self.onnx_node.name
        ]

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"] = [
            "#pragma HLS INTERFACE axis port=in0_V",
            "#pragma HLS INTERFACE axis port=out0_V",
            "#pragma HLS INTERFACE ap_ctrl_none port=return",
            "#pragma HLS dataflow disable_start_propagation",
        ]

    def get_number_output_values(self):
        return self.get_nodeattr("NumIters")

    def get_input_datatype(self, ind=0):
        # not supported
        raise Exception("get_input_datatype not implemented for TlastMarker")

    def get_output_datatype(self, ind=0):
        # not supported
        raise Exception("get_output_datatype not implemented for TlastMarker")

    def get_normal_input_shape(self, ind=0):
        # not supported
        raise Exception("get_normal_input_shape not implemented for TlastMarker")

    def get_normal_output_shape(self, ind=0):
        # not supported
        raise Exception("get_normal_input_shape not implemented for TlastMarker")

    def get_folded_input_shape(self, ind=0):
        stream_width = self.get_nodeattr("StreamWidth")
        elem_width = self.get_nodeattr("ElemWidth")
        n_packed_elems = stream_width // elem_width
        n_iters = self.get_nodeattr("NumIters")
        return (1, n_iters, n_packed_elems)

    def get_folded_output_shape(self, ind=0):
        return self.get_folded_input_shape()

    def get_instream_width(self, ind=0):
        stream_width = self.get_nodeattr("StreamWidth")
        return stream_width

    def get_outstream_width(self, ind=0):
        stream_width = self.get_nodeattr("StreamWidth")
        return stream_width

    def strm_decl(self):
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        self.code_gen_dict["$STREAMDECLARATIONS$"].append('hls::stream<InDType> in0_V ("in0_V");')
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<OutDType> out0_V ("out0_V");'
        )

    def get_verilog_top_module_intf_names(self):
        intf_names = super().get_verilog_top_module_intf_names()
        stream_width = self.get_nodeattr("StreamWidth")
        intf_names["s_axis"] = [("in0_V", stream_width)]
        intf_names["m_axis"] = [("out0_V", stream_width)]
        return intf_names
