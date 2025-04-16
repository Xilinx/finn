# Copyright (c) 2024, Advanced Micro Devices, Inc.
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
# * Neither the name of Xilinx nor the names of its
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

from finn.custom_op.fpgadataflow.fmpadding_pixel import FMPadding_Pixel
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend


class FMPadding_Pixel_hls(FMPadding_Pixel, HLSBackend):
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(FMPadding_Pixel.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "streamtools.h"']

    def defines(self, var):
        odim_h, odim_w = self.get_padded_odim()
        stride_h, stride_w = self.get_nodeattr("Stride")
        self.code_gen_dict["$DEFINES$"] = [
            """
            #define OutputDim_x {}\n
            #define OutputDim_y {}\n
            #define Stride_x {}\n
            #define Stride_y {}\n
            #define NumChannels {}\n
            #define SIMD {}\n
            """.format(
                odim_w,
                odim_h,
                stride_w,
                stride_h,
                self.get_nodeattr("NumChannels"),
                self.get_nodeattr("SIMD"),
            )
        ]

    def docompute(self):
        in_t = self.get_input_datatype().get_hls_datatype_str()
        odim_h, odim_w = self.get_padded_odim()
        stride_h, stride_w = self.get_nodeattr("Stride")
        hls_call = "FMPadding_Pixel_Nonsquare"
        self.code_gen_dict["$DOCOMPUTE$"] = [
            """{}<OutputDim_x, OutputDim_y, Stride_x, Stride_y, NumChannels,
            SIMD, {}> (in0_V, out0_V);""".format(
                hls_call, in_t
            )
        ]

    def blackboxfunction(self):
        packed_bits = self.get_instream_width()
        packed_hls_type = "ap_uint<%d>" % packed_bits
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            "void %s(hls::stream<%s > &in0_V, hls::stream<%s > &out0_V)"
            % (
                self.onnx_node.name,
                packed_hls_type,
                packed_hls_type,
            )
        ]

    def execute_node(self, context, graph):
        HLSBackend.execute_node(self, context, graph)
