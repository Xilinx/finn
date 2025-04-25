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

from finn.custom_op.fpgadataflow.fmpadding import FMPadding
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend


class FMPadding_hls(FMPadding, HLSBackend):
    """Corresponds to finn-hlslib FMPadding_Batch function.
    Pads input image by given amount."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(FMPadding.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "streamtools.h"']

    def defines(self, var):
        idim_h, idim_w = self.get_nodeattr("ImgDim")
        odim_h, odim_w = self.get_padded_odim()
        pad = self.get_nodeattr("Padding")
        pad_h = pad[0] + pad[2]
        pad_w = pad[1] + pad[3]
        is_square_img = idim_h == idim_w
        is_square_pad = pad_h == pad_w

        if is_square_img and is_square_pad:
            self.code_gen_dict["$DEFINES$"] = [
                """#define ImgDim1 {}\n#define OutputDim1 {}\n
                #define PaddingBefore1 {}\n#define PaddingBehind1 {}\n
                #define NumChannels1 {}\n#define SIMD1 {}\n
                #define numReps {}\n""".format(
                    idim_h,
                    odim_h,
                    pad[0],
                    pad[2],
                    self.get_nodeattr("NumChannels"),
                    self.get_nodeattr("SIMD"),
                    self.get_nodeattr("numInputVectors"),
                )
            ]
        else:
            self.code_gen_dict["$DEFINES$"] = [
                """
                #define OutputDim1_x {}\n
                #define OutputDim1_y {}\n
                #define PaddingLeft1 {}\n
                #define PaddingRight1 {}\n
                #define PaddingTop1 {}\n
                #define PaddingBottom1 {}\n
                #define NumChannels1 {}\n
                #define SIMD1 {}\n
                #define numReps {}\n
                """.format(
                    odim_w,
                    odim_h,
                    pad[1],
                    pad[3],
                    pad[0],
                    pad[2],
                    self.get_nodeattr("NumChannels"),
                    self.get_nodeattr("SIMD"),
                    self.get_nodeattr("numInputVectors"),
                )
            ]

    def docompute(self):
        in_t = self.get_input_datatype().get_hls_datatype_str()
        idim_h, idim_w = self.get_nodeattr("ImgDim")
        pad = self.get_nodeattr("Padding")
        pad_h = pad[0] + pad[2]
        pad_w = pad[1] + pad[3]
        is_square_img = idim_h == idim_w
        is_square_pad = pad_h == pad_w

        if is_square_img and is_square_pad:
            hls_call = "FMPadding_Batch"
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """{}<ImgDim1, OutputDim1, PaddingBefore1, PaddingBehind1, NumChannels1, SIMD1,
                {}> (in0_V, out0_V, numReps);""".format(
                    hls_call, in_t
                )
            ]
        else:
            hls_call = "FMPadding_nonsquare_Batch"
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """{}<OutputDim1_x, OutputDim1_y, PaddingLeft1, PaddingRight1,
                PaddingTop1, PaddingBottom1, NumChannels1,
                SIMD1, {}> (in0_V, out0_V, numReps);""".format(
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
