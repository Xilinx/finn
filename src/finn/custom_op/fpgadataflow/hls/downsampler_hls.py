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

from finn.custom_op.fpgadataflow.downsampler import DownSampler
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend


class DownSampler_hls(DownSampler, HLSBackend):
    """Corresponds to finn-hlslib ConvolutionInputGenerator_*_kernel1 function.
    Basically performs a down sampling of the image removing rows and columns."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(DownSampler.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "slidingwindow.h"']

    def defines(self, var):
        self.code_gen_dict["$DEFINES$"] = []

        ifm_ch = self.get_nodeattr("NumChannels")
        self.code_gen_dict["$DEFINES$"] += ["#define IFMChannels {}".format(ifm_ch)]

        ibits = self.get_input_datatype().bitwidth()
        self.code_gen_dict["$DEFINES$"] += ["#define Input_precision {}".format(ibits)]

        idim = self.get_nodeattr("ImgDim")
        self.code_gen_dict["$DEFINES$"] += ["#define IFMDim {}".format(idim)]

        simd = self.get_nodeattr("SIMD")
        self.code_gen_dict["$DEFINES$"] += ["#define SIMD {}".format(simd)]

        stride = self.get_nodeattr("Stride")
        self.code_gen_dict["$DEFINES$"] += ["#define Stride {}".format(stride)]

        batch_size = self.get_nodeattr("numInputVectors")
        self.code_gen_dict["$DEFINES$"] += ["#define numReps {}".format(batch_size)]

    def docompute(self):
        dim_var = "1D" if (self.get_nodeattr("is1D") == 1) else "2D"
        self.code_gen_dict["$DOCOMPUTE$"] = [
            f"""ConvolutionInputGenerator_{dim_var}_kernel1<IFMChannels, Input_precision,
            IFMDim, SIMD,Stride> (in0_V, out0_V, numReps);"""
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
