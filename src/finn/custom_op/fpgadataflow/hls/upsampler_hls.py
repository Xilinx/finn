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

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.upsampler import UpsampleNearestNeighbour


class UpsampleNearestNeighbour_hls(UpsampleNearestNeighbour, HLSBackend):
    """
    Corresponds to finn-hlslib UpsampleNearestNeighbour function.
    Upsampling is done with the Nearest Neighbour algorithm.
    The layer expects square feature maps for the in and output.
    """

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(UpsampleNearestNeighbour.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "upsample.hpp"']

    def defines(self, var):
        self.code_gen_dict["$DEFINES$"] = []

        HI = self.get_nodeattr("HI")
        self.code_gen_dict["$DEFINES$"] += ["#define HI {}".format(HI)]

        WI = self.get_nodeattr("WI")
        self.code_gen_dict["$DEFINES$"] += ["#define WI {}".format(WI)]

        HO = self.get_nodeattr("HO")
        self.code_gen_dict["$DEFINES$"] += ["#define HO {}".format(HO)]

        WO = self.get_nodeattr("WO")
        self.code_gen_dict["$DEFINES$"] += ["#define WO {}".format(WO)]

        SIMD = self.get_nodeattr("SIMD")

        CF = self.get_nodeattr("NumChannels") // SIMD
        self.code_gen_dict["$DEFINES$"] += ["#define CF {}".format(CF)]

    def docompute(self):
        self.code_gen_dict["$DOCOMPUTE$"] = [
            """upsample_nn<HI, HO, WI, WO, CF, CF>(in0_V, out0_V);"""
        ]

    def blackboxfunction(self):
        simd = self.get_nodeattr("SIMD")
        input_elem_hls_type = self.get_input_datatype().get_hls_datatype_str()
        output_elem_hls_type = self.get_output_datatype().get_hls_datatype_str()
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            "void %s(hls::stream<hls::vector<%s, %d>> "
            "&in0_V, hls::stream<hls::vector<%s, %d>> &out0_V)"
            % (
                self.onnx_node.name,
                input_elem_hls_type,
                simd,
                output_elem_hls_type,
                simd,
            )
        ]

    def execute_node(self, context, graph):
        HLSBackend.execute_node(self, context, graph)
