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

from qonnx.custom_op.general.maxpoolnhwc import compute_pool_output_dim

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.streamingmaxpool import StreamingMaxPool


class StreamingMaxPool_hls(StreamingMaxPool, HLSBackend):
    """Class that corresponds to finn-hlslib StreamingMaxPool_batch function."""

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(StreamingMaxPool.get_nodeattr_types(self))
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

        # verify the number of inputs
        if len(self.onnx_node.input) == 1:
            info_messages.append("The number of inputs is correct")
        else:
            info_messages.append("""StreamingMaxPool_Batch needs 1 data input""")

        return info_messages

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "maxpool.h"']

    def defines(self, var):
        numReps = 1
        ifm_dim, k, ifm_ch = self.get_1d_attrs_normalized()
        ceil_mode = self.get_nodeattr("CeilMode")
        output_size = compute_pool_output_dim(ifm_dim[1], k[1], k[1], 0, ceil_mode)

        if self.is_1d():
            self.code_gen_dict["$DEFINES$"] = [
                """#define ImgDim {}\n #define PoolDim {}\n
                #define NumChannels {}\n #define PE {}\n #define OutputSize {}
                \n #define numReps {}""".format(
                    ifm_dim[1],
                    k[1],
                    self.get_nodeattr("NumChannels"),
                    self.get_nodeattr("PE"),
                    output_size,
                    numReps,
                )
            ]
        else:
            self.code_gen_dict["$DEFINES$"] = [
                """#define ImgDim {}\n #define PoolDim {}\n
                #define NumChannels {}\n #define numReps {}""".format(
                    ifm_dim[1],
                    k[1],
                    self.get_nodeattr("NumChannels"),
                    numReps,
                )
            ]

    def docompute(self):
        dtype = self.get_input_datatype()
        if dtype.bitwidth() == 1:
            if self.is_1d():
                raise Exception("Binary 1d MaxPool not implemented on HLS backend")
            else:
                op = "StreamingMaxPool"
            self.code_gen_dict["$DOCOMPUTE$"] = [
                "%s<ImgDim, PoolDim, NumChannels>(in0_V, out0_V);" % op
            ]
        else:
            dtype = self.get_input_datatype()
            dtype_hls = dtype.get_hls_datatype_str()
            minval_str = str(int(dtype.min()))
            if self.is_1d():
                op = "StreamingMaxPool_Precision_1d"
                self.code_gen_dict["$DOCOMPUTE$"] = [
                    """%s<ImgDim, PoolDim, NumChannels, PE,
                     OutputSize, %s, %s>(in0_V, out0_V);"""
                    % (op, dtype_hls, minval_str)
                ]
            else:
                op = "StreamingMaxPool_Precision"
                self.code_gen_dict["$DOCOMPUTE$"] = [
                    "%s<ImgDim, PoolDim, NumChannels, %s, %s>(in0_V, out0_V);"
                    % (op, dtype_hls, minval_str)
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
