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

import math
import numpy as np

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.streamingdatawidthconverter import (
    StreamingDataWidthConverter,
)

# does not do anything at the ONNX node-by-node level, and input-output
# tensor shapes are the same. performs data width conversion at the rtlsim level


class StreamingDataWidthConverter_hls(StreamingDataWidthConverter, HLSBackend):
    """Class that corresponds to finn-hlslib StreamingDataWidthConverterGeneralized_Batch
    function."""

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(StreamingDataWidthConverter.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "streamtools.h"']

    def get_ap_int_max_w(self):
        # the generalized kernel uses an inWidth+outWidth intermediate buffer
        base = super().get_ap_int_max_w()
        return max(base, self.get_instream_width() + self.get_outstream_width())

    def defines(self, var):
        # in cases of convolution input generator and downsampling,
        # we have a 4D input and padding / cropping can only happen
        # for the final 2 dimensions,
        # so we use numReps to represent the first 2 dimensions
        # + batching if shape[0] != 1
        numReps = int(np.prod(self.get_folded_input_shape()[:-2]))

        # assuming folded shapes are at least 2 dim-long
        numInWords = int(np.prod(self.get_folded_input_shape()[-2:-1]))
        numOutWords = int(np.prod(self.get_folded_output_shape()[-2:-1]))

        inWidth = self.get_nodeattr("inWidth")
        outWidth = self.get_nodeattr("outWidth")

        self.code_gen_dict["$DEFINES$"] = [
            "#define InWidth %d " % inWidth,
            "#define OutWidth %d " % outWidth,
            "#define NumInWords %d " % numInWords,
            "#define NumOutWords %d " % numOutWords,
            "#define numReps %d" % numReps,
        ]

    def strm_decl(self):
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> in0_V ("in0_V");'.format(self.get_instream_width())
        )

        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> out0_V ("out0_V");'.format(self.get_outstream_width())
        )

    def docompute(self):
        # TODO continue with fxns below, they are copy-pasted
        # the generalized kernel handles arbitrary width ratios plus
        # padding/cropping in a single call (no LCM cascade needed)
        op = "StreamingDataWidthConverterGeneralized_Batch"
        self.code_gen_dict["$DOCOMPUTE$"] = [
            "%s<InWidth, OutWidth, NumInWords, NumOutWords>(in0_V, out0_V, numReps);" % op
        ]

    def blackboxfunction(self):
        in_packed_bits = self.get_instream_width()
        in_packed_hls_type = "ap_uint<%d>" % in_packed_bits
        out_packed_bits = self.get_outstream_width()
        out_packed_hls_type = "ap_uint<%d>" % out_packed_bits
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            "void %s(hls::stream<%s > &in0_V, hls::stream<%s > &out0_V)"
            % (
                self.onnx_node.name,
                in_packed_hls_type,
                out_packed_hls_type,
            )
        ]

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"] = ["#pragma HLS INTERFACE axis port=in0_V"]
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=out0_V")
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE ap_ctrl_none port=return")

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node
        if mode == "cppsim":
            in_shape = self.get_normal_input_shape()
            out_shape = self.get_normal_output_shape()
            inp = context[node.input[0]]
            assert str(inp.dtype) == "float32", "Input datatype is not float32"
            assert inp.shape == tuple(in_shape), "Input shape does not match expected shape."

            # cppsim emulates the DWC in numpy: pad the trailing elements per frame
            # with zeros, or crop them. The generated C++ is intentionally not run --
            # the cppsim npy flow reverses the endianness of the leftover word when a
            # larger input word is split into smaller output words. TODO: fix cppsim.
            output = np.zeros(out_shape, dtype=np.float32)
            if out_shape[-1] > in_shape[-1]:
                output[..., : in_shape[-1]] = inp[..., : in_shape[-1]]
            else:
                output[..., : out_shape[-1]] = inp[..., : out_shape[-1]]
            output = np.asarray([output], dtype=np.float32).reshape(*out_shape)
            context[node.output[0]] = output
        elif mode == "rtlsim":
            HLSBackend.execute_node(self, context, graph)
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )

    def lut_estimation(self):
        """Calculates resource estimations for LUTs"""

        # TODO: This calculation does not currently take into account the extra
        # tracking variables, nor the muxing of one of the stream ports to the buffer
        # which shifts according to how many elements are in the buffer
        # the true LUT cost is between 2*(inw+outw) and 10*(inw+outw)

        inw = self.get_instream_width()
        outw = self.get_outstream_width()

        # we use an intermediate buffer of size inwidth+outwidth
        intw = inw + outw

        # we assume a shift-based implementation
        # even if we don't use LUTs explicitly, we make some unavailable
        # to other logic because they're tied into the DWC control sets

        cnt_luts = 0
        cset_luts = 0

        cnt_luts += abs(math.ceil(math.log(intw / inw, 2)))

        cset_luts += intw + outw

        # generalized DWC cost penalty, this value is temporary
        cnt_luts *= 8

        return int(cnt_luts + cset_luts)
