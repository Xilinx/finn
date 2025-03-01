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
import os
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.streamingdatawidthconverter import (
    StreamingDataWidthConverter,
)
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy

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
            'hls::stream<ap_uint<{}>> in0_{} ("in0_{}");'.format(
                self.get_instream_width(), self.hls_sname(), self.hls_sname()
            )
        )

        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> out_{} ("out_{}");'.format(
                self.get_outstream_width(), self.hls_sname(), self.hls_sname()
            )
        )

    def docompute(self):
        # TODO continue with fxns below, they are copy-pasted
        op = "StreamingDataWidthConverterGeneralized_Batch"

        self.code_gen_dict["$DOCOMPUTE$"] = [
            "%s<InWidth, OutWidth, NumInWords,NumOutWords" % op
            + ">(in0_%s, out_%s, numReps);" % (self.hls_sname(), self.hls_sname())
        ]

    def blackboxfunction(self):
        in_packed_bits = self.get_instream_width()
        in_packed_hls_type = "ap_uint<%d>" % in_packed_bits
        out_packed_bits = self.get_outstream_width()
        out_packed_hls_type = "ap_uint<%d>" % out_packed_bits
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            "void %s(hls::stream<%s > &in0_%s, hls::stream<%s > &out_%s)"
            % (
                self.onnx_node.name,
                in_packed_hls_type,
                self.hls_sname(),
                out_packed_hls_type,
                self.hls_sname(),
            )
        ]

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"] = [
            "#pragma HLS INTERFACE axis port=in0_" + self.hls_sname()
        ]
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE axis port=out_" + self.hls_sname()
        )
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE ap_ctrl_none port=return")

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node
        exp_shape = self.get_normal_input_shape()
        folded_ishape = self.get_folded_input_shape()

        # TODO ensure codegen dir exists
        if mode == "cppsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        elif mode == "rtlsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )

        inp = context[node.input[0]]
        assert str(inp.dtype) == "float32", "Input datatype is not float32"
        assert inp.shape == tuple(exp_shape), "Input shape does not match expected shape."

        if self.get_input_datatype() == DataType["BIPOLAR"]:
            # store bipolar activations as binary
            inp = (inp + 1) / 2
            export_idt = DataType["BINARY"]
        else:
            export_idt = self.get_input_datatype()
        # reshape input into folded shape

        reshaped_input = inp.reshape(folded_ishape)
        np.save(os.path.join(code_gen_dir, "input_0.npy"), reshaped_input)

        exp_shape = self.get_normal_output_shape()

        if mode == "cppsim":
            # cppsim simply passes through the values because
            # the DWC fails some test cases due to
            # endianness differences in the cppsim flow
            # of passing numpy arrays. TODO: Fix?
            # Essentially need to fix cppsim to reverse
            # endian and then back same as rtlsim
            # for this particular (and maybe all) cases
            # only shows up for the DWC, since when a word
            # leftover appears when breaking down larger in
            # words to smaller out words, the remainder should
            # now be the LSB, but is the other way around on the
            # cpp output.

            in_shape = self.get_normal_input_shape()
            out_shape = self.get_normal_output_shape()
            inp = context[node.input[0]]
            assert str(inp.dtype) == "float32", "Input datatype is not float32"
            assert inp.shape == tuple(in_shape), "Input shape does not match expected shape."

            # initialize as zeroes to introduce padding if needed
            output = np.zeros((out_shape), dtype=np.float32)
            if out_shape[-1] > in_shape[-1]:
                output[..., : in_shape[-1]] = inp[..., : in_shape[-1]]
            else:
                output[..., : out_shape[-1]] = inp[..., : out_shape[-1]]

            output = np.asarray([output], dtype=np.float32).reshape(*out_shape)
            context[node.output[0]] = output

        elif mode == "rtlsim":
            sim = self.get_rtlsim()
            nbits = self.get_instream_width()
            rtlsim_inp = npy_to_rtlsim_input(
                "{}/input_0.npy".format(code_gen_dir), export_idt, nbits
            )
            super().reset_rtlsim(sim)
            super().toggle_clk(sim)
            rtlsim_output = self.rtlsim(sim, rtlsim_inp)
            odt = export_idt
            target_bits = odt.bitwidth()
            packed_bits = self.get_outstream_width()

            out_npy_path = "{}/output.npy".format(code_gen_dir)
            out_shape = self.get_folded_output_shape()

            rtlsim_output_to_npy(
                rtlsim_output, out_npy_path, odt, out_shape, packed_bits, target_bits
            )

            # load and reshape output
            output_pre_reshape = np.load(out_npy_path)
            output = np.asarray([output_pre_reshape], dtype=np.float32).reshape(exp_shape)
            context[node.output[0]] = output

        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to "rtlsim" """.format(
                    mode
                )
            )
        # binary -> bipolar if needed
        if self.get_output_datatype() == DataType["BIPOLAR"]:
            out = context[node.output[0]]
            out = 2 * out - 1
            context[node.output[0]] = out
        assert context[node.output[0]].shape == tuple(
            exp_shape
        ), """Output
        shape doesn't match expected shape, should be same as input shape"""

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
