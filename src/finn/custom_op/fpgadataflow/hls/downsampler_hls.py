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

import numpy as np
import os

from finn.custom_op.fpgadataflow.downsampler import DownSampler
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy


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
        sname = self.hls_sname()
        self.code_gen_dict["$DOCOMPUTE$"] = [
            f"""ConvolutionInputGenerator_{dim_var}_kernel1<IFMChannels, Input_precision,
            IFMDim, SIMD,Stride> (in0_{sname}, out_{sname}, numReps);"""
        ]

    def blackboxfunction(self):
        packed_bits = self.get_instream_width()
        packed_hls_type = "ap_uint<%d>" % packed_bits
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            "void %s(hls::stream<%s > &in0_%s, hls::stream<%s > &out_%s)"
            % (
                self.onnx_node.name,
                packed_hls_type,
                self.hls_sname(),
                packed_hls_type,
                self.hls_sname(),
            )
        ]

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node
        exp_ishape = self.get_normal_input_shape()
        exp_oshape = self.get_normal_output_shape()
        folded_ishape = self.get_folded_input_shape()

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
        assert (
            inp.shape == exp_ishape
        ), """Input shape doesn't
        match expected shape (numInputVectors, ImgDim, ImgDim, NumChannels)."""
        export_idt = self.get_input_datatype()

        reshaped_input = inp.reshape(folded_ishape)
        np.save(os.path.join(code_gen_dir, "input_0.npy"), reshaped_input)

        if mode == "cppsim":
            # execute the precompiled model
            super().exec_precompiled_singlenode_model()
            # load output npy file
            super().npy_to_dynamic_output(context)
            assert (
                context[node.output[0]].shape == exp_oshape
            ), "cppsim did not produce expected output shape"
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
            output = np.load(out_npy_path)
            output = np.asarray([output], dtype=np.float32).reshape(*exp_oshape)
            context[node.output[0]] = output
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )
        assert (
            context[node.output[0]].shape == exp_oshape
        ), """Output shape doesn't match expected shape
            (1, OutputDim, OutputDim, NumChannels)."""
