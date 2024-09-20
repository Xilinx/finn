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

import os
import numpy as np

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.upsampler import UpsampleNearestNeighbour
from finn.custom_op.fpgadataflow import templates
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy


class UpsampleNearestNeighbour_hls(UpsampleNearestNeighbour, HLSBackend):
    """
    Corresponds to finn-hlslib UpsampleNearestNeighbour_Batch function.
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

    def verify_node(self):
        pass

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
        self.code_gen_dict["$DEFINES$"] += ["#define SIMD {}".format(SIMD)]

        CF = self.get_nodeattr("NumChannels") // SIMD
        self.code_gen_dict["$DEFINES$"] += ["#define CF {}".format(CF)]

    def docompute(self):
        self.code_gen_dict["$DOCOMPUTE$"] = [
            """upsample_nn<HI, WI, HO, WO, CF>(in0_%s, out_%s);"""
            % (self.hls_sname(), self.hls_sname())
        ]

    def blackboxfunction(self):
        input_elem_hls_type = self.get_input_datatype().get_hls_datatype_str()
        output_elem_hls_type = self.get_output_datatype().get_hls_datatype_str()
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            "void %s(hls::stream<hls::vector<%s, SIMD>> &in0_%s, hls::stream<hls::vector<%s, SIMD>> &out_%s)"
            % (
                self.onnx_node.name,
                input_elem_hls_type,
                self.hls_sname(),
                output_elem_hls_type,
                self.hls_sname(),
            )
        ]

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node
        exp_ishape = self.get_normal_input_shape()
        exp_oshape = self.get_normal_output_shape()

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
        self.dynamic_input_to_npy(context, 1, target_dir=code_gen_dir)

        if mode == "cppsim":
            # execute the precompiled model
            super().exec_precompiled_singlenode_model()
            # load output npy file
            super().npy_to_dynamic_output(context)
            assert (
                context[node.output[0]].shape == exp_oshape
            ), "cppsim did not produce expected folded output shape"
            context[node.output[0]] = context[node.output[0]].reshape(*exp_oshape)
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

    def read_npy_data(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        npy_type = "float"
        self.code_gen_dict["$READNPYDATA$"] = []
        input_elem_hls_type = self.get_input_datatype().get_hls_datatype_str()
        npy_in = "%s/input_0.npy" % (code_gen_dir)
        self.code_gen_dict["$READNPYDATA$"].append(
            'npy2vectorstream<%s, %s, SIMD>("%s", in0_%s);'
            % (
                input_elem_hls_type,
                npy_type,
                npy_in,
                self.hls_sname(),
            )
        )

    def dataoutstrm(self):
        npy_type = "float"
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        oshape = self.get_folded_output_shape()
        oshape_cpp_str = str(oshape).replace("(", "{").replace(")", "}")
        npy_out = "%s/output.npy" % code_gen_dir
        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            'vectorstream2npy<%s, %s, SIMD>(out_%s, %s, "%s");'
            % (
                self.get_output_datatype().get_hls_datatype_str(),
                npy_type,
                self.hls_sname(),
                oshape_cpp_str,
                npy_out,
            )
        ]
    
    def strm_decl(self):
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<hls::vector<{}, SIMD>> in0_{} ("in0_{}");'.format(
                self.get_input_datatype().get_hls_datatype_str(),
                self.hls_sname(),
                self.hls_sname()
            )
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<hls::vector<{}, SIMD>> out_{} ("out_{}");'.format(
                self.get_output_datatype().get_hls_datatype_str(),
                self.hls_sname(),
                self.hls_sname()
            )
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<hls::vector<{}, SIMD>> debug_out_{} ("debug_out_{}");'.format(
                self.get_output_datatype().get_hls_datatype_str(),
                self.hls_sname(),
                self.hls_sname()
            )
        )

    def pragmas(self):
        super().pragmas()
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS aggregate variable=in0_%s compact=bit" % self.hls_sname())
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS aggregate variable=out_%s compact=bit" % self.hls_sname())
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS dataflow disable_start_propagation")
        