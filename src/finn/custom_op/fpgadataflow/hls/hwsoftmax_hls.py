############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3 Clause
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

import numpy as np

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.hwsoftmax import HWSoftmax


class HWSoftmax_hls(HWSoftmax, HLSBackend):
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(HWSoftmax.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        # override HLSBackend defaults for the SoftMax kernel
        my_attrs["cpp_interface"] = ("s", False, "hls_vector")
        my_attrs["hls_style"] = ("s", False, "freerunning")
        return my_attrs

    def get_exp_cycles(self):
        # SoftMax is a 3-stage dataflow pipeline (max, exp+sum, divide) with
        # depth=N/SIMD FIFOs between stages. Each vector incurs pipeline stall
        # overhead as stages wait for max/sum values before proceeding.
        folded = self.get_folded_input_shape()
        n_beats = int(np.prod(folded[:-1]))
        n = self.get_normal_input_shape()[-1]
        simd = self.get_nodeattr("SIMD")
        beats_per_vec = max(1, n // simd)
        num_vectors = n_beats // beats_per_vec
        # Per-vector stalls (max/sum reads) + pipeline fill + FP operation latencies
        return n_beats + 2 * num_vectors + 2 * beats_per_vec + 30

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = [
            "#include <hls_vector.h>",
            '#include "softmax.hpp"',
            '#include "utils.hpp"',
        ]

    def defines(self, var):
        simd = self.get_nodeattr("SIMD")
        idtype = self.get_input_datatype()
        w = self.get_nodeattr("ifm_dim")[-1]
        self.code_gen_dict["$DEFINES$"] = [
            f"""
            constexpr unsigned  SIMD = {simd};
            constexpr unsigned  W = {w};
            using  TI = {idtype.get_hls_datatype_str()};
            using  F = float;
           """
        ]

    def docompute(self):
        self.code_gen_dict["$DOCOMPUTE$"] = [
            """
                static hls::stream<hls::vector<TI,SIMD>>  src0;
                static hls::stream<hls::vector<float,SIMD>>  dst0;

                move(in0_V, src0);
                static SoftMax<TI, float, W, SIMD> sm_inst;
                sm_inst.execute(src0, dst0);
                move(dst0, out0_V);
        """
        ]

    def blackboxfunction(self):
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            f"""
            void {self.onnx_node.name}(
                hls::stream<hls::vector<TI,SIMD>> &in0_V,
                hls::stream<hls::vector<float,SIMD>> &out0_V
                )
            """
        ]

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"] = [
            """
            #pragma HLS interface AXIS port=in0_V
            #pragma HLS interface AXIS port=out0_V
            #pragma HLS aggregate  variable=in0_V compact=bit
            #pragma HLS aggregate  variable=out0_V compact=bit

            #pragma HLS interface ap_ctrl_none port=return
            #pragma HLS dataflow disable_start_propagation
            """
        ]

    def execute_node(self, context, graph):
        HLSBackend.execute_node(self, context, graph)

    def timeout_value(self):
        """Set timeout value for HLS functions defined for one clock cycle"""
        self.code_gen_dict["$TIMEOUT_VALUE$"] = [str(np.prod(self.get_normal_input_shape()))]
