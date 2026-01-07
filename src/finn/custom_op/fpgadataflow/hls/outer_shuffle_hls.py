############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

import math
import numpy as np
from typing import Optional

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.outer_shuffle import OuterShuffle


def auto_size_simd(I_dim: int, SIMD: int) -> Optional[int]:
    """
    Return the smallest divisor d of I_dim such that d > SIMD.
    if no such divisor exists, return None.
    """
    if I_dim <= 0:
        raise ValueError("I_dim must be a positive integer")
    if SIMD < 0:
        raise ValueError("SIMD must be a non-negative integer")

    candidates = []
    limit = int(math.isqrt(I_dim))
    for a in range(1, limit + 1):
        if I_dim % a == 0:
            b = I_dim // a
            if a > SIMD:
                candidates.append(a)
            if b > SIMD:
                candidates.append(b)

    if not candidates:
        return None

    return min(candidates)


class OuterShuffle_hls(OuterShuffle, HLSBackend):
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

        # check some constraints that it is a legal shuffle_hls
        last_dim = self.get_nodeattr("transpose_in_shape")[-1]
        SIMD = self.get_nodeattr("SIMD")
        if last_dim % SIMD != 0:
            new_simd = auto_size_simd(last_dim, SIMD)
            if new_simd is not None:
                self.set_nodeattr("SIMD", new_simd)
            else:
                raise RuntimeError("Unable to determine a new SIMD value for this transpose.")

    def get_nodeattr_types(self):
        return OuterShuffle.get_nodeattr_types(self) | HLSBackend.get_nodeattr_types(self)

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = [
            '#include "input_gen.hpp"',
            "#include <ap_int.h>",
            "#include <hls_vector.h>",
            "#include <hls_stream.h>",
        ]

    def defines(self, var):
        simd = self.get_nodeattr("SIMD")
        dtype = self.get_input_datatype()
        self.code_gen_dict["$DEFINES$"] = [
            f"""
            constexpr unsigned  SIMD = {simd};
            using  TE = {dtype.get_hls_datatype_str()};
            using  TV = hls::vector<TE, SIMD>;
            """
        ]

    def get_exp_cycles(self):
        out_shape = self.get_nodeattr("transpose_out_shape")
        simd = self.get_nodeattr("SIMD")
        return int(np.prod(out_shape) / simd)

    def docompute(self):
        simd = self.get_nodeattr("SIMD")
        out_shape = self.get_nodeattr("transpose_out_shape")
        out_shape[-1] = int(out_shape[-1] / simd)
        loop_coeffs = [1 if x == 1 else int(x / simd) for x in self.get_nodeattr("loop_coeffs")]
        interleaved = [int(item) for pair in zip(out_shape, loop_coeffs) for item in pair]
        self.code_gen_dict["$DOCOMPUTE$"] = [
            f"""
            hls::stream<TV>  src0;
            hls::stream<TV>  dst0;
            #pragma HLS stream variable=src0 depth=2
            #pragma HLS stream variable=dst0 depth=2

            move(in0_V, src0);
            input_gen<-1,{np.prod(out_shape)},{','.join(map(str,interleaved))}>(src0, dst0);
            move(dst0, out0_V);

            """
        ]

    def blackboxfunction(self):
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            f"""
            void {self.onnx_node.name} (
                hls::stream<TV> &in0_V,
                hls::stream<TV> &out0_V
            )
            """
        ]

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"] = [
            """
            #pragma HLS interface AXIS port=in0_V
            #pragma HLS interface AXIS port=out0_V
            #pragma HLS aggregate variable=in0_V compact=bit
            #pragma HLS aggregate variable=out0_V compact=bit

            #pragma HLS interface ap_ctrl_none port=return
            #pragma HLS dataflow disable_start_propagation
            """
        ]

    def execute_node(self, context, graph):
        HLSBackend.execute_node(self, context, graph)

    def timeout_value(self):
        """Set timeout value for HLS functions defined for one clock cycle"""
        self.code_gen_dict["$TIMEOUT_VALUE$"] = [str(np.prod(self.get_normal_input_shape()))]
