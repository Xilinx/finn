############################################################################
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3 Clause
#
############################################################################

import numpy as np

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.layernorm import LayerNorm


class LayerNorm_hls(LayerNorm, HLSBackend):
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(LayerNorm.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        my_attrs.update(
            {
                "cpp_interface": ("s", False, "hls_vector"),
                "hls_style": ("s", False, "freerunning"),
            }
        )
        return my_attrs

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = [
            "#include <hls_vector.h>",
            '#include "layernorm.hpp"',
        ]

    def defines(self, var):
        simd = self.get_nodeattr("SIMD")
        idtype = self.get_input_datatype()
        n = self.get_nodeattr("ifm_dim")[-1]
        self.code_gen_dict["$DEFINES$"] = [
            f"""
            constexpr unsigned  SIMD = {simd};
            constexpr unsigned  N = {n};
            using  TI = {idtype.get_hls_datatype_str()};
            using  TO = float;
           """
        ]

    def docompute(self):
        self.code_gen_dict["$DOCOMPUTE$"] = ["layernorm<N>(in0_V, out0_V);"]

    def blackboxfunction(self):
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            f"""
            void {self.onnx_node.name}(
                hls::stream<hls::vector<TI,SIMD>> &in0_V,
                hls::stream<hls::vector<TO,SIMD>> &out0_V
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
        timeout_value = max(np.prod(self.get_normal_input_shape()), 100)
        self.code_gen_dict["$TIMEOUT_VALUE$"] = [str(timeout_value)]
