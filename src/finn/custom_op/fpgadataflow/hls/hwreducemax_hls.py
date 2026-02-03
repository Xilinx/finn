############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

import numpy as np

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.hwreducemax import HWReduceMax


class HWReduceMax_hls(HWReduceMax, HLSBackend):

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(HWReduceMax.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = [
            "#include <iostream>",
            "#include <hls_vector.h>",
            '#include "reducemax.hpp"',
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
                static hls::stream<hls::vector<TI,SIMD>> src0;
#pragma HLS stream variable=src0 depth=W/SIMD
                static hls::stream<hls::vector<TI,SIMD>> dst0;
#pragma HLS stream variable=dst0 depth=2

                move(in0_V, src0);
                static ReduceMax<TI, TI, W, SIMD> rm_inst;
                rm_inst.execute(src0, dst0);
                
                hls::vector<TI,SIMD> result_vec = dst0.read();
                out0_V.write(result_vec);
        """
        ]

    def blackboxfunction(self):
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            f"""
            void {self.onnx_node.name}(
                hls::stream<hls::vector<TI,SIMD>> &in0_V,
                hls::stream<hls::vector<TI,SIMD>> &out0_V
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
        self.code_gen_dict["$TIMEOUT_VALUE$"] = [str(np.prod(self.get_normal_input_shape()))]

    def dataoutstrm(self):
        self.code_gen_dict["$DATAOUTSTREAM$"] = []
        for o in range(self.get_number_output_values()):
            folded_shape = self.get_folded_output_shape(o)
            shape_cpp_str = f"""{{{','.join((str(i) for i in folded_shape))}}}"""
            dtype_str = "float"
            out_vector = f"strm{o}" if self.get_nodeattr("hls_style") == "freerunning" else f"out{o}_V"
            self.code_gen_dict["$DATAOUTSTREAM$"].append(
                f"vectorstream2npy<{dtype_str}, {dtype_str}, 1>({out_vector}, {shape_cpp_str}, "
                f'"{self.get_nodeattr("code_gen_dir_cppsim")}/output_{o}.npy");'
            )