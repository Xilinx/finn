###################################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright for portions of this file is held by AMD and Microsoft under
# MIT license as part of project Brainsmith.
# All other copyright is held by AMD and is provided under BSD-3-Clause license.
#
###################################################################################

from finn.custom_op.fpgadataflow.crop import Crop
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend


class Crop_hls(Crop, HLSBackend):
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        return Crop.get_nodeattr_types(self) | HLSBackend.get_nodeattr_types(self)

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = [
            '#include "crop.hpp"',
        ]

    def defines(self, var):
        simd = self.get_nodeattr("SIMD")
        dtype = self.get_input_datatype()
        height, width = self.get_nodeattr("ImgDim")
        if height == 0:
            # pretend that height is 1 for code generation
            height = 1
        ch = self.get_nodeattr("NumChannels")
        self.code_gen_dict["$DEFINES$"] = [
            f"""
            constexpr unsigned  SIMD      = {simd};
            constexpr unsigned  H      = {height};
            constexpr unsigned  W      = {width};
            constexpr unsigned  CF     = {ch // simd};
            constexpr unsigned  CROP_N = {self.get_nodeattr("CropNorth")};
            constexpr unsigned  CROP_E = {self.get_nodeattr("CropEast")};
            constexpr unsigned  CROP_S = {self.get_nodeattr("CropSouth")};
            constexpr unsigned  CROP_W = {self.get_nodeattr("CropWest")};
            using  TV = hls::vector<{dtype.get_hls_datatype_str()}, SIMD>;
            """
        ]

    def docompute(self):
        self.code_gen_dict["$DOCOMPUTE$"] = [
            """
            hls::stream<TV>  src0;
            hls::stream<TV>  dst0;
            #pragma HLS stream variable=src0 depth=2
            #pragma HLS stream variable=dst0 depth=2

            move(in0_V, src0);
            crop< H, W,	CF, CROP_N, CROP_E, CROP_S, CROP_W, TV>(src0, dst0);
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
