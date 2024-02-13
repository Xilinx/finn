# Copyright (c) 2020, Xilinx
# Copyright (C) 2024, Advanced Micro Devices, Inc.
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
import warnings
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.convolutioninputgenerator import (
    ConvolutionInputGenerator,
)
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy

# ONNX i/o tensor shape assumptions for ConvolutionInputGenerator1D:
# input 0 is the input tensor, shape NHWC = (1, IFMDim_H, IFMDim_W, IFMChannels)
# output 0 is the output tensor, shape NHWC:
#     = (1, OFMDim_H, OFMDim_W, (ConvKernelDim_H*ConvKernelDim_W)*IFMChannels)

# note: the actual data layout produced by the hlslib kernels is different
# for depthwise and non-depthwise ops.
# * non-depthwise SWG: (1, OFMDim_H, OFMDim_W, K_H, K_W, IFMChannels/SIMD, SIMD)
# * depthwise SWG: (1, OFMDim_H, OFMDim_W, IFMChannels/SIMD, K_H, K_W, SIMD)
# see test_fpgadataflow_slidingwindow.py for an example of how to transform
# between the two layouts


class ConvolutionInputGenerator_hls(ConvolutionInputGenerator, HLSBackend):
    """Class that corresponds to one of the 1D finn-hlslib ConvolutionInputGenerator
    (sliding window) function variants. Depending on the combination of
    attributes (e.g. depthwise or not, whether dilation is 0) a different
    variant will be picked for the actual HLS implementation."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(ConvolutionInputGenerator.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def get_swu_variant(self):
        # checks which variant of the ConvolutionInputGenerator (SWU) can be used
        # For the 2D case, we have 4 variants:
        # ConvolutioninputGenerator, ConvolutioninputGenerator_dws,
        # ConvolutioninputGenerator_kernel_stride, ConvolutioninputGenerator_kernel_stride_dws
        # For the 1D case, we have 5 variants: ConvolutionInputGenerator_1D_parallel,
        # ConvolutionInputGenerator_1D_dws_naive, ConvolutionInputGenerator_1D,
        # ConvolutioninputGenerator_1D_dws, ConvolutionInputGenerator_1D_dws_stride
        is_dws = self.get_nodeattr("depthwise")
        if self.get_nodeattr("is1D"):
            is_strided = np.prod(self.get_nodeattr("Stride")) > 1
            is_stride_2 = np.prod(self.get_nodeattr("Stride")) == 2
            is_dilated = np.prod(self.get_nodeattr("Dilation")) > 1
            if self.use_parallel_window_output():
                return "ConvolutionInputGenerator_1D_parallel"
            if not is_dws:
                return "ConvolutionInputGenerator_1D"
            if is_dws:
                if (is_strided and not is_stride_2) or (is_dilated):
                    return "ConvolutionInputGenerator_1D_dws_naive"
                elif is_stride_2:
                    return "ConvolutionInputGenerator_1D_dws_stride"
                else:
                    return "ConvolutionInputGenerator_1D_dws"
        else:
            k = self.get_nodeattr("ConvKernelDim")[0]
            stride = self.get_nodeattr("Stride")[0]
            hls_call = "ConvolutionInputGenerator"
            if k % stride != 0:
                hls_call += "_kernel_stride"
            if is_dws:
                hls_call += "_dws"
            return hls_call

    def use_parallel_window_output(self):
        if not self.get_nodeattr("is1D"):
            return False
        # If 1D, check if simple "ConvolutionInputGenerator_1D_parallel" variant can be used to
        # feed window in parallel to the following layer, enabling full SIMD unfolding.
        stride = self.get_nodeattr("Stride")
        dilation = self.get_nodeattr("Dilation")
        stride_h, stride_w = stride
        dilation_h, dilation_w = dilation
        ram_style = self.get_nodeattr("ram_style")

        fully_unfolded = self.get_nodeattr("SIMD") == self.get_nodeattr("IFMChannels")
        non_dws = self.get_nodeattr("depthwise") == 0
        no_stride = stride_h == 1 and stride_w == 1
        no_dilation = dilation_h == 1 and dilation_w == 1
        supported_ram_style = ram_style in ["auto", "distributed"]
        if self.get_nodeattr("parallel_window") == 1:
            if fully_unfolded and non_dws and no_stride and no_dilation and supported_ram_style:
                return True
            else:
                warnings.warn(
                    "{}: Parallel window output variant is not supported for this node,\
                     please inspect requirements in use_parallel_window_output method\
                     of the custom_op".format(
                        self.onnx_node.name
                    )
                )
                return False
        else:
            return False

    def get_exp_cycles(self):
        simd = self.get_nodeattr("SIMD")
        # 2D case
        if not self.get_nodeattr("is1D"):
            ifm_ch = self.get_nodeattr("IFMChannels")
            k_h, k_w = self.get_nodeattr("ConvKernelDim")
            ifm_dim_h, ifm_dim_w = self.get_nodeattr("IFMDim")
            ofm_dim_h, ofm_dim_w = self.get_nodeattr("OFMDim")
            stride_h, stride_w = self.get_nodeattr("Stride")
            dilation_h, dilation_w = self.get_nodeattr("Dilation")

            # since mmv != 1 is not supported yet, we set mmv for now to 1
            mmv = 1
            # see https://github.com/Xilinx/finn-hlslib/blob/master/slidingwindow.h
            cycles_write_block = (ofm_dim_w * k_w * k_h * (ifm_ch / simd)) / mmv
            cycles_read_block = stride_w * ifm_dim_w * (ifm_ch / simd)
            max_cycles = max(cycles_write_block, cycles_read_block)
            exp_cycles = ifm_dim_w * k_h * dilation_h * (ifm_ch / simd) + ofm_dim_h * max_cycles
        # 1D case
        else:
            (
                ifm_ch,
                [ifm_dim_h, ifm_dim_w],
                [ofm_dim_h, ofm_dim_w],
                [k_h, k_w],
                [stride_h, stride_w],
                [dilation_h, dilation_w],
            ) = self.get_1d_conv_attrs_normalized()

            swu_variant = self.get_swu_variant()
            if swu_variant == "ConvolutionInputGenerator_1D_parallel":
                exp_cycles = k_w + ofm_dim_w
            elif swu_variant == "ConvolutionInputGenerator_1D":
                exp_cycles = 1 + ofm_dim_w * k_w * ifm_ch / simd
            elif swu_variant in [
                "ConvolutionInputGenerator_1D_dws",
                "ConvolutionInputGenerator_1D_dws_stride",
            ]:
                exp_cycles = (
                    1 + ofm_dim_w * k_w * ifm_ch / simd + (ifm_ch / simd) * (k_w - 1) - (k_w - 1)
                )
            elif swu_variant == "ConvolutionInputGenerator_1D_dws_naive":
                cycles_read_block = ifm_dim_w * ifm_ch / simd
                cycles_write_block = ofm_dim_w * k_w * ifm_ch / simd
                exp_cycles = cycles_read_block + cycles_write_block

        return int(exp_cycles)

    def bram_estimation(self):
        simd = self.get_nodeattr("SIMD")
        is1D = self.get_nodeattr("is1D")
        if not is1D:
            ifm_ch = self.get_nodeattr("IFMChannels")
            ifm_dim = self.get_nodeattr("IFMDim")[0]
            k = self.get_nodeattr("ConvKernelDim")[0]
            stride = self.get_nodeattr("Stride")[0]
        else:
            (
                ifm_ch,
                [ifm_dim_h, ifm_dim_w],
                [ofm_dim_h, ofm_dim_w],
                [k_h, k_w],
                [stride_h, stride_w],
                [dilation_h, dilation_w],
            ) = self.get_1d_conv_attrs_normalized()
        ram_style = self.get_nodeattr("ram_style")
        swu_variant = self.get_swu_variant()
        if swu_variant == "ConvolutionInputGenerator_1D_parallel":
            return 0
        if ram_style == "block" or ram_style == "auto":
            if not is1D:
                ram_depth = ifm_dim * ifm_ch / simd
            else:
                if swu_variant == "ConvolutionInputGenerator_1D":
                    ram_depth = (k_w - 1) * ifm_ch / simd
                elif swu_variant == "ConvolutionInputGenerator_1D_dws_naive":
                    ram_depth = ifm_dim_w * ifm_ch / simd
                elif swu_variant in [
                    "ConvolutionInputGenerator_1D_dws",
                    "ConvolutionInputGenerator_1D_dws_stride",
                ]:
                    ram_depth = k_w * ifm_ch / simd
            # after calculate the ram_depth depending on the variant
            # determine ram_width
            if ram_depth <= 512:
                ram_width = 36
            elif ram_depth <= 1024:
                ram_width = 18
            elif ram_depth <= 2048:
                ram_width = 9
            elif ram_depth <= 4096:
                ram_width = 4
            elif ram_depth <= 8192:
                ram_width = 2
            else:
                ram_width = 1

            width_mul = math.ceil(simd * self.get_input_datatype().bitwidth() / ram_width)
            if not is1D:
                depth_mul = math.ceil(ifm_dim * ifm_ch / simd / ram_depth)
                return int((k + stride) * width_mul * depth_mul)
            else:
                depth_mul = math.ceil(ram_depth / 18432)
                return int(width_mul * depth_mul)
        else:
            return 0

    def lut_estimation(self):
        simd = self.get_nodeattr("SIMD")
        is1D = self.get_nodeattr("is1D")
        if not is1D:
            ifm_ch = self.get_nodeattr("IFMChannels")
            ifm_dim = self.get_nodeattr("IFMDim")[0]
            k = self.get_nodeattr("ConvKernelDim")[0]
            stride = self.get_nodeattr("Stride")[0]
        else:
            (
                ifm_ch,
                [ifm_dim_h, ifm_dim_w],
                [ofm_dim_h, ofm_dim_w],
                [k_h, k_w],
                [stride_h, stride_w],
                [dilation_h, dilation_w],
            ) = self.get_1d_conv_attrs_normalized()
        ram_style = self.get_nodeattr("ram_style")
        swu_variant = self.get_swu_variant()
        if swu_variant == "ConvolutionInputGenerator_1D_parallel":
            ram_luts = math.ceil(simd * self.get_input_datatype().bitwidth() * (k_w + 1) / 64)
        if ram_style == "distributed":
            if not is1D:
                ram_luts = int(
                    (k + stride)
                    * (
                        simd
                        * self.get_input_datatype().bitwidth()
                        * math.ceil(ifm_dim * ifm_ch / simd / 64)
                    )
                )
            if swu_variant == "ConvolutionInputGenerator_1D":
                ram_luts = math.ceil(self.get_input_datatype().bitwidth() * (k_w - 1) * ifm_ch / 64)
            elif swu_variant == "ConvolutionInputGenerator_1D_dws_naive":
                ram_luts = math.ceil(self.get_input_datatype().bitwidth() * ifm_dim_w * ifm_ch / 64)
            elif swu_variant in [
                "ConvolutionInputGenerator_1D_dws",
                "ConvolutionInputGenerator_1D_dws_stride",
            ]:
                ram_luts = math.ceil(self.get_input_datatype().bitwidth() * k_w * ifm_ch / 64)
        else:
            ram_luts = 0
        return 300 + ram_luts

    def uram_estimation(self):
        simd = self.get_nodeattr("SIMD")
        is1D = self.get_nodeattr("is1D")
        if not is1D:
            ifm_ch = self.get_nodeattr("IFMChannels")
            ifm_dim = self.get_nodeattr("IFMDim")[0]
            k = self.get_nodeattr("ConvKernelDim")[0]
            stride = self.get_nodeattr("Stride")[0]
        else:
            (
                ifm_ch,
                [ifm_dim_h, ifm_dim_w],
                [ofm_dim_h, ofm_dim_w],
                [k_h, k_w],
                [stride_h, stride_w],
                [dilation_h, dilation_w],
            ) = self.get_1d_conv_attrs_normalized()

        ram_style = self.get_nodeattr("ram_style")
        swu_variant = self.get_swu_variant()
        if swu_variant == "ConvolutionInputGenerator_1D_parallel":
            return 0
        if ram_style == "ultra":
            if not is1D:
                return int(
                    (k + stride)
                    * (
                        math.ceil(simd * self.get_input_datatype().bitwidth() / 64)
                        * math.ceil(ifm_dim * ifm_ch / simd / 4096)
                    )
                )
            else:
                if swu_variant == "ConvolutionInputGenerator_1D":
                    width_mul = math.ceil(simd * self.get_input_datatype().bitwidth() / 72)
                    depth_mul = math.ceil((k_w - 1) * ifm_ch / simd / 4096)
                    return width_mul * depth_mul
                elif swu_variant == "ConvolutionInputGenerator_1D_dws_naive":
                    width_mul = math.ceil(simd * self.get_input_datatype().bitwidth() / 72)
                    depth_mul = math.ceil(ifm_dim_w * ifm_ch / simd / 4096)
                    return width_mul * depth_mul
                elif swu_variant in [
                    "ConvolutionInputGenerator_1D_dws",
                    "ConvolutionInputGenerator_1D_dws_stride",
                ]:
                    width_mul = math.ceil(simd * self.get_input_datatype().bitwidth() / 72)
                    depth_mul = math.ceil(k_w * ifm_ch / simd / 4096)
                    return width_mul * depth_mul
        else:
            return 0

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node
        exp_ishape = self.get_normal_input_shape()
        exp_oshape = self.get_normal_output_shape()
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
        assert (
            inp.shape == exp_ishape
        ), """Input shape doesn't
        match expected shape (1, ifm_dim, ifm_dim, ifm_ch)."""
        if self.get_input_datatype() == DataType["BIPOLAR"]:
            # store bipolar activations as binary
            inp = (inp + 1) / 2
            export_idt = DataType["BINARY"]
        else:
            export_idt = self.get_input_datatype()
        # reshape input into folded form
        inp = inp.reshape(folded_ishape)
        # make copy before saving array
        reshaped_input = inp.copy()
        np.save(os.path.join(code_gen_dir, "input_0.npy"), reshaped_input)

        if mode == "cppsim":
            # execute the precompiled model
            super().exec_precompiled_singlenode_model()
            # load output npy file
            super().npy_to_dynamic_output(context)
            assert (
                context[node.output[0]].shape == exp_oshape
            ), "cppsim \
            did not produce expected output shape"
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
        # binary -> bipolar if needed
        if self.get_output_datatype() == DataType["BIPOLAR"]:
            out = context[node.output[0]]
            out = 2 * out - 1
            context[node.output[0]] = out
        assert (
            context[node.output[0]].shape == exp_oshape
        ), """Output
        shape doesn't match expected shape (1, ofm_dim_h, ofm_dim_w, k_h*k_w*ifm_ch)."""

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "slidingwindow.h"']

    def defines(self, var):
        numReps = 1
        is1D = self.get_nodeattr("is1D")
        simd = self.get_nodeattr("SIMD")
        ifm_precision = self.get_input_datatype().bitwidth()
        if not is1D:
            ifm_dim = self.get_nodeattr("IFMDim")[0]
            ifm_ch = self.get_nodeattr("IFMChannels")
            ofm_dim = self.get_nodeattr("OFMDim")[0]
            k = self.get_nodeattr("ConvKernelDim")[0]
            stride = self.get_nodeattr("Stride")[0]
        else:
            (
                ifm_ch,
                [ifm_dim_h, ifm_dim_w],
                [ofm_dim_h, ofm_dim_w],
                [k_h, k_w],
                [stride_h, stride_w],
                [dilation_h, dilation_w],
            ) = self.get_1d_conv_attrs_normalized()

        swu_variant = self.get_swu_variant()

        # check all different 1D scenarios
        if swu_variant in [
            "ConvolutionInputGenerator_1D_parallel",
            "ConvolutionInputGenerator_1D",
            "ConvolutionInputGenerator_1D_dws_stride",
        ]:
            self.code_gen_dict["$DEFINES$"] = [
                """
            #define ConvKernelDim1_x {}\n
            #define IFMChannels1 {}\n
            #define Input_precision1 {}\n
            #define IFMDim1_x {}\n
            #define OFMDim1_x {}\n
            #define Stride1_x {}\n
            #define SIMD1 {}\n
            #define numReps {}
            """.format(
                    k_w,
                    ifm_ch,
                    ifm_precision,
                    ifm_dim_w,
                    ofm_dim_w,
                    stride_w,
                    simd,
                    numReps,
                )
            ]
        elif swu_variant == "ConvolutionInputGenerator_1D_dws":
            self.code_gen_dict["$DEFINES$"] = [
                """
            #define ConvKernelDim1_x {}\n
            #define IFMChannels1 {}\n
            #define Input_precision1 {}\n
            #define IFMDim1_x {}\n
            #define OFMDim1_x {}\n
            #define SIMD1 {}\n
            #define numReps {}
            """.format(
                    k_w,
                    ifm_ch,
                    ifm_precision,
                    ifm_dim_w,
                    ofm_dim_w,
                    simd,
                    numReps,
                )
            ]
        elif swu_variant == "ConvolutionInputGenerator_1D_dws_naive":
            self.code_gen_dict["$DEFINES$"] = [
                """
            #define ConvKernelDim1_x {}\n
            #define IFMChannels1 {}\n
            #define Input_precision1 {}\n
            #define IFMDim1_x {}\n
            #define OFMDim1_x {}\n
            #define Stride1_x {}\n
            #define Dilation1_x {}\n
            #define SIMD1 {}\n
            #define numReps {}
            """.format(
                    k_w,
                    ifm_ch,
                    ifm_precision,
                    ifm_dim_w,
                    ofm_dim_w,
                    stride_w,
                    dilation_w,
                    simd,
                    numReps,
                )
            ]
        # default to 2D cases
        else:
            self.code_gen_dict["$DEFINES$"] = [
                """#define ConvKernelDim1 {}\n #define IFMChannels1 {}\n
                #define Input_precision1 {}\n #define IFMDim1 {}\n
                #define OFMDim1 {}\n #define SIMD1 {}\n
                #define Stride1 {}\n #define numReps {}""".format(
                    k, ifm_ch, ifm_precision, ifm_dim, ofm_dim, simd, stride, numReps
                )
            ]

    def docompute(self):
        ram_style = self.get_nodeattr("ram_style")
        map_to_hls_ram_style = {
            "auto": "ap_resource_dflt()",
            "block": "ap_resource_bram()",
            "distributed": "ap_resource_lutram()",
            "ultra": "ap_resource_uram()",
        }
        hls_ram_style = map_to_hls_ram_style[ram_style]
        swu_variant = self.get_swu_variant()

        # check which 1D ConvolutionInputGenerator is needed
        if swu_variant == "ConvolutionInputGenerator_1D_parallel":
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """{}<ConvKernelDim1_x, IFMChannels1, Input_precision1,
                IFMDim1_x, OFMDim1_x, Stride1_x, SIMD1>
                (in0_{}, out_{}, numReps, {});""".format(
                    swu_variant, self.hls_sname(), self.hls_sname(), hls_ram_style
                )
            ]
        elif swu_variant == "ConvolutionInputGenerator_1D":
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """{}<ConvKernelDim1_x, IFMChannels1, Input_precision1,
                IFMDim1_x, OFMDim1_x, Stride1_x, SIMD1>
                (in0_{}, out_{}, numReps, {});""".format(
                    swu_variant, self.hls_sname(), self.hls_sname(), hls_ram_style
                )
            ]
        elif swu_variant == "ConvolutionInputGenerator_1D_dws":
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """{}<ConvKernelDim1_x, IFMChannels1, Input_precision1,
                IFMDim1_x, OFMDim1_x, SIMD1>
                (in0_{}, out_{}, numReps, {});""".format(
                    swu_variant, self.hls_sname(), self.hls_sname(), hls_ram_style
                )
            ]
        elif swu_variant == "ConvolutionInputGenerator_1D_dws_stride":
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """{}<ConvKernelDim1_x, IFMChannels1, Input_precision1,
                IFMDim1_x, OFMDim1_x, Stride1_x, SIMD1>
                (in0_{}, out_{}, numReps, {});""".format(
                    swu_variant, self.hls_sname(), self.hls_sname(), hls_ram_style
                )
            ]
        elif swu_variant == "ConvolutionInputGenerator_1D_dws_naive":
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """{}<ConvKernelDim1_x, IFMChannels1, Input_precision1,
                IFMDim1_x, OFMDim1_x, Stride1_x, Dilation1_x, SIMD1>
                (in0_{}, out_{}, numReps, {});""".format(
                    swu_variant, self.hls_sname(), self.hls_sname(), hls_ram_style
                )
            ]
        else:
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """{}<ConvKernelDim1, IFMChannels1, Input_precision1, IFMDim1,
                    OFMDim1, SIMD1, Stride1> (in0_{}, out_{}, numReps, {});""".format(
                    swu_variant, self.hls_sname(), self.hls_sname(), hls_ram_style
                )
            ]

    def dataoutstrm(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_output_datatype()
        if dtype == DataType["BIPOLAR"]:
            # use binary for bipolar storage
            dtype = DataType["BINARY"]
        elem_bits = dtype.bitwidth()
        packed_bits = self.get_outstream_width()
        packed_hls_type = "ap_uint<%d>" % packed_bits
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"
        npy_out = "%s/output.npy" % code_gen_dir
        oshape = self.get_folded_output_shape()
        oshape_cpp_str = str(oshape).replace("(", "{").replace(")", "}")
        if self.use_parallel_window_output():
            # pass the number of pixels in the folded output to apintstream2npy, needed
            # to unpack the ouput correctly and reverse only the inner SIMD dimension
            k_h, k_w = self.get_nodeattr("ConvKernelDim")
            multi_pixel_out = k_h * k_w
        else:
            multi_pixel_out = 1

        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            'apintstream2npy<%s, %s, %d, %s>(out_%s, %s, "%s", true, 1, %d);'
            % (
                packed_hls_type,
                elem_hls_type,
                elem_bits,
                npy_type,
                self.hls_sname(),
                oshape_cpp_str,
                npy_out,
                multi_pixel_out,
            )
        ]

    def blackboxfunction(self):
        if self.use_parallel_window_output():
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                """void {}(hls::stream<ap_uint<SIMD1*Input_precision1>> &in0_{},
                    hls::stream<ap_uint<ConvKernelDim1_x*SIMD1*Input_precision1>>
                    &out_{})""".format(
                    self.onnx_node.name, self.hls_sname(), self.hls_sname()
                )
            ]
        else:
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                """void {}(hls::stream<ap_uint<SIMD1*Input_precision1>> &in0_{},
                    hls::stream<ap_uint<SIMD1*Input_precision1>> &out_{})""".format(
                    self.onnx_node.name, self.hls_sname(), self.hls_sname()
                )
            ]
