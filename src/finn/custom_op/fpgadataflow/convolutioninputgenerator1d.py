# Copyright (c) 2020, Xilinx
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
from qonnx.custom_op.general.im2col import compute_conv_output_dim

from finn.custom_op.fpgadataflow.hlscustomop import HLSCustomOp
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy

# This operation should only be used for 1D convolutions. Either the
# IFMDim_H or IFMDim_W should be '1', which represents the so-called
# dummy-dimension

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


class ConvolutionInputGenerator1D(HLSCustomOp):
    """Class that corresponds to one of the 1D finn-hlslib ConvolutionInputGenerator
    (sliding window) function variants. Depending on the combination of
    attributes (e.g. depthwise or not, whether dilation is 0) a different
    variant will be picked for the actual HLS implementation."""

    def __init__(self, onnx_node):
        super().__init__(onnx_node)

    def get_nodeattr_types(self):
        my_attrs = {
            "ConvKernelDim": ("ints", True, []),  # [H, W] = [Y, X]
            "IFMChannels": ("i", True, 0),
            "IFMDim": ("ints", True, []),  # [H, W] = [Y, X]
            "OFMDim": ("ints", True, []),  # [H, W] = [Y, X]
            "SIMD": ("i", True, 0),
            "Stride": ("ints", True, []),  # [H, W] = [Y, X]
            "Dilation": ("ints", True, []),  # [H, W] = [Y, X]
            # FINN DataTypes for inputs, weights, outputs
            "inputDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
            "depthwise": ("i", False, 0, {0, 1}),
            # FPGA resource type for ConvolutionInputGenerator input buffer
            # auto -- let Vivado HLS decide
            # block -- use BRAM
            # distributed -- use LUTRAM
            # ultra -- use URAM
            "ram_style": (
                "s",
                False,
                "distributed",
                {"auto", "block", "distributed", "ultra"},
            ),
            "parallel_window": ("i", False, 0, {0, 1}),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_normal_input_shape(self):
        ifm_dim_h, ifm_dim_w = self.get_nodeattr("IFMDim")
        ifm_ch = self.get_nodeattr("IFMChannels")
        ishape = (1, ifm_dim_h, ifm_dim_w, ifm_ch)
        return ishape

    def get_folded_input_shape(self):
        ifm_dim_h, ifm_dim_w = self.get_nodeattr("IFMDim")
        ifm_ch = self.get_nodeattr("IFMChannels")
        simd = self.get_nodeattr("SIMD")
        assert ifm_ch % simd == 0, "SIMD must divide IFMChannels"
        wf = int(ifm_ch / simd)
        folded_ishape = (1, ifm_dim_h, ifm_dim_w, wf, simd)
        return folded_ishape

    def get_normal_output_shape(self):
        k_h, k_w = self.get_nodeattr("ConvKernelDim")
        ifm_dim_h, ifm_dim_w = self.get_nodeattr("IFMDim")
        ifm_ch = self.get_nodeattr("IFMChannels")
        stride_h, stride_w = self.get_nodeattr("Stride")
        dilation_h, dilation_w = self.get_nodeattr("Dilation")
        pad = 0
        ofm_dim_h = compute_conv_output_dim(ifm_dim_h, k_h, stride_h, pad, dilation_h)
        ofm_dim_w = compute_conv_output_dim(ifm_dim_w, k_w, stride_w, pad, dilation_w)
        oshape = (1, ofm_dim_h, ofm_dim_w, k_h * k_w * ifm_ch)
        return oshape

    def get_folded_output_shape(self):
        k_h, k_w = self.get_nodeattr("ConvKernelDim")
        ifm_dim_h, ifm_dim_w = self.get_nodeattr("IFMDim")
        ifm_ch = self.get_nodeattr("IFMChannels")
        stride_h, stride_w = self.get_nodeattr("Stride")
        dilation_h, dilation_w = self.get_nodeattr("Dilation")
        simd = self.get_nodeattr("SIMD")
        pad = 0
        ofm_dim_h = compute_conv_output_dim(ifm_dim_h, k_h, stride_h, pad, dilation_h)
        ofm_dim_w = compute_conv_output_dim(ifm_dim_w, k_w, stride_w, pad, dilation_w)
        assert ifm_ch % simd == 0, "SIMD must divide IFMChannels"
        if self.use_parallel_window_output():
            wf = int((ifm_ch) // simd)
            folded_oshape = (1, ofm_dim_h, ofm_dim_w, wf, k_h * k_w * simd)
        else:
            wf = int((k_h * k_w * ifm_ch) // simd)
            folded_oshape = (1, ofm_dim_h, ofm_dim_w, wf, simd)
        return folded_oshape

    def make_shape_compatible_op(self, model):
        exp_ishape = self.get_normal_input_shape()
        oshape = self.get_normal_output_shape()
        ishape = tuple(model.get_tensor_shape(self.onnx_node.input[0]))
        assert ishape == exp_ishape, "Unexpect input shape for ConvInpGen."
        return super().make_const_shape_op(oshape)

    def infer_node_datatype(self, model):
        node = self.onnx_node
        # data type stays the same
        dtype = model.get_tensor_datatype(node.input[0])
        model.set_tensor_datatype(node.output[0], dtype)

    def verify_node(self):
        pass

    def get_input_datatype(self):
        """Returns FINN DataType of input."""
        return DataType[self.get_nodeattr("inputDataType")]

    def get_output_datatype(self):
        """Returns FINN DataType of output."""
        return DataType[self.get_nodeattr("outputDataType")]

    def get_instream_width(self):
        ibits = self.get_input_datatype().bitwidth()
        simd = self.get_nodeattr("SIMD")
        ifm_ch = self.get_nodeattr("IFMChannels")
        assert ifm_ch % simd == 0, "SIMD must divide IFMChannels"
        in_width = simd * ibits
        return in_width

    def get_outstream_width(self):
        if self.use_parallel_window_output():
            # feed all window pixels in parallel
            k_h, k_w = self.get_nodeattr("ConvKernelDim")
            return self.get_instream_width() * k_h * k_w
        else:
            # if parallel variant not in use: same width for output and input stream
            return self.get_instream_width()

    def get_number_output_values(self):
        folded_oshape = self.get_folded_output_shape()
        num_output_elems = np.prod(folded_oshape[:-1])
        return num_output_elems

    def get_swu_variant(self):
        # checks which variant of the 1D ConvolutionInputGenerator (SWU) can be used
        # We have 5 variants: ConvolutionInputGenerator_1D_parallel,
        # ConvolutionInputGenerator_1D_dws_naive, ConvolutionInputGenerator_1D,
        # ConvolutioninputGenerator_1D_dws, ConvolutionInputGenerator_1D_dws_stride
        is_dws = self.get_nodeattr("depthwise")
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

    def get_1d_conv_attrs_normalized(self):
        # support both (1, D) and (D, 1) cases transparently:
        # For the kernel, presenting the input data of size D as
        # [H, W] = [Y, X] = [1, D] or [D, 1]
        # effectively gives the same result.
        # For consistency and ease of programming, this function
        # returns the attributes of the layer as follows:
        # [H, W] = [Y, X] = [1, D] or [D, 1] are always mapped to [1, D].
        # The dummy ('1') dimension is the Y-dimension.
        ifm_ch = self.get_nodeattr("IFMChannels")
        k = self.get_nodeattr("ConvKernelDim")
        ifm_dim = self.get_nodeattr("IFMDim")
        ofm_dim = self.get_nodeattr("OFMDim")
        stride = self.get_nodeattr("Stride")
        dilation = self.get_nodeattr("Dilation")

        # see defines() for an explanation
        if ifm_dim[1] == 1:
            ifm_dim = ifm_dim[::-1]
            ofm_dim = ofm_dim[::-1]
            k = k[::-1]
            stride = stride[::-1]
            dilation = dilation[::-1]

        return (ifm_ch, ifm_dim, ofm_dim, k, stride, dilation)

    def use_parallel_window_output(self):
        # Check if simple "ConvolutionInputGenerator_1D_parallel" variant can be used to
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
            if (
                fully_unfolded
                and non_dws
                and no_stride
                and no_dilation
                and supported_ram_style
            ):
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
        (
            ifm_ch,
            [ifm_dim_h, ifm_dim_w],
            [ofm_dim_h, ofm_dim_w],
            [k_h, k_w],
            [stride_h, stride_w],
            [dilation_h, dilation_w],
        ) = self.get_1d_conv_attrs_normalized()

        # since mmv != 1 is not supported yet, we set mmv for now to 1
        # mmv = 1
        # see https://github.com/Xilinx/finn-hlslib/blob/master/slidingwindow.h
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
                1
                + ofm_dim_w * k_w * ifm_ch / simd
                + (ifm_ch / simd) * (k_w - 1)
                - (k_w - 1)
            )
        elif swu_variant == "ConvolutionInputGenerator_1D_dws_naive":
            cycles_read_block = ifm_dim_w * ifm_ch / simd
            cycles_write_block = ofm_dim_w * k_w * ifm_ch / simd
            exp_cycles = cycles_read_block + cycles_write_block

        return int(exp_cycles)

    def bram_estimation(self):
        simd = self.get_nodeattr("SIMD")
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
            if swu_variant == "ConvolutionInputGenerator_1D":
                ram_depth = (k_w - 1) * ifm_ch / simd
            elif swu_variant == "ConvolutionInputGenerator_1D_dws_naive":
                ram_depth = ifm_dim_w * ifm_ch / simd
            elif swu_variant in [
                "ConvolutionInputGenerator_1D_dws",
                "ConvolutionInputGenerator_1D_dws_stride",
            ]:
                ram_depth = k_w * ifm_ch / simd
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
            width_mul = math.ceil(
                simd * self.get_input_datatype().bitwidth() / ram_width
            )
            depth_mul = math.ceil(ram_depth / 18432)
            return width_mul * depth_mul
        else:
            return 0

    def lut_estimation(self):
        simd = self.get_nodeattr("SIMD")
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
            ram_luts = math.ceil(
                simd * self.get_input_datatype().bitwidth() * (k_w + 1) / 64
            )
        elif ram_style == "distributed":
            if swu_variant == "ConvolutionInputGenerator_1D":
                ram_luts = math.ceil(
                    self.get_input_datatype().bitwidth() * (k_w - 1) * ifm_ch / 64
                )
            elif swu_variant == "ConvolutionInputGenerator_1D_dws_naive":
                ram_luts = math.ceil(
                    self.get_input_datatype().bitwidth() * ifm_dim_w * ifm_ch / 64
                )
            elif swu_variant in [
                "ConvolutionInputGenerator_1D_dws",
                "ConvolutionInputGenerator_1D_dws_stride",
            ]:
                ram_luts = math.ceil(
                    self.get_input_datatype().bitwidth() * k_w * ifm_ch / 64
                )
        else:
            ram_luts = 0
        return 300 + ram_luts

    def uram_estimation(self):
        simd = self.get_nodeattr("SIMD")
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
        elif ram_style == "ultra":
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
        (
            ifm_ch,
            [ifm_dim_h, ifm_dim_w],
            [ofm_dim_h, ofm_dim_w],
            [k_h, k_w],
            [stride_h, stride_w],
            [dilation_h, dilation_w],
        ) = self.get_1d_conv_attrs_normalized()
        simd = self.get_nodeattr("SIMD")
        ifm_precision = self.get_input_datatype().bitwidth()
        swu_variant = self.get_swu_variant()

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
        if swu_variant == "ConvolutionInputGenerator_1D_dws":
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
        if swu_variant == "ConvolutionInputGenerator_1D_dws_naive":
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

    def read_npy_data(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_input_datatype()
        if dtype == DataType["BIPOLAR"]:
            # use binary for bipolar storage
            dtype = DataType["BINARY"]
        elem_bits = dtype.bitwidth()
        packed_bits = self.get_instream_width()
        packed_hls_type = "ap_uint<%d>" % packed_bits
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"
        npy_in = "%s/input_0.npy" % code_gen_dir
        self.code_gen_dict["$READNPYDATA$"] = []
        self.code_gen_dict["$READNPYDATA$"].append(
            'npy2apintstream<%s, %s, %d, %s>("%s", in0);'
            % (packed_hls_type, elem_hls_type, elem_bits, npy_type, npy_in)
        )

    def strm_decl(self):
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> in0 ("in0");'.format(self.get_instream_width())
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> out ("out");'.format(self.get_outstream_width())
        )

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

        # check which ConvolutionInputGenerator is needed
        if swu_variant == "ConvolutionInputGenerator_1D_parallel":
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """{}<ConvKernelDim1_x, IFMChannels1, Input_precision1,
                IFMDim1_x, OFMDim1_x, Stride1_x, SIMD1>
                (in0, out, numReps, {});""".format(
                    swu_variant, hls_ram_style
                )
            ]
        if swu_variant == "ConvolutionInputGenerator_1D":
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """{}<ConvKernelDim1_x, IFMChannels1, Input_precision1,
                IFMDim1_x, OFMDim1_x, Stride1_x, SIMD1>
                (in0, out, numReps, {});""".format(
                    swu_variant, hls_ram_style
                )
            ]
        if swu_variant == "ConvolutionInputGenerator_1D_dws":
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """{}<ConvKernelDim1_x, IFMChannels1, Input_precision1,
                IFMDim1_x, OFMDim1_x, SIMD1>
                (in0, out, numReps, {});""".format(
                    swu_variant, hls_ram_style
                )
            ]
        if swu_variant == "ConvolutionInputGenerator_1D_dws_stride":
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """{}<ConvKernelDim1_x, IFMChannels1, Input_precision1,
                IFMDim1_x, OFMDim1_x, Stride1_x, SIMD1>
                (in0, out, numReps, {});""".format(
                    swu_variant, hls_ram_style
                )
            ]
        if swu_variant == "ConvolutionInputGenerator_1D_dws_naive":
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """{}<ConvKernelDim1_x, IFMChannels1, Input_precision1,
                IFMDim1_x, OFMDim1_x, Stride1_x, Dilation1_x, SIMD1>
                (in0, out, numReps, {});""".format(
                    swu_variant, hls_ram_style
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
            'apintstream2npy<%s, %s, %d, %s>(out, %s, "%s", true, 1, %d);'
            % (
                packed_hls_type,
                elem_hls_type,
                elem_bits,
                npy_type,
                oshape_cpp_str,
                npy_out,
                multi_pixel_out,
            )
        ]

    def save_as_npy(self):
        self.code_gen_dict["$SAVEASCNPY$"] = []

    def blackboxfunction(self):
        if self.use_parallel_window_output():
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                """void {}(hls::stream<ap_uint<SIMD1*Input_precision1>> &in0,
                    hls::stream<ap_uint<ConvKernelDim1_x*SIMD1*Input_precision1>>
                    &out)""".format(
                    self.onnx_node.name
                )
            ]
        else:
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                """void {}(hls::stream<ap_uint<SIMD1*Input_precision1>> &in0,
                    hls::stream<ap_uint<SIMD1*Input_precision1>> &out)""".format(
                    self.onnx_node.name
                )
            ]

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"] = [
            "#pragma HLS INTERFACE axis port=in0 name=in0_" + self.hls_sname()
        ]
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE axis port=out name=out_" + self.hls_sname()
        )
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE ap_ctrl_none port=return"
        )
