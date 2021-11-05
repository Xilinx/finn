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

from finn.core.datatype import DataType
from finn.custom_op.fpgadataflow.hlscustomop import HLSCustomOp
from finn.custom_op.general.im2col import compute_conv_output_dim
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

    def get_1d_conv_attrs_normalized(self):
        # support both (1, D) and (D, 1) cases transparently:
        # For the kernel, presenting the input data of size D as
        # [H, W] = [Y, X] = [1, D] or [D, 1]
        # effectively gives the same result. Because the
        # ConvolutionInputGenerator_NonSquare_Dilated(_dws) kernel currently only
        # supports dilation>1 along the X-axis and the
        # ConvolutionInputGenerator_NonSquare only works for stride>1 along the
        # X-axis, we are working with the following assumption:
        # the dummy ('1') dimension is the Y-dimension, i.e.
        # images and kernels (and their attributes) of dimension
        # [H, W] = [Y, X] = [D, 1] or [1, D] are always mapped to [1, D]
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

        if self.get_nodeattr("SIMD") == self.get_nodeattr("IFMChannels"):
            if self.get_nodeattr("depthwise") == 0:
                if stride_h == 1 and stride_w == 1:
                    if dilation_h == 1 and dilation_w == 1:
                        return True

        return False

    def get_exp_cycles(self):
        simd = self.get_nodeattr("SIMD")
        (
            ifm_ch,
            ifm_dim,
            ofm_dim,
            k,
            stride,
            dilation,
        ) = self.get_1d_conv_attrs_normalized()
        ifm_dim_h, ifm_dim_w = ifm_dim
        ofm_dim_h, ofm_dim_w = ofm_dim
        k_h, k_w = k
        stride_h, stride_w = stride
        dilation_h, dilation_w = dilation

        # since mmv != 1 is not supported yet, we set mmv for now to 1
        mmv = 1
        # see https://github.com/Xilinx/finn-hlslib/blob/master/slidingwindow.h
        if self.use_parallel_window_output():
            exp_cycles = k_w + ofm_dim_w
        else:
            cycles_write_block = (ofm_dim_w * k_w * k_h * (ifm_ch / simd)) / mmv
            cycles_read_block = stride_w * ifm_dim_w * (ifm_ch / simd)
            max_cycles = max(cycles_write_block, cycles_read_block)
            exp_cycles = (
                ifm_dim_w * k_h * dilation_h * (ifm_ch / simd) + ofm_dim_h * max_cycles
            )

        return int(exp_cycles)

    def bram_estimation(self):
        # NOTE: not tested for correctness
        simd = self.get_nodeattr("SIMD")
        ifm_ch = self.get_nodeattr("IFMChannels")
        ifm_dim = np.prod(self.get_nodeattr("IFMDim"))
        k = np.prod(self.get_nodeattr("ConvKernelDim"))
        stride = np.prod(self.get_nodeattr("Stride"))
        ram_style = self.get_nodeattr("ram_style")
        if ram_style == "block" or ram_style == "auto":
            ram_depth = ifm_dim * ifm_ch / simd
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
            return int(
                (k + stride)
                * (
                    math.ceil(simd * self.get_input_datatype().bitwidth() / ram_width)
                    * math.ceil(ifm_dim * ifm_ch / simd / ram_depth)
                )
            )
        else:
            return 0

    def lut_estimation(self):
        # NOTE: not tested for correctness
        simd = self.get_nodeattr("SIMD")
        ifm_ch = self.get_nodeattr("IFMChannels")
        ifm_dim = np.prod(self.get_nodeattr("IFMDim"))
        k = np.prod(self.get_nodeattr("ConvKernelDim"))
        stride = np.prod(self.get_nodeattr("Stride"))
        ram_style = self.get_nodeattr("ram_style")
        if ram_style == "distributed":
            ram_luts = int(
                (k + stride)
                * (
                    simd
                    * self.get_input_datatype().bitwidth()
                    * math.ceil(ifm_dim * ifm_ch / simd / 64)
                )
            )
        else:
            ram_luts = 0
        return 300 + ram_luts

    def uram_estimation(self):
        # NOTE: not tested for correctness
        simd = self.get_nodeattr("SIMD")
        ifm_ch = self.get_nodeattr("IFMChannels")
        ifm_dim = np.prod(self.get_nodeattr("IFMDim"))
        k = np.prod(self.get_nodeattr("ConvKernelDim"))
        stride = np.prod(self.get_nodeattr("Stride"))
        ram_style = self.get_nodeattr("ram_style")
        if ram_style == "ultra":
            return int(
                (k + stride)
                * (
                    math.ceil(simd * self.get_input_datatype().bitwidth() / 64)
                    * math.ceil(ifm_dim * ifm_ch / simd / 4096)
                )
            )
        else:
            return 0

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node
        exp_ishape = self.get_normal_input_shape()
        exp_oshape = self.get_normal_output_shape()
        folded_ishape = self.get_folded_input_shape()
        folded_oshape = self.get_folded_output_shape()

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
                context[node.output[0]].shape == folded_oshape
            ), "cppsim \
            did not produce expected ofolded utput shape"
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
            ifm_dim,
            ofm_dim,
            k,
            stride,
            dilation,
        ) = self.get_1d_conv_attrs_normalized()
        simd = self.get_nodeattr("SIMD")
        ifm_precision = self.get_input_datatype().bitwidth()
        ifm_dim_y, ifm_dim_x = ifm_dim
        ofm_dim_y, ofm_dim_x = ofm_dim
        k_y, k_x = k
        dilation_y, dilation_x = dilation
        # For a 1d convolution with stride=[S,1] or [1,S], the finn-hlslib function
        # of ConvInpGen must be created with [stride_y, stride_x] = [S, S].
        # TODO: changes in finn-hlslib (slidingwindow.h)
        stride_y = np.prod(stride)
        stride_x = np.prod(stride)

        if dilation_x > 1:
            assert (
                dilation_y == 1
            ), "Dilation value greater than 1 along y-axis is not yet supported"
            self.code_gen_dict["$DEFINES$"] = [
                """
            #define ConvKernelDim1_x {}\n
            #define ConvKernelDim1_y {}\n
            #define IFMChannels1 {}\n
            #define Input_precision1 {}\n
            #define IFMDim1_x {}\n
            #define IFMDim1_y {}\n
            #define OFMDim1_x {}\n
            #define OFMDim1_y {}\n
            #define SIMD1 {}\n
            #define Stride1_x {}\n
            #define Stride1_y {}\n
            #define Dilation1_x {}\n
            #define Dilation1_y {}\n
            #define numReps {}
            """.format(
                    k_x,
                    k_y,
                    ifm_ch,
                    ifm_precision,
                    ifm_dim_x,
                    ifm_dim_y,
                    ofm_dim_x,
                    ofm_dim_y,
                    simd,
                    stride_x,
                    stride_y,
                    dilation_x,
                    dilation_y,
                    numReps,
                )
            ]
        else:
            ofm_dim = self.get_nodeattr("OFMDim")
            self.code_gen_dict["$DEFINES$"] = [
                """
            #define ConvKernelDim1_x {}\n
            #define ConvKernelDim1_y {}\n
            #define IFMChannels1 {}\n
            #define Input_precision1 {}\n
            #define IFMDim1_x {}\n
            #define IFMDim1_y {}\n
            #define OFMDim1_x {}\n
            #define OFMDim1_y {}\n
            #define SIMD1 {}\n
            #define Stride1_x {}\n
            #define Stride1_y {}\n
            #define numReps {}
            """.format(
                    k_x,
                    k_y,
                    ifm_ch,
                    ifm_precision,
                    ifm_dim_x,
                    ifm_dim_y,
                    ofm_dim_x,
                    ofm_dim_y,
                    simd,
                    stride_x,
                    stride_y,
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

        # check which ConvolutionInputGenerator is needed
        if self.use_parallel_window_output():
            hls_call = "ConvolutionInputGenerator_1D_parallel"
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """{}<ConvKernelDim1_x, IFMChannels1, Input_precision1,
                IFMDim1_x, OFMDim1_x, SIMD1, Stride1_x>
                (in0, out, numReps, {});""".format(
                    hls_call, hls_ram_style
                )
            ]
        else:
            hls_call = "ConvolutionInputGenerator_NonSquare"
            dilation_h, dilation_w = self.get_nodeattr("Dilation")
            if dilation_h > 1 or dilation_w > 1:
                hls_call += "_Dilated"
                if self.get_nodeattr("depthwise") == 1:
                    hls_call += "_dws"
                self.code_gen_dict["$DOCOMPUTE$"] = [
                    """{}<ConvKernelDim1_x, ConvKernelDim1_y, IFMChannels1,
                    Input_precision1, IFMDim1_x, IFMDim1_y, OFMDim1_x, OFMDim1_y,
                    SIMD1, Stride1_x, Stride1_y, Dilation1_x, Dilation1_y>
                    (in0, out, numReps, {});""".format(
                        hls_call, hls_ram_style
                    )
                ]
            elif self.get_nodeattr("depthwise") == 1:
                hls_call += "_dws"
                self.code_gen_dict["$DOCOMPUTE$"] = [
                    """{}<ConvKernelDim1_x, ConvKernelDim1_y, IFMChannels1,
                    Input_precision1, IFMDim1_x, IFMDim1_y, OFMDim1_x, OFMDim1_y,
                    SIMD1, Stride1_x, Stride1_y> (in0, out, numReps, {});""".format(
                        hls_call, hls_ram_style
                    )
                ]
            else:
                self.code_gen_dict["$DOCOMPUTE$"] = [
                    """{}<ConvKernelDim1_x, ConvKernelDim1_y, IFMChannels1,
                    Input_precision1, IFMDim1_x, IFMDim1_y, OFMDim1_x, OFMDim1_y,
                    SIMD1, Stride1_x, Stride1_y> (in0, out, numReps, {});""".format(
                        hls_call, hls_ram_style
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
        self.code_gen_dict["$PRAGMAS$"] = ["#pragma HLS INTERFACE axis port=in0"]
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=out")
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE ap_ctrl_none port=return"
        )
