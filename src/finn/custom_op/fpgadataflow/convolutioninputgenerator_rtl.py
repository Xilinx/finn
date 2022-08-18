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
from math import copysign
import numpy as np
import os

from qonnx.core.datatype import DataType
from qonnx.custom_op.general import im2col
from qonnx.custom_op.general.im2col import compute_conv_output_dim
from finn.custom_op.fpgadataflow.hlscustomop import HLSCustomOp
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy

from finn.util.basic import (
    get_rtlsim_trace_depth,
    make_build_dir,
)

try:
    from pyverilator import PyVerilator
except ModuleNotFoundError:
    PyVerilator = None

# RTL Convolution Input Generator / Sliding Window Generator (SWG)
# Matches and extends the functionality of all ConvolutionInputGenerator_* functions
# in finn-hlslib by generating HDL code for two different implementation styles:
# - Addressable cyclic buffer: to be used when out_width <= in_width
# - Parallel registers + line buffers: to be used when out_width > in_width
# Supports non-square, 1D, strided, dilated, and depthwise convolutions.
# Note: the actual data layout produced is different for depthwise and non-depthwise ops:
# * non-depthwise SWG: (1, OFMDim_H, OFMDim_W, K_H, K_W, IFMChannels/SIMD, SIMD)
# * depthwise SWG: (1, OFMDim_H, OFMDim_W, IFMChannels/SIMD, K_H, K_W, SIMD)

class ConvolutionInputGenerator_rtl(HLSCustomOp):
    """Class that does not correspond to one of the finn-hlslib ConvolutionInputGenerator
    (sliding window) function variants! ... """

    def __init__(self, onnx_node):
        super().__init__(onnx_node)

    def get_nodeattr_types(self):
        my_attrs = {
            "ConvKernelDim": ("ints", True, []),  # [H, W] = [Y, X]
            "IFMChannels": ("i", True, 0),
            "IFMDim": ("ints", True, []),  # [H, W] = [Y, X]
            "OFMDim": ("ints", True, []),  # [H, W] = [Y, X]
            "SIMD": ("i", True, 0),
            "M": ("i", False, 1),
            "parallel_window": ("i", False, 0, {0, 1}),
            "Stride": ("ints", True, []),  # [H, W] = [Y, X]
            "Dilation": ("ints", True, []),  # [H, W] = [Y, X]
            # FINN DataTypes for inputs, weights, outputs
            "inputDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
            "depthwise": ("i", False, 0, {0, 1}),
            # FPGA resource type for ConvolutionInputGenerator input buffer
            # auto -- let Vivado decide
            # block -- use BRAM
            # distributed -- use LUTRAM
            # ultra -- use URAM
            "ram_style": (
                "s",
                False,
                "auto",
                {"auto", "block", "distributed", "ultra"},
            ),
            "gen_top_module": ("s", False, ""),
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
        M = self.get_nodeattr("M")
        assert ifm_ch % simd == 0, "SIMD must divide IFMChannels"
        wf = int(ifm_ch / simd)
        #folded_ishape = (1, ifm_dim_h, ifm_dim_w, wf, simd)
        #round up to support ifm_dim % M != 0
        if ifm_dim_w == 1:
            folded_ishape = (1, math.ceil(ifm_dim_h/M), ifm_dim_w, wf, int(simd*M))
        else:
            folded_ishape = (1, ifm_dim_h, math.ceil(ifm_dim_w/M), wf, int(simd*M))
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
        M = self.get_nodeattr("M")
        pad = 0
        ofm_dim_h = compute_conv_output_dim(ifm_dim_h, k_h, stride_h, pad, dilation_h)
        ofm_dim_w = compute_conv_output_dim(ifm_dim_w, k_w, stride_w, pad, dilation_w)
        assert ifm_ch % simd == 0, "SIMD must divide IFMChannels"
        if (self.get_nodeattr("parallel_window")):
            wf = int((ifm_ch) // simd)
            #folded_oshape = (1, ofm_dim_h, ofm_dim_w, wf, k_h * k_w * simd)
            if ofm_dim_w == 1:
                folded_oshape = (1, int(ofm_dim_h/M), ofm_dim_w, wf, k_h * k_w * int(simd*M))
            else:
                folded_oshape = (1, ofm_dim_h, int(ofm_dim_w/M), wf, k_h * k_w * int(simd*M))
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
        M = self.get_nodeattr("M")
        assert ifm_ch % simd == 0, "SIMD must divide IFMChannels"
        in_width = simd * ibits * M
        return in_width

    def get_outstream_width(self):
        if (self.get_nodeattr("parallel_window")):
            # feed all window pixels in parallel
            k_h, k_w = self.get_nodeattr("ConvKernelDim")
            return self.get_instream_width() * k_h * k_w
        else:
            # if parallel variant not in use: same width for output and input stream
            return self.get_instream_width()

    def get_number_input_values(self):
        folded_ishape = self.get_folded_input_shape()
        num_input_elems = np.prod(folded_ishape[:-1])
        return num_input_elems

    def get_number_output_values(self):
        folded_oshape = self.get_folded_output_shape()
        num_output_elems = np.prod(folded_oshape[:-1])
        return num_output_elems

    def get_exp_cycles(self):
        # TODO: update
        simd = self.get_nodeattr("SIMD")
        ifm_ch = self.get_nodeattr("IFMChannels")
        k = self.get_nodeattr("ConvKernelDim")
        ifm_dim = self.get_nodeattr("IFMDim")
        ofm_dim = self.get_nodeattr("OFMDim")
        stride = self.get_nodeattr("Stride")
        dilation = self.get_nodeattr("Dilation")
        ifm_dim_h, ifm_dim_w = ifm_dim
        ofm_dim_h, ofm_dim_w = ofm_dim
        k_h, k_w = k
        stride_h, stride_w = stride
        dilation_h, dilation_w = dilation
        mmv = 1

        if (self.get_nodeattr("parallel_window")):
            exp_cycles = ifm_dim_w + 1
        else:
            cycles_write_block = (ofm_dim_w * k_w * k_h * (ifm_ch / simd)) / mmv
            cycles_read_block = stride_w * ifm_dim_w * (ifm_ch / simd)
            max_cycles = max(cycles_write_block, cycles_read_block)
            exp_cycles = (
                ifm_dim_w * k_h * dilation_h * (ifm_ch / simd) + ofm_dim_h * max_cycles
            )

        return int(exp_cycles)

    def bram_estimation(self):
        # TODO: update
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
        # TODO: update
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
        # TODO: update
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
            #code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
            raise Exception(
                """cppsim not possible for RTL SWG""".format(
                    mode
                )
            )
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
        # disable this check to allow for IFMdim % M != 0 case (see below) where input comes from MMV-output capable node
        #assert (
        #    inp.shape == exp_ishape
        #), """Input shape doesn't
        #match expected shape (1, ifm_dim, ifm_dim, ifm_ch)."""
        if self.get_input_datatype() == DataType["BIPOLAR"]:
            # store bipolar activations as binary
            inp = (inp + 1) / 2
            export_idt = DataType["BINARY"]
        else:
            export_idt = self.get_input_datatype()

        # pad test input stream to work when IFMdim % M != 0
        # during normal operation, the AXI Stream should not care, in the last cycle garbage elements are read but not used
        # TODO: only works for 1D case
        mmv_stream_padding_px = int((np.prod(folded_ishape) - np.prod(inp.shape)) / exp_ishape[-1])
        if exp_ishape [2] == 1:
            inp = np.pad(inp, ((0,0),(0,mmv_stream_padding_px),(0,0),(0,0)), 'constant')
        else:
            inp = np.pad(inp, ((0,0),(0,0),(0,mmv_stream_padding_px),(0,0)), 'constant')
        # reshape input into folded form
        inp = inp.reshape(folded_ishape)
        # make copy before saving array
        reshaped_input = inp.copy()
        np.save(os.path.join(code_gen_dir, "input_0.npy"), reshaped_input)

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
        pass

    def defines(self, var):
        pass

    def read_npy_data(self):
        pass

    def strm_decl(self):
        pass

    def docompute(self):
        pass

    def dataoutstrm(self):
        pass

    def save_as_npy(self):
        pass

    def blackboxfunction(self):
        pass

    def pragmas(self):
        pass
        
    def generate_hdl(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        #f_debug = open(os.path.join(code_gen_dir, "swg_hdl_debuginfo.log"), "w")
        code_gen_dict = {}

        ##### BEGIN INITIALIZE/CHECK CONFIGURATION #####
        ifm_ch = self.get_nodeattr("IFMChannels")
        k = self.get_nodeattr("ConvKernelDim")
        ifm_dim = self.get_nodeattr("IFMDim")
        ofm_dim = self.get_nodeattr("OFMDim")
        stride = self.get_nodeattr("Stride")
        dilation = self.get_nodeattr("Dilation")
        depthwise = self.get_nodeattr("depthwise")

        n = 1
        h, w = ifm_dim
        c = 1 # assume SIMD=C (parallelize across all channels)
        k_h, k_w = k
        pad = [0,0,0,0] # padding happens in separate padding node for now
        pad_val = 0
        stride_h, stride_w = stride
        dilation_h, dilation_w = dilation

        in_shape = (n,c,h,w) #NCHW

        in_image = np.empty(in_shape, dtype=int)
        in_image_padded = np.pad(
            in_image,
            ((0, 0), (0, 0), (pad[0], pad[2]), (pad[1], pad[3])),
            mode="constant",
            constant_values=pad_val,
        )
        in_shape_padded = in_image_padded.shape
        h_padded = in_shape_padded[2]
        w_padded = in_shape_padded[3]

        pad_h = pad[0] + pad[2]
        pad_w = pad[1] + pad[3]
        out_dim_h = im2col.compute_conv_output_dim(h, k_h, stride_h, pad_h, dilation_h)
        out_dim_w = im2col.compute_conv_output_dim(w, k_w, stride_w, pad_w, dilation_w)

        # init folding config
        simd = self.get_nodeattr("SIMD")
        M = self.get_nodeattr("M")
        if (self.get_nodeattr("parallel_window")):
            mmv_in = M*1
            mmv_out = M*k_h*k_w
            assert ifm_ch==simd, "Constraint violated: SIMD must be equal to C"
        else:
            mmv_in = 1
            mmv_out = 1
            assert ifm_ch%simd==0, "Constraint violated: SIMD must divide C"

        # TODO: check allowed hyperparams
        # for 1D case: it does not matter if dummy dim is x or y
        # TODO: move/duplicate these checks in corresponding convert_to_hls transformation (?)

        # choose implementation style
        if (mmv_out > 1 or (k_h==1 and k_w==1)):
            impl_style = "parallel"
        else:
            impl_style = "default"

        ##### END INITIALIZE/CHECK CONFIGURATION #####

        ##### BEGIN CODE GEN FOR DEFAULT STYLE #####
        if (impl_style == "default"):
            # Default implementation style for MMV_out = 1: addressable cyclic buffer
            # Computing incremental addressing scheme directly..

            # compute index/address increments for each nested loop
            channel_factor = int(ifm_ch/simd)

            # compute minimal buffer length (assuming it holds 1 complete window)
            buffer_min_size = ((k_h-1) * dilation_h * w + (k_w-1) * dilation_w + 1) * channel_factor

            kernel_width = (k_w-1)*dilation_w+1 # incl. dilation
            addr_incr_end_simd = 1
            addr_incr_end_window_elem = (dilation_w-1) * channel_factor + 1
            
            remaining_line = (w - kernel_width) * channel_factor
            skip_lines = (dilation_h-1) * w * channel_factor
            addr_incr_end_window_row = remaining_line + skip_lines + 1 # 1 = wrap around of minimally sized buffer
            
            addr_incr_end_window = -buffer_min_size + stride_w * channel_factor + 1 # 1 = wrap around of minimally sized buffer

            # rows that are skipped due to imperfect stride<->W combination
            skip_columns = w%(kernel_width + (out_dim_w-1)*stride_w)
            remaining_line = (skip_columns + kernel_width) * channel_factor # increment from oldest buffer position (top left) to end of line
            skip_lines = (stride_h-1) * w * channel_factor
            addr_incr_end_row = -buffer_min_size + remaining_line + skip_lines + 1 # 1 = wrap around of minimally sized buffer

            if (depthwise):
                addr_incr_end_window_elem = dilation_w * channel_factor
                addr_incr_end_window_row = (channel_factor 
                                            + (w - kernel_width) * channel_factor
                                            + (dilation_h-1) * w * channel_factor
                                           )
                addr_incr_end_simd = -buffer_min_size + (channel_factor + 1)

            # add additional buffer space in case of stride > 1
            # this minimizes cycle count, as it allows an earlier pre-load of skipped input elements
            buffer_actual_size = (buffer_min_size + max(0,((stride_w-1)   - (int(mmv_out*k_h*k_w/mmv_in)))*channel_factor)
                                                  + max(0,((stride_h-1)*w - (int(mmv_out*k_h*k_w/mmv_in)))*channel_factor))
            code_gen_dict["$BUF_ELEM_TOTAL$"] = [str(buffer_actual_size)]

            assert not(abs(addr_incr_end_window) > buffer_actual_size), "ERROR: W increment > buffer size, wrap logic doesn't account for this"
            assert not(abs(addr_incr_end_row) > buffer_actual_size), "ERROR: H increment > buffer size, wrap logic doesn't account for this"

            kernel_width = (k_w-1)*dilation_w+1 # incl. dilation
            kernel_height = (k_h-1)*dilation_h+1 # incl. dilation
            skip_columns = w%(kernel_width + (out_dim_w-1)*stride_w)
            skip_rows = h%(kernel_height + (out_dim_h-1)*stride_h)
            code_gen_dict["$LAST_READ_ELEM$"] = [str(h*w*channel_factor-1)]
            code_gen_dict["$LAST_WRITE_ELEM$"] = [str(((h - skip_rows - 1) * w + (w - skip_columns))*channel_factor -1)]

            loop_h_iterations = out_dim_h
            loop_w_iterations = out_dim_w
            loop_kh_iterations = k_h
            loop_kw_iterations = k_w
            loop_simd_iterations = channel_factor

            if (depthwise and channel_factor > 1):
                # re-arrange existing controller loop structure for depthwise convolutions
                loop_kh_iterations = channel_factor
                loop_kw_iterations = k_h
                loop_simd_iterations = k_w
                addr_incr_end_simd_ = addr_incr_end_simd
                addr_incr_end_simd = addr_incr_end_window_elem
                addr_incr_end_window_elem = addr_incr_end_window_row
                addr_incr_end_window_row = addr_incr_end_simd_
                elem_per_window = k_h*k_w         

                tail_incr_w = addr_incr_end_window + buffer_min_size - channel_factor     
                tail_incr_h = addr_incr_end_row + buffer_min_size - channel_factor
                tail_incr_last_window = buffer_min_size-1                                                
                code_gen_dict["$TAIL_INCR_GENERATION$"] = ["""
                always @ (counter_loop_kh, counter_loop_w, counter_loop_h) begin
                         if (counter_loop_kh >= 0)
                             tail_incr_reg = 1;
                         else if (counter_loop_w >= 0)
                             tail_incr_reg = {};
                         else if (counter_loop_h >= 0)
                             tail_incr_reg = {};
                         else
                             tail_incr_reg = {};
                end
                """.format(tail_incr_w, tail_incr_h, tail_incr_last_window)]
            else:
                # depthwise output format is equivalent to non-depthwise if SIMD=C
                elem_per_window = k_h*k_w*channel_factor

                tail_incr_w = addr_incr_end_window + buffer_min_size - 1     
                tail_incr_h = addr_incr_end_row + buffer_min_size - 1
                tail_incr_last_window = buffer_min_size-1
                code_gen_dict["$TAIL_INCR_GENERATION$"] = ["""
                always @ (counter_loop_w, counter_loop_h) begin
                        if (counter_loop_w >= 0)
                            tail_incr_reg = {};
                        else if (counter_loop_h >= 0)
                            tail_incr_reg = {};
                        else
                            tail_incr_reg = {};
                end
                """.format(tail_incr_w, tail_incr_h, tail_incr_last_window)]

            # support SIMD = C and k_w = 1 cases
            # for k = [k_h, k_w] = [1, k_w], no adjustment is needed
            # for k = [k_h, k_w] = [1, 1], do not use this impl. style (mmv_out=K=1)
            # innermost loop is executed at least once -> adjust if needed
            if (loop_simd_iterations == 1):
                # skip innermost SIMD loop completely
                if (loop_kw_iterations == 1):
                    # skip innermost KW loop completely
                    code_gen_dict["$INNERMOST_STATE$"]=["STATE_LOOP_KH"]
                    loop_kh_iterations -= 1  # -1 because state is initial state
                else:
                    code_gen_dict["$INNERMOST_STATE$"]=["STATE_LOOP_KW"]
                    loop_kw_iterations -= 1 # -1 because state is initial state
            else:
                code_gen_dict["$INNERMOST_STATE$"]=["STATE_LOOP_SIMD"]
                loop_simd_iterations -= 1 # -1 because state is initial state
            
            code_gen_dict["$LOOP_H_ITERATIONS$"]=[str(loop_h_iterations-1)]
            code_gen_dict["$LOOP_W_ITERATIONS$"]=[str(loop_w_iterations-1)]
            code_gen_dict["$LOOP_KH_ITERATIONS$"]=[str(loop_kh_iterations-1)]
            code_gen_dict["$LOOP_KW_ITERATIONS$"]=[str(loop_kw_iterations-1)]
            code_gen_dict["$LOOP_SIMD_ITERATIONS$"]=[str(loop_simd_iterations-1)]

            incr_bitwidth = 1 + math.ceil(math.log2(max(abs(addr_incr_end_simd)+1, 
                                                        abs(addr_incr_end_window_elem)+1, 
                                                        abs(addr_incr_end_window_row)+1, 
                                                        abs(addr_incr_end_window)+1, 
                                                        abs(addr_incr_end_row)+1, 
                                                        abs(tail_incr_w)+1, 
                                                        abs(tail_incr_h)+1,
                                                        abs(tail_incr_last_window)+1)))
            code_gen_dict["$INCR_BITWIDTH$"] = [str(incr_bitwidth)]
            code_gen_dict["$ADDR_INCREMENT_MAP$"]=["'{{ {}'d0, {}'d{}, {}'d{}, {}'d{}, {}'d{}, {}'d{}}}".format(incr_bitwidth, 
                                                int(copysign(incr_bitwidth,addr_incr_end_simd)),abs(addr_incr_end_simd),
                                                int(copysign(incr_bitwidth,addr_incr_end_window_elem)),abs(addr_incr_end_window_elem),
                                                int(copysign(incr_bitwidth,addr_incr_end_window_row)),abs(addr_incr_end_window_row),
                                                int(copysign(incr_bitwidth,addr_incr_end_window)),abs(addr_incr_end_window),
                                                int(copysign(incr_bitwidth,addr_incr_end_row)),abs(addr_incr_end_row))]

            code_gen_dict["$ELEM_PER_WINDOW$"] = [str(elem_per_window)]

            with open(os.environ['FINN_ROOT']+"/finn-rtllib/swg/swg_template_default.sv", "r") as f:
                template = f.read()
       
        ##### END CODE GEN FOR DEFAULT STYLE #####
    
        ##### BEGIN CODE GEN FOR PARALLEL STYLE #####
        elif (impl_style == "parallel"):
            # Out width > In width: Parallel implementation style using registers + line buffers
            idx_c, idx_h, idx_w = im2col.get_im2col_indices_nchw(
            in_shape,
            k_h,
            k_w,
            pad,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w
            )

            cols = in_image_padded[:, idx_c, idx_h, idx_w]
            cols = cols.transpose(1, 2, 0).reshape(k_h * k_w * c, -1)

            # result shape is (k_H*k_W*N, out_dim_H*out_dim_W), convert to NCHW
            out_image = cols.reshape(n, c, k_h, k_w, out_dim_h, out_dim_w)
            # (N=0,C=1,kh=2,kw=3,H=4,W=5) -> (N=0,H=4,W=5,kh=2,kw=3,C=1)
            out_image = out_image.transpose(0, 4, 5, 2, 3, 1)
            out_image = out_image.reshape(n, out_dim_h, out_dim_w, k_h * k_w * c)

            idx_px = idx_h*w+idx_w # sequential pixel indices
    
            k, cycles = idx_px.shape

            output_elements = mmv_out
            output_cycles = int(cycles/(mmv_out/k))

            # TODO: what happens when output_cycles=OFMdim % M != 0
            # ...try to support IFMdim % M != 0 first, so we can work with the usual k=3 where OFMdim = IFMdim - -2
            # the additional garbage input elements that are read in the last cycle are not read by any window anyway
            idx_px = idx_px.transpose()
            idx_px = idx_px.reshape(output_cycles, output_elements)
            idx_px = idx_px.transpose()
            # result: first dim is number of parallel output elements, 
            # second dim is the input element (pixel in case of SIMD=C) index that each output element outputs per cycle

            buffer = []
            buffer_max_size = 0
            schedule = []
            next_in_px = 0
            oldest_px = 0

            def schedule_append(schedule, op):
                if len(schedule) > 0 and schedule[-1][1] == op:
                    count, op_ = schedule[-1]
                    schedule[-1] = (count+1, op_)
                else:
                    schedule.append((1, op))
                return schedule
            
            # compute schedule and buffer read pattern (output driven)
            idx_px_relative = idx_px.copy()
            output_elem, output_cycles = idx_px_relative.shape
            
            for x in range(output_cycles):
                # load missing inputs into buffer
                for y in range(output_elem):
                    while int(idx_px_relative[y,x]) not in buffer:
                        # load M inputs at once (keep "buffer" list 1D for now, handle actual 2D buffer generation later)
                        for m in range(M):
                            buffer.append(next_in_px)
                            next_in_px += 1
                        schedule = schedule_append(schedule,'w')
                
                # discard unused buffer elements
                oldest_px = np.min(idx_px_relative[:,x:])
                #check whether M elements can be shifted out, not just the single oldest one
                #while all([buffer[i] < oldest_px for i in range(M)]):
                if all([buffer[i] < oldest_px for i in range(M)]):
                    # M buffer elements are shifted out at once
                    for m in range(M):
                        buffer.pop(0)
        
                # adjust relative buffer index of current x (according to last discarded buffer elements)
                for y in range(output_elem):
                    idx_px_relative[y,x] -= oldest_px
                
                # read from buffer    
                # + simultaneously load next pixel(s) into buffer if there are any left
                if (next_in_px > (h_padded*w_padded-1)):
                    # read only (append above)
                    schedule = schedule_append(schedule,'r')
                else:
                    # load M inputs at once
                    for m in range(M):
                        buffer.append(next_in_px)
                        next_in_px += 1
                    schedule = schedule_append(schedule,'wr')

                # record max needed buffer depth
                if len(buffer) > buffer_max_size:
                    buffer_max_size = len(buffer)

            # insert dummy write operations for data at the input FM tail-end that is never read (e.g. in case of stride > 1)
            while next_in_px <= (h_padded*w_padded-1):
                next_in_px += 1
                schedule = schedule_append(schedule,'w')

            # find buffer access patterns
            buffer_access_patterns = []
            for x in range(output_cycles):
                if idx_px_relative[:,x].tolist() not in buffer_access_patterns:
                    buffer_access_patterns.append(idx_px_relative[:,x].tolist())

            # Experimental implementation to map fixed controller loop structure to R/W schedule by analyzing
            # the access pattern given by Im2Col, rather than direct computation.
            # TODO: Probably replace this with a directly-computed schedule, similar to the default implementation style.
            def compact_schedule(schedule):
                # leave first sequence (pre-load) as is
                start_sequence = schedule[0]
                loop_sequence_1_counter = 1
                loop_sequence_1 = schedule[1]
                loop_counter = 0
                loop_sequence_2 = None
                end_sequence = None

                i = 2
                if i < len(schedule):
                    loop_sequence_1 += schedule[i]
                    i += 1
                while i+1 < len(schedule):
                    candidate = schedule[i] + schedule[i+1]
                    if candidate == loop_sequence_1:
                        loop_sequence_1_counter += 1
                        i += 2
                    else:
                        break

                if i < len(schedule):
                    loop_sequence_2 = schedule[i]
                    i += 1
                if i+1 < len(schedule):
                    candidate = schedule[i] + schedule[i+1]
                    if candidate != loop_sequence_1:
                        loop_sequence_2 += schedule[i]
                    i -= 1
                    loop_sequence_total_len = (int(len(loop_sequence_2)/2)) + loop_sequence_1_counter*(int(len(loop_sequence_1)/2))
                    loop_sequence_total = loop_sequence_2 + loop_sequence_1_counter*loop_sequence_1
                    while i+loop_sequence_total_len < len(schedule):
                        candidate = schedule[i] 
                        for x in range (i+1, i+loop_sequence_total_len):
                            candidate += schedule[x]

                        if candidate == loop_sequence_total:
                            loop_counter += 1
                            i += loop_sequence_total_len
                        else:
                            break
                else:
                    if i < len(schedule):
                        end_sequence = loop_sequence_2 + schedule[i]
                        i += 1
                        loop_sequence_2 = None
                    else:
                        end_sequence = loop_sequence_2
                        loop_sequence_2 = None

                if i < len(schedule):
                    end_sequence = schedule[i]
                    i += 1
                if i < len(schedule):
                    end_sequence = end_sequence + schedule[i]
                    i += 1

                assert len(start_sequence) == 1*2, "ERROR: invalid start sequence"
                assert len(loop_sequence_1) == 2*2, "ERROR: invalid loop 1 sequence"
                if loop_sequence_2:
                    assert len(loop_sequence_2) <= 2*2, "ERROR: invalid loop 2 sequence"
                if end_sequence:
                    assert len(end_sequence) <= 2*2, "ERROR: invalid end sequence"
                assert i == len(schedule), "ERROR: schedule could not be compacted %d / %d" %(i, len(schedule))

                return (start_sequence, loop_counter, loop_sequence_1_counter,
                        loop_sequence_1, loop_sequence_2, end_sequence)

            ### determine buffer partitioning into REG FIFOs (parallel access) and BRAM FIFOs (line buffers)
            # TODO: this part doesn't fully account for M for 2D buffers yet

            # how many "unused" registers are allowed between buffer positions that will be accessed in parallel
            # example:
            # 0: only consecutive access patterns will be implemented in regs, rest in (LUTRAM/BRAM) line buffers
            # 2: [0, 3, 6] access pattern is still allowed and will be implemented with one 7-position shift reg
            REG_BRAM_THRESHOLD = 8

            code_gen_dict["$BUF_ELEM_TOTAL$"] = [str(buffer_max_size)]

            assert len(buffer_access_patterns) == 1, "ERROR: Buffer access pattern is not static"
            buf_static_access_pattern = buffer_access_patterns[0]
            reg_fifos = []
            reg_fifos_depth = []
            bram_fifos = []
            bram_fifos_depth = []
            current = []
            for i in range(len(buf_static_access_pattern)):
                access_idx = buf_static_access_pattern[i]
                if len(current) == 0:
                    current.append(access_idx)
                else:
                    # assume non-decreasing index order in access pattern
                    # TODO: this assumption does not hold for M>1 case (2D buffer)
                    distance = access_idx - max(current)
                    if not (distance-1 > REG_BRAM_THRESHOLD):
                        for i in range(distance-1):
                            # insert dummy into REG FIFO (not read as part of window)
                            current.append(-1)
                        # assign this access to same REG FIFO as previous one
                        current.append(access_idx)
                    else:
                        # assign skipped accesses to new BRAM FIFO
                        bram_fifos.append([-1]*(distance-1))
                        bram_fifos_depth.append(math.ceil((distance-1)/M)) # really ceil?
                        # start with new REG FIFO
                        reg_fifos.append(current)
                        #reg_fifos_depth.append(math.ceil((max(current)+1)/M)) # fix for M again
                        reg_fifos_depth.append(len(current))
                        current = []
                        current.append(access_idx)
            reg_fifos.append(current)
            #reg_fifos_depth.append(math.ceil((max(current)+1)/M)) # fix for M again
            reg_fifos_depth.append(len(current))

            code_gen_dict["$GENERATE_REG_FIFOS$"] = []
            for i in range(len(reg_fifos)):
                code_gen_dict["$GENERATE_REG_FIFOS$"].append(
                    """
                    wire [IN_WIDTH-1:0] reg_fifo_{id}_in;
                    wire [IN_WIDTH-1:0] reg_fifo_{id}_out;
                    wire [IN_WIDTH*{len}-1:0] reg_fifo_{id};
                    {name}_reg_buffer
                    #(
                    .WIDTH(IN_WIDTH),
                    .DEPTH({len})
                    )
                    reg_buffer_inst_{id}
                    (
                        .CLK(CLK),
                        .shift_enable(shift_enable),
                        .shift_in(reg_fifo_{id}_in),
                        .shift_out(reg_fifo_{id}_out),
                        .data_out(reg_fifo_{id})
                    );""".format(name=self.get_verilog_top_module_name(), id=i, len=reg_fifos_depth[i]))

            code_gen_dict["$GENERATE_BRAM_FIFOS$"] = []
            for i in range(len(bram_fifos)):
                code_gen_dict["$GENERATE_BRAM_FIFOS$"].append(
                    """
                    wire [IN_WIDTH-1:0] bram_fifo_{id}_in;
                    wire [IN_WIDTH-1:0] bram_fifo_{id}_out;
                    {name}_ram_buffer
                    #(
                    .WIDTH(IN_WIDTH),
                    .DEPTH({len})
                    )
                    ram_buffer_inst_{id}
                    (
                        .CLK(CLK),
                        .RST(RST),
                        .shift_enable(shift_enable),
                        .shift_in(bram_fifo_{id}_in),
                        .shift_out(bram_fifo_{id}_out)
                    );""".format(name=self.get_verilog_top_module_name(), id=i, len=bram_fifos_depth[i]))

            code_gen_dict["$GENERATE_OUTPUT_MAPPING$"] = []
            out_idx = mmv_out-1
            for fifo_id, reg_fifo in enumerate(reg_fifos):
                for fifo_idx, access_idx in enumerate(reg_fifo):
                    if(access_idx != -1):
                        #code_gen_dict["$GENERATE_OUTPUT_MAPPING$"].append(
                        #    "assign data_out[OUT_ELEM_WIDTH*{out_idx}+:OUT_ELEM_WIDTH] = reg_fifo_{fifo_id}[{fifo_idx}]; //{access_idx}".format(
                        #        out_idx=out_idx, fifo_id=fifo_id, fifo_idx=fifo_idx, access_idx=access_idx
                        #    )
                        #)
                        code_gen_dict["$GENERATE_OUTPUT_MAPPING$"].append(
                            "assign data_out[OUT_ELEM_WIDTH*{out_idx}+:OUT_ELEM_WIDTH] = reg_fifo_{fifo_id}[{access_idx}*{mmv}*OUT_ELEM_WIDTH+OUT_ELEM_WIDTH*{mmv_idx}+:OUT_ELEM_WIDTH];".format(
                                out_idx=out_idx, fifo_id=fifo_id, 
                                access_idx=reg_fifos_depth[fifo_id]-1-int((max(reg_fifo)-access_idx)/M), 
                                mmv_idx=(max(reg_fifo)-access_idx)%M,
                                mmv = M
                            )
                        )
                        # reversal: out_idx=0 -> oldest buffer element -> highest access_idx
                        out_idx = out_idx-1
            assert out_idx==-1, "ERROR: Not all output vector elements connected"

            code_gen_dict["$GENERATE_BUFFER_CONNECTION$"] = []
            for i in range(len(reg_fifos)):
                if i == 0:
                    # first FIFO containing newest elements -> input comes from input reg
                    code_gen_dict["$GENERATE_BUFFER_CONNECTION$"].append(
                        """assign reg_fifo_{fifo_id}_in = reg_input;""".format(fifo_id=i,))
                else:
                    # other REG FIFOs -> input comes from connected BRAM FIFO (line buffer)
                    input_fifo_id = i-1
                    code_gen_dict["$GENERATE_BUFFER_CONNECTION$"].append(
                        """assign reg_fifo_{fifo_id}_in = bram_fifo_{input_fifo_id}_out;""".format(fifo_id=i, input_fifo_id=input_fifo_id))
            for i in range(len(bram_fifos)):
                input_fifo_id = i
                code_gen_dict["$GENERATE_BUFFER_CONNECTION$"].append(
                    """assign bram_fifo_{fifo_id}_in = reg_fifo_{input_fifo_id}_out;""".format(fifo_id=i, input_fifo_id=input_fifo_id))

            def convert_tuple(seq):
                mapping = {'w': ("1'b1", "1'b0"),
                            'r': ("1'b0", "1'b1"),
                            'wr':("1'b1", "1'b1"),
                            'n': ("1'b0", "1'b0")}
                if seq:
                    if len(seq) == 2:
                        return (seq[0], mapping[seq[1]], 0, mapping['n'])
                    if len(seq) == 4:
                        return (seq[0], mapping[seq[1]], seq[2], mapping[seq[3]])
                else:
                    return (0, mapping['n'], 0, mapping['n'])

            start_sequence,loop_counter,loop_sequence_1_counter,loop_sequence_1,loop_sequence_2,end_sequence = compact_schedule(schedule)

            start_sequence = convert_tuple(start_sequence)
            loop_sequence_1 = convert_tuple(loop_sequence_1)
            loop_sequence_2 = convert_tuple(loop_sequence_2)
            end_sequence = convert_tuple(end_sequence)

            cycles_total = 0
            for t in schedule:
                cycles_total += t[0]
            code_gen_dict["$CYCLES_TOTAL$"] = [str(cycles_total)]

            code_gen_dict["$START_COUNTER$"]=[str(start_sequence[0])]
            code_gen_dict["$LOOP_MAIN_COUNTER$"]=[str(loop_sequence_1_counter)]
            code_gen_dict["$LOOP_INTER_COUNTER$"]=[str(loop_counter)]

            code_gen_dict["$LOOP_MAIN_1_COUNTER$"]=[str(loop_sequence_1[0])]
            code_gen_dict["$LOOP_MAIN_2_COUNTER$"]=[str(loop_sequence_1[2])]

            code_gen_dict["$LOOP_INTER_1_COUNTER$"]=[str(loop_sequence_2[0])]
            code_gen_dict["$LOOP_INTER_2_COUNTER$"]=[str(loop_sequence_2[2])]

            code_gen_dict["$LOOP_END_1_COUNTER$"]=[str(end_sequence[0])]
            code_gen_dict["$LOOP_END_2_COUNTER$"]=[str(end_sequence[2])]

            code_gen_dict["$READ_CMD_MAP$"]=["{{ {}, {}, {}, {}, {}, {}, {} }}".format(
                start_sequence[1][0],loop_sequence_1[1][0],loop_sequence_1[3][0],loop_sequence_2[1][0],loop_sequence_2[3][0],end_sequence[1][0],end_sequence[3][0])]
            code_gen_dict["$WRITE_CMD_MAP$"]=["{{ {}, {}, {}, {}, {}, {}, {} }}".format(
                start_sequence[1][1],loop_sequence_1[1][1],loop_sequence_1[3][1],loop_sequence_2[1][1],loop_sequence_2[3][1],end_sequence[1][1],end_sequence[3][1])]

            with open(os.environ['FINN_ROOT']+"/finn-rtllib/swg/swg_template_parallel.sv", "r") as f:
                template = f.read()

        ##### END CODE GEN FOR PARALLEL STYLE #####

        ##### BEGIN GENERAL CODE GEN #####
        code_gen_dict["$TOP_MODULE_NAME$"] = [self.get_verilog_top_module_name()]
        # save top module name so we can refer to it even after this node has been renamed 
        # (e.g. by GiveUniqueNodeNames(prefix) during MakeZynqProject)
        self.set_nodeattr("gen_top_module", self.get_verilog_top_module_name())
        code_gen_dict["$BIT_WIDTH$"] = [str(self.get_input_datatype().bitwidth())]
        code_gen_dict["$SIMD$"] = [str(simd)]
        code_gen_dict["$MMV_IN$"] = [str(mmv_in)]
        code_gen_dict["$MMV_OUT$"] = [str(mmv_out)]
        
        ram_style = self.get_nodeattr("ram_style")
        if ram_style == "auto":
            code_gen_dict["$RAM_STYLE$"]=[""]
        else:
            code_gen_dict["$RAM_STYLE$"]=["(* ram_style = \"{}\" *)".format(ram_style)]

        with open(os.environ['FINN_ROOT']+"/finn-rtllib/swg/swg_template_wrapper.v", "r") as f:
            template_wrapper = f.read()

        for key in code_gen_dict:
            # transform list into long string separated by '\n'
            code_gen_line = "\n".join(code_gen_dict[key])
            template = template.replace(key, code_gen_line)
            template_wrapper = template_wrapper.replace(key, code_gen_line)

        f = open(os.path.join(code_gen_dir, self.get_nodeattr("gen_top_module") + "_impl.sv"), "w")
        f.write(template)
        f.close()
        f = open(os.path.join(code_gen_dir, self.get_nodeattr("gen_top_module") + "_wrapper.v"), "w")
        f.write(template_wrapper)
        f.close()
        #f_debug.close()

        #set ipgen_path and ip_path so that HLS-Synth transformation and stich_ip transformation do not complain
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)
        ##### END GENERAL CODE GEN #####

    def prepare_rtlsim(self):
        """Creates a Verilator emulation library for the RTL code generated
        for this node, sets the rtlsim_so attribute to its path and returns
        a PyVerilator wrapper around it."""
        # Modified to use generated (System-)Verilog instead of HLS output products

        if PyVerilator is None:
            raise ImportError("Installation of PyVerilator is required.")

        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        verilog_paths = [code_gen_dir]    
        verilog_files = [self.get_nodeattr("gen_top_module") + "_wrapper.v",
                         self.get_nodeattr("gen_top_module") + "_impl.sv"]

        # build the Verilator emu library
        sim = PyVerilator.build(
            verilog_files,
            build_dir=make_build_dir("pyverilator_" + self.onnx_node.name + "_"),
            verilog_path=verilog_paths,
            trace_depth=get_rtlsim_trace_depth(),
            top_module_name=self.get_verilog_top_module_name(),
        )
        # save generated lib filename in attribute
        self.set_nodeattr("rtlsim_so", sim.lib._name)
        return sim


    def code_generation_ipi(self):
        """Constructs and returns the TCL for node instantiation in Vivado IPI."""
        vlnv = self.get_nodeattr("ip_vlnv")
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

        cmd = ["add_files -norecurse %s" % (os.path.join(code_gen_dir, self.get_nodeattr("gen_top_module") + "_wrapper.v")),
            "add_files -norecurse %s" % (os.path.join(code_gen_dir, self.get_nodeattr("gen_top_module") + "_impl.sv")),
            "create_bd_cell -type module -reference %s %s" % (self.get_nodeattr("gen_top_module"), self.onnx_node.name)]

        return cmd

    def code_generation_ipgen(self, model, fpgapart, clk):
        """Normally: Generates c++ code and tcl script for ip generation.
           Here: Generates (System-)Verilog code for ip generation."""
        self.generate_hdl()

    def ipgen_singlenode_code(self):
        """Normally: Builds the bash script for ip generation using the CallHLS from
        finn.util.hls."""
        pass

    def code_generation_cppsim(self, model):
        """Normally: Generates c++ code for simulation (cppsim)."""
        pass

    def compile_singlenode_code(self):
        pass
