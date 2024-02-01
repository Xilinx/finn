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
import shutil
from qonnx.core.datatype import DataType
from qonnx.custom_op.general import im2col
from qonnx.custom_op.general.im2col import compute_conv_output_dim

from finn.custom_op.fpgadataflow.convolutioninputgenerator import (
    ConvolutionInputGenerator,
)
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.util.basic import get_rtlsim_trace_depth, make_build_dir
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy

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
# Note: the actual data layout produced is different for depthwise and non-depthwise:
# * non-depthwise SWG: (1, OFMDim_H, OFMDim_W, K_H, K_W, IFMChannels/SIMD, SIMD)
# * depthwise SWG: (1, OFMDim_H, OFMDim_W, IFMChannels/SIMD, K_H, K_W, SIMD)

# NOTE: "Parallel" implementation style not yet implemented in this version!


class ConvolutionInputGenerator_rtl(ConvolutionInputGenerator, RTLBackend):
    """Class that corresponds to finn-rtllib swg module.
    Generates an RTL ConvolutionInputGenerator implementation
    based on (System-)Verilog templates, defined in finn-rtllib/swg."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            # additional parallelization parameter - not yet implemented
            "M": ("i", False, 1),
        }
        my_attrs.update(ConvolutionInputGenerator.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    def get_number_input_values(self):
        """Function to get the number of expected input values."""
        folded_ishape = self.get_folded_input_shape()
        num_input_elems = np.prod(folded_ishape[:-1])
        return num_input_elems

    def use_parallel_window_output(self):
        return self.get_nodeattr("parallel_window")

    def get_buffer_depth(self):
        """Returns total depth of the internal buffer, depending on
        implementation style."""
        ifm_ch = self.get_nodeattr("IFMChannels")
        k = self.get_nodeattr("ConvKernelDim")
        ifm_dim = self.get_nodeattr("IFMDim")
        stride = self.get_nodeattr("Stride")
        dilation = self.get_nodeattr("Dilation")
        simd = self.get_nodeattr("SIMD")

        k_h, k_w = k
        h, w = ifm_dim
        stride_h, stride_w = stride
        dilation_h, dilation_w = dilation
        mmv_in = 1
        mmv_out = 1
        channel_factor = int(ifm_ch / simd)
        impl_style = self.select_impl_style()
        if impl_style == "default":
            buffer_min_size = (
                (k_h - 1) * dilation_h * w + (k_w - 1) * dilation_w + 1
            ) * channel_factor
            # add additional buffer space in case of stride > 1
            # this minimizes cycle count as it allows an earlier pre-load of inputs
            buffer_depth = (
                buffer_min_size
                + max(
                    0,
                    ((stride_w - 1) - (int(mmv_out * k_h * k_w / mmv_in))) * channel_factor,
                )
                + max(
                    0,
                    ((stride_h - 1) * w - (int(mmv_out * k_h * k_w / mmv_in))) * channel_factor,
                )
            )
        elif impl_style == "parallel":
            buffer_min_size = (
                (k_h - 1) * dilation_h * w + (k_w - 1) * dilation_w
            ) * channel_factor + 1
            buffer_depth = buffer_min_size + 1
        return buffer_depth

    def get_exp_cycles(self):
        impl_style = self.select_impl_style()

        if impl_style == "parallel":
            exp_cycles = self.get_number_input_values() + 2
        elif impl_style == "default":
            simd = self.get_nodeattr("SIMD")
            ifm_ch = self.get_nodeattr("IFMChannels")
            k = self.get_nodeattr("ConvKernelDim")
            ifm_dim = self.get_nodeattr("IFMDim")
            ofm_dim = self.get_nodeattr("OFMDim")
            stride = self.get_nodeattr("Stride")
            dilation = self.get_nodeattr("Dilation")
            depthwise = self.get_nodeattr("depthwise")
            ifm_dim_h, ifm_dim_w = ifm_dim
            ofm_dim_h, ofm_dim_w = ofm_dim
            k_h, k_w = k
            stride_h, stride_w = stride
            dilation_h, dilation_w = dilation

            channel_factor = int(ifm_ch / simd)
            if ifm_dim_h == 1 or ifm_dim_w == 1:
                # 1D case
                (
                    ifm_ch,
                    [ifm_dim_h, ifm_dim_w],
                    [ofm_dim_h, ofm_dim_w],
                    [k_h, k_w],
                    [stride_h, stride_w],
                    [dilation_h, dilation_w],
                ) = self.get_1d_conv_attrs_normalized()

                if depthwise:
                    exp_cycles = (
                        +ofm_dim_w * k_w * channel_factor
                        + channel_factor * (k_w - 1) * (stride_w - 1)
                        - (k_w - 1)
                        + 2
                    )
                else:
                    exp_cycles = ofm_dim_w * k_w * channel_factor + 2
            else:
                # 2D case
                buffer_min_size = (
                    (k_h - 1) * dilation_h * ifm_dim_w + (k_w - 1) * dilation_w + 1
                ) * channel_factor
                cycles_write_block = ofm_dim_w * k_w * k_h * channel_factor
                cycles_read_block = stride_w * ifm_dim_w * channel_factor
                max_cycles = max(cycles_write_block, cycles_read_block)
                if depthwise:
                    max_cycles += ofm_dim_w * (stride_w - 1) * (channel_factor - 1)
                exp_cycles = buffer_min_size + ofm_dim_h * max_cycles
                if depthwise:
                    exp_cycles += (stride_h - 1) * ifm_dim_w * channel_factor

        return int(exp_cycles)

    def bram_estimation(self):
        simd = self.get_nodeattr("SIMD")
        ram_style = self.get_nodeattr("ram_style")
        impl_style = self.select_impl_style()
        [k_h, k_w] = self.get_nodeattr("ConvKernelDim")
        [ifm_dim_h, ifm_dim_w] = self.get_nodeattr("IFMDim")
        [dilation_h, dilation_w] = self.get_nodeattr("Dilation")

        if ram_style == "block" or ram_style == "auto":
            buffer_width = simd * self.get_input_datatype().bitwidth()
            if impl_style == "default":
                buffer_depth = self.get_buffer_depth()
                buffer_count = 1
            elif impl_style == "parallel":
                if ifm_dim_h == 1 or ifm_dim_w == 1:
                    return 0  # 1D case (no line buffers needed)
                kernel_width = (k_w - 1) * dilation_w + 1
                buffer_depth = (ifm_dim_w - kernel_width) + ifm_dim_w * (dilation_h - 1)
                buffer_count = k_h - 1

            # NOTE: Actual BRAM usage might be lower in some cases
            # due to imperfect modeling of Vivado behavior
            if buffer_depth <= 512:
                ram_width = 36
            elif buffer_depth <= 1024:
                ram_width = 18
            elif buffer_depth <= 2048:
                ram_width = 9
            elif buffer_depth <= 4096:
                ram_width = 4
            elif buffer_depth <= 8192:
                ram_width = 2
            else:
                ram_width = 1

            ram_cascade_depth = math.ceil(buffer_depth / 16384)
            ram_cascade_width = math.ceil(buffer_width / ram_width)
            cascade_savings = 0
            if buffer_depth > 16384:
                remainder_depth = buffer_depth % 16384
                if remainder_depth <= 512:
                    remainder_width = 36
                elif remainder_depth <= 1024:
                    remainder_width = 18
                elif remainder_depth <= 2048:
                    remainder_width = 9
                elif remainder_depth <= 4096:
                    remainder_width = 4
                elif remainder_depth <= 8192:
                    remainder_width = 2
                else:
                    remainder_width = 1

                remainder_cascade_width = math.ceil(buffer_width / remainder_width)
                cascade_savings = ram_cascade_width - remainder_cascade_width

            return int((ram_cascade_depth * ram_cascade_width - cascade_savings) * buffer_count)
        else:
            return 0

    def lut_estimation(self):
        simd = self.get_nodeattr("SIMD")
        ram_style = self.get_nodeattr("ram_style")
        buffer_width = simd * self.get_input_datatype().bitwidth()
        buffer_depth = self.get_buffer_depth()
        if ram_style == "distributed":
            ram_luts = int(buffer_width * math.ceil(buffer_depth / 38))
        else:
            ram_luts = 0
        return 300 + ram_luts

    def uram_estimation(self):
        simd = self.get_nodeattr("SIMD")
        ram_style = self.get_nodeattr("ram_style")
        impl_style = self.select_impl_style()
        [k_h, k_w] = self.get_nodeattr("ConvKernelDim")
        [ifm_dim_h, ifm_dim_w] = self.get_nodeattr("IFMDim")
        [dilation_h, dilation_w] = self.get_nodeattr("Dilation")

        if ram_style == "ultra":
            buffer_width = simd * self.get_input_datatype().bitwidth()
            if impl_style == "default":
                buffer_depth = self.get_buffer_depth()
                buffer_count = 1
            elif impl_style == "parallel":
                if ifm_dim_h == 1 or ifm_dim_w == 1:
                    return 0  # 1D case (no line buffers needed)
                kernel_width = (k_w - 1) * dilation_w + 1
                buffer_depth = (ifm_dim_w - kernel_width) + ifm_dim_w * (dilation_h - 1)
                buffer_count = k_h - 1

            ram_depth = 4096
            ram_width = 72
            ram_cascade_depth = math.ceil(buffer_depth / ram_depth)
            ram_cascade_width = math.ceil(buffer_width / ram_width)
            return int(ram_cascade_depth * ram_cascade_width * buffer_count)
        else:
            return 0

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node
        exp_ishape = self.get_normal_input_shape()
        exp_oshape = self.get_normal_output_shape()
        folded_ishape = self.get_folded_input_shape()

        if mode == "cppsim":
            raise Exception("cppsim not possible for RTL SWG, please set exec_mode to rtlsim")
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
        ), """Input shape doesn't match expected shape (1, ifm_dim, ifm_dim, ifm_ch)."""
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

        sim = self.get_rtlsim()
        nbits = self.get_instream_width()
        rtlsim_inp = npy_to_rtlsim_input("{}/input_0.npy".format(code_gen_dir), export_idt, nbits)
        super().reset_rtlsim(sim)
        super().toggle_clk(sim)
        rtlsim_output = self.rtlsim(sim, rtlsim_inp)
        odt = export_idt
        target_bits = odt.bitwidth()
        packed_bits = self.get_outstream_width()
        out_npy_path = "{}/output.npy".format(code_gen_dir)
        out_shape = self.get_folded_output_shape()
        rtlsim_output_to_npy(rtlsim_output, out_npy_path, odt, out_shape, packed_bits, target_bits)
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

    def prepare_codegen_default(self):
        """Fills code generation dict for the default implementation style by computing
        the incremental addressing scheme for the circular buffer."""
        if self.get_nodeattr("dynamic_mode"):
            template_select = "/finn-rtllib/swg/swg_template_default_dynamic.sv"
        else:
            template_select = "/finn-rtllib/swg/swg_template_default.sv"
        template_path = os.environ["FINN_ROOT"] + template_select
        code_gen_dict = {}

        ifm_ch = self.get_nodeattr("IFMChannels")
        k = self.get_nodeattr("ConvKernelDim")
        ifm_dim = self.get_nodeattr("IFMDim")
        stride = self.get_nodeattr("Stride")
        dilation = self.get_nodeattr("Dilation")
        depthwise = self.get_nodeattr("depthwise")
        simd = self.get_nodeattr("SIMD")

        k_h, k_w = k
        h, w = ifm_dim
        pad = [0, 0, 0, 0]  # padding happens in separate padding node for now
        stride_h, stride_w = stride
        dilation_h, dilation_w = dilation
        pad_h = pad[0] + pad[2]
        pad_w = pad[1] + pad[3]
        out_dim_h = im2col.compute_conv_output_dim(h, k_h, stride_h, pad_h, dilation_h)
        out_dim_w = im2col.compute_conv_output_dim(w, k_w, stride_w, pad_w, dilation_w)
        mmv_in = 1
        mmv_out = 1
        channel_factor = int(ifm_ch / simd)

        # compute minimal buffer length (assuming it holds 1 complete window)
        buffer_min_size = ((k_h - 1) * dilation_h * w + (k_w - 1) * dilation_w + 1) * channel_factor

        buffer_actual_size = self.get_buffer_depth()
        code_gen_dict["$BUF_ELEM_TOTAL$"] = [str(buffer_actual_size)]

        # compute some intermediate values, e.g., kernel "width" = k_w incl. dilation
        # or cols/rows that are skipped due to imperfect stride<->dim combination
        kernel_width = (k_w - 1) * dilation_w + 1
        kernel_height = (k_h - 1) * dilation_h + 1
        skip_columns = w % (kernel_width + (out_dim_w - 1) * stride_w)
        skip_rows = h % (kernel_height + (out_dim_h - 1) * stride_h)

        # compute address increment values for 5-loop nest
        addr_incr_end_simd = 1
        addr_incr_end_window_elem = (dilation_w - 1) * channel_factor + 1
        addr_incr_end_window_row = (
            ((w - kernel_width) * channel_factor)  # remaining line
            + ((dilation_h - 1) * w * channel_factor)  # skip lines
            + 1  # wrap-around of minimally sized buffer
        )
        addr_incr_end_window = -buffer_min_size + stride_w * channel_factor + 1
        addr_incr_end_row = (
            -buffer_min_size
            + ((skip_columns + kernel_width) * channel_factor)  # remaining line
            + ((stride_h - 1) * w * channel_factor)  # skip lines
            + 1
        )

        # re-use same controller structure -> re-assign address increments
        if depthwise:
            addr_incr_end_window_elem = dilation_w * channel_factor
            addr_incr_end_window_row = (
                channel_factor
                + (w - kernel_width) * channel_factor
                + (dilation_h - 1) * w * channel_factor
            )
            addr_incr_end_simd = -buffer_min_size + (channel_factor + 1)

        # sanity check for wrap logic
        assert not (
            abs(addr_incr_end_window) > buffer_actual_size
        ), "ERROR: W increment > buffer size, try setting parallel_window=1"
        assert not (
            abs(addr_incr_end_row) > buffer_actual_size
        ), "ERROR: H increment > buffer size, try setting parallel_window=1"

        # set certain threshold indices to detect when reading/writing finishes
        code_gen_dict["$LAST_READ_ELEM$"] = [str(h * w * channel_factor - 1)]
        code_gen_dict["$LAST_WRITE_ELEM$"] = [
            str(((h - skip_rows - 1) * w + (w - skip_columns)) * channel_factor - 1)
        ]

        # default controller loop structure: # iterations (counters) map directly
        loop_h_iterations = out_dim_h
        loop_w_iterations = out_dim_w
        loop_kh_iterations = k_h
        loop_kw_iterations = k_w
        loop_simd_iterations = channel_factor

        if depthwise and channel_factor > 1:
            # re-arrange existing controller loop structure for depthwise convolutions
            loop_kh_iterations = channel_factor
            loop_kw_iterations = k_h
            loop_simd_iterations = k_w
            addr_incr_end_simd_ = addr_incr_end_simd
            addr_incr_end_simd = addr_incr_end_window_elem
            addr_incr_end_window_elem = addr_incr_end_window_row
            addr_incr_end_window_row = addr_incr_end_simd_
            elem_per_window = k_h * k_w

            tail_incr_w = addr_incr_end_window + buffer_min_size - channel_factor
            tail_incr_h = addr_incr_end_row + buffer_min_size - channel_factor
            tail_incr_last_window = buffer_min_size - 1
            code_gen_dict["$IS_DEPTHWISE$"] = ["1"]
        else:
            # depthwise output format is equivalent to non-depthwise if SIMD=C
            elem_per_window = k_h * k_w * channel_factor

            tail_incr_w = addr_incr_end_window + buffer_min_size - 1
            tail_incr_h = addr_incr_end_row + buffer_min_size - 1
            tail_incr_last_window = buffer_min_size - 1
            code_gen_dict["$IS_DEPTHWISE$"] = ["0"]

        # support SIMD = IFMChannels and k_w = 1 cases
        # for k = [k_h, k_w] = [1, k_w], no adjustment is needed
        # for k = [k_h, k_w] = [1, 1], do not use this impl. style (mmv_out=K=1)
        # innermost loop is executed at least once -> adjust if needed
        if loop_simd_iterations == 1:
            # skip innermost SIMD loop completely
            if loop_kw_iterations == 1:
                # skip innermost KW loop completely
                code_gen_dict["$INNERMOST_STATE$"] = ["STATE_LOOP_KH"]
                loop_kh_iterations -= 1  # -1 because state is initial state
            else:
                code_gen_dict["$INNERMOST_STATE$"] = ["STATE_LOOP_KW"]
                loop_kw_iterations -= 1  # -1 because state is initial state
        else:
            code_gen_dict["$INNERMOST_STATE$"] = ["STATE_LOOP_SIMD"]
            loop_simd_iterations -= 1  # -1 because state is initial state

        cntr_bitwidth = math.ceil(
            math.log2(
                max(
                    loop_h_iterations - 2 + 1,
                    loop_w_iterations - 2 + 1,
                    loop_kh_iterations - 2 + 1,
                    loop_kw_iterations - 2 + 1,
                    loop_simd_iterations - 2 + 1,
                )
            )
        )
        code_gen_dict["$CNTR_BITWIDTH$"] = [str(cntr_bitwidth)]
        code_gen_dict["$LOOP_H_ITERATIONS$"] = [str(loop_h_iterations - 2)]
        code_gen_dict["$LOOP_W_ITERATIONS$"] = [str(loop_w_iterations - 2)]
        code_gen_dict["$LOOP_KH_ITERATIONS$"] = [str(loop_kh_iterations - 2)]
        code_gen_dict["$LOOP_KW_ITERATIONS$"] = [str(loop_kw_iterations - 2)]
        code_gen_dict["$LOOP_SIMD_ITERATIONS$"] = [str(loop_simd_iterations - 2)]

        incr_bitwidth = 1 + math.ceil(
            math.log2(
                max(
                    abs(addr_incr_end_simd) + 1,
                    abs(addr_incr_end_window_elem) + 1,
                    abs(addr_incr_end_window_row) + 1,
                    abs(addr_incr_end_window) + 1,
                    abs(addr_incr_end_row) + 1,
                    abs(tail_incr_w) + 1,
                    abs(tail_incr_h) + 1,
                    abs(tail_incr_last_window) + 1,
                )
            )
        )
        code_gen_dict["$INCR_BITWIDTH$"] = [str(incr_bitwidth)]
        code_gen_dict["$HEAD_INCR_SIMD$"] = [str(addr_incr_end_simd)]
        code_gen_dict["$HEAD_INCR_KW$"] = [str(addr_incr_end_window_elem)]
        code_gen_dict["$HEAD_INCR_KH$"] = [str(addr_incr_end_window_row)]
        code_gen_dict["$HEAD_INCR_W$"] = [str(addr_incr_end_window)]
        code_gen_dict["$HEAD_INCR_H$"] = [str(addr_incr_end_row)]
        code_gen_dict["$TAIL_INCR_W$"] = [str(tail_incr_w)]
        code_gen_dict["$TAIL_INCR_H$"] = [str(tail_incr_h)]
        code_gen_dict["$TAIL_INCR_LAST$"] = [str(tail_incr_last_window)]

        code_gen_dict["$ELEM_PER_WINDOW$"] = [str(elem_per_window)]
        code_gen_dict["$SIMD$"] = [str(simd)]
        code_gen_dict["$MMV_IN$"] = [str(mmv_in)]
        code_gen_dict["$MMV_OUT$"] = [str(mmv_out)]

        return template_path, code_gen_dict

    def prepare_codegen_parallel(self):
        """Fills code generation dict for the parallel implementation style by computing
        the loop controller configuration and partitioning the fixed buffer into
        shift-registers (for parallel read access) and line buffers (for efficient
        LUTRAM/BRAM/URAM implementation)."""
        template_path = os.environ["FINN_ROOT"] + "/finn-rtllib/swg/swg_template_parallel.sv"
        code_gen_dict = {}

        ifm_ch = self.get_nodeattr("IFMChannels")
        k = self.get_nodeattr("ConvKernelDim")
        ifm_dim = self.get_nodeattr("IFMDim")
        stride = self.get_nodeattr("Stride")
        dilation = self.get_nodeattr("Dilation")
        simd = self.get_nodeattr("SIMD")
        M = self.get_nodeattr("M")

        k_h, k_w = k
        h, w = ifm_dim
        pad = [0, 0, 0, 0]  # padding happens in separate padding node for now
        stride_h, stride_w = stride
        dilation_h, dilation_w = dilation
        pad_h = pad[0] + pad[2]
        pad_w = pad[1] + pad[3]
        out_dim_h = im2col.compute_conv_output_dim(h, k_h, stride_h, pad_h, dilation_h)
        out_dim_w = im2col.compute_conv_output_dim(w, k_w, stride_w, pad_w, dilation_w)
        mmv_in = M * 1
        mmv_out = M * k_h * k_w
        channel_factor = int(ifm_ch / simd)

        # compute minimal buffer length (assuming it holds 1 complete window)
        buffer_min_size = ((k_h - 1) * dilation_h * w + (k_w - 1) * dilation_w) * channel_factor + 1

        buffer_actual_size = self.get_buffer_depth()
        code_gen_dict["$BUF_ELEM_TOTAL$"] = [str(buffer_actual_size)]

        # compute some intermediate values, e.g., kernel "width" = k_w incl. dilation
        # or cols/rows that are skipped due to imperfect stride<->dim combination
        kernel_width = (k_w - 1) * dilation_w + 1
        kernel_height = (k_h - 1) * dilation_h + 1
        skip_columns = w % (kernel_width + (out_dim_w - 1) * stride_w)
        skip_rows = h % (kernel_height + (out_dim_h - 1) * stride_h)

        # set certain threshold indices to detect when reading/writing finishes
        code_gen_dict["$LAST_READ_ELEM$"] = [str(h * w * channel_factor - 1)]
        code_gen_dict["$LAST_WRITE_ELEM$"] = [
            str(((h - skip_rows - 1) * w + (w - skip_columns)) * channel_factor - 1)
        ]

        # re-use default controller loop structure
        loop_h_iterations = out_dim_h
        loop_w_iterations = out_dim_w
        loop_kh_iterations = channel_factor
        loop_kw_iterations = 1
        loop_simd_iterations = 1

        if loop_kh_iterations == 1:
            if loop_w_iterations == 1:
                code_gen_dict["$INNERMOST_STATE$"] = ["STATE_LOOP_H"]
                loop_h_iterations -= 1  # -1 because state is initial state
            else:
                code_gen_dict["$INNERMOST_STATE$"] = ["STATE_LOOP_W"]
                loop_w_iterations -= 1  # -1 because state is initial state
        else:
            code_gen_dict["$INNERMOST_STATE$"] = ["STATE_LOOP_KH"]
            loop_kh_iterations -= 1  # -1 because state is initial state

        # set head address increment values
        addr_incr_end_simd = 1
        addr_incr_end_window_elem = 1
        addr_incr_end_window_row = 1
        addr_incr_end_window = (stride_w - 1) * channel_factor + 1
        addr_incr_end_row = ((skip_columns + (kernel_width - 1)) * channel_factor + 1) + (
            (stride_h - 1) * w * channel_factor
        )

        # add init value for CURRENT_ELEM counter = last elem of first window
        code_gen_dict["$FIRST_WRITE_ELEM$"] = [str(buffer_min_size - 1)]

        cntr_bitwidth = math.ceil(
            math.log2(
                max(
                    loop_h_iterations - 2 + 1,
                    loop_w_iterations - 2 + 1,
                    loop_kh_iterations - 2 + 1,
                    loop_kw_iterations - 2 + 1,
                    loop_simd_iterations - 2 + 1,
                )
            )
        )
        code_gen_dict["$CNTR_BITWIDTH$"] = [str(cntr_bitwidth)]
        code_gen_dict["$LOOP_H_ITERATIONS$"] = [str(loop_h_iterations - 2)]
        code_gen_dict["$LOOP_W_ITERATIONS$"] = [str(loop_w_iterations - 2)]
        code_gen_dict["$LOOP_KH_ITERATIONS$"] = [str(loop_kh_iterations - 2)]
        code_gen_dict["$LOOP_KW_ITERATIONS$"] = [str(loop_kw_iterations - 2)]
        code_gen_dict["$LOOP_SIMD_ITERATIONS$"] = [str(loop_simd_iterations - 2)]

        incr_bitwidth = 1 + math.ceil(
            math.log2(
                max(
                    abs(addr_incr_end_simd) + 1,
                    abs(addr_incr_end_window_elem) + 1,
                    abs(addr_incr_end_window_row) + 1,
                    abs(addr_incr_end_window) + 1,
                    abs(addr_incr_end_row) + 1,
                )
            )
        )
        code_gen_dict["$INCR_BITWIDTH$"] = [str(incr_bitwidth)]
        code_gen_dict["$HEAD_INCR_SIMD$"] = [str(addr_incr_end_simd)]
        code_gen_dict["$HEAD_INCR_KW$"] = [str(addr_incr_end_window_elem)]
        code_gen_dict["$HEAD_INCR_KH$"] = [str(addr_incr_end_window_row)]
        code_gen_dict["$HEAD_INCR_W$"] = [str(addr_incr_end_window)]
        code_gen_dict["$HEAD_INCR_H$"] = [str(addr_incr_end_row)]
        # not used, set to zero:
        code_gen_dict["$TAIL_INCR_W$"] = ["0"]
        code_gen_dict["$TAIL_INCR_H$"] = ["0"]
        code_gen_dict["$TAIL_INCR_LAST$"] = ["0"]
        code_gen_dict["$IS_DEPTHWISE$"] = ["0"]

        code_gen_dict["$SIMD$"] = [str(simd)]
        code_gen_dict["$MMV_IN$"] = [str(mmv_in)]
        code_gen_dict["$MMV_OUT$"] = [str(mmv_out)]

        # prepare buffer partitioning into "reg_fifos" and "bram_fifos"
        # use normalized ([H,W]=[1,W]) dimensions for 1D case
        (
            ifm_ch,
            [ifm_dim_h, ifm_dim_w],
            [ofm_dim_h, ofm_dim_w],
            [k_h, k_w],
            [stride_h, stride_w],
            [dilation_h, dilation_w],
        ) = self.get_1d_conv_attrs_normalized()

        reg_fifos = []
        bram_fifos_depth = []

        px_idx = 0
        for ky in range(k_h):
            reg_fifo = []
            for kx in range(k_w):
                for c in range(channel_factor):
                    if c < (channel_factor - 1):
                        if not (ky == 0 and kx == 0):
                            reg_fifo.append(-1)
                            px_idx += 1
                    else:
                        reg_fifo.append(px_idx)
                        px_idx += 1
                if kx < (k_w - 1):
                    reg_fifo.extend([-1] * ((dilation_w - 1) * channel_factor))
                    px_idx += (dilation_w - 1) * channel_factor
            reg_fifos.append(reg_fifo)

            if ky < (k_h - 1):
                line_buffer_len = ((w - kernel_width) + w * (dilation_h - 1)) * channel_factor
                bram_fifos_depth.append(line_buffer_len)
                px_idx += line_buffer_len

        code_gen_dict["$GENERATE_REG_FIFOS$"] = []
        for i, reg_fifo in enumerate(reg_fifos):
            code_gen_dict["$GENERATE_REG_FIFOS$"].append(
                """
                wire [IN_WIDTH-1:0] reg_fifo_{id}_in;
                wire [IN_WIDTH-1:0] reg_fifo_{id}_out;
                wire [IN_WIDTH*{len}-1:0] reg_fifo_{id};
                swg_reg_buffer
                #(
                .WIDTH(IN_WIDTH),
                .DEPTH({len})
                )
                reg_buffer_inst_{id}
                (
                    .clk(clk),
                    .shift_enable(shift_enable),
                    .shift_in(reg_fifo_{id}_in),
                    .shift_out(reg_fifo_{id}_out),
                    .data_out(reg_fifo_{id})
                );""".format(
                    id=i,
                    len=len(reg_fifo),
                )
            )

        code_gen_dict["$GENERATE_BRAM_FIFOS$"] = []
        for i, bram_fifo_depth in enumerate(bram_fifos_depth):
            code_gen_dict["$GENERATE_BRAM_FIFOS$"].append(
                """
                wire [IN_WIDTH-1:0] bram_fifo_{id}_in;
                wire [IN_WIDTH-1:0] bram_fifo_{id}_out;
                swg_ram_buffer
                #(
                .WIDTH(IN_WIDTH),
                .DEPTH({len}),
                .RAM_STYLE("{ram_style}")
                )
                ram_buffer_inst_{id}
                (
                    .clk(clk),
                    .rst_n(rst_n),
                    .shift_enable(shift_enable),
                    .shift_in(bram_fifo_{id}_in),
                    .shift_out(bram_fifo_{id}_out)
                );""".format(
                    id=i,
                    len=bram_fifo_depth,
                    ram_style=self.get_nodeattr("ram_style"),
                )
            )

        code_gen_dict["$GENERATE_OUTPUT_MAPPING$"] = []
        out_idx = mmv_out - 1
        for fifo_id, reg_fifo in enumerate(reg_fifos):
            for fifo_idx, access_idx in enumerate(reg_fifo):
                if access_idx != -1:
                    code_gen_dict["$GENERATE_OUTPUT_MAPPING$"].append(
                        """assign data_out[OUT_ELEM_WIDTH*{out_idx}+:OUT_ELEM_WIDTH]
                        = reg_fifo_{fifo_id}[{access_idx}*{mmv}*OUT_ELEM_WIDTH+
                        OUT_ELEM_WIDTH*{mmv_idx}+:OUT_ELEM_WIDTH];""".format(
                            out_idx=out_idx,
                            fifo_id=fifo_id,
                            access_idx=len(reg_fifo) - 1 - int((max(reg_fifo) - access_idx) / M),
                            mmv_idx=(max(reg_fifo) - access_idx) % M,
                            mmv=M,
                        )
                    )
                    # reversal: out_idx=0 -> oldest buffer element -> highest access_idx
                    out_idx = out_idx - 1
        assert out_idx == -1, "ERROR: Not all output vector elements connected"

        code_gen_dict["$GENERATE_BUFFER_CONNECTION$"] = []
        for i in range(len(reg_fifos)):
            if i == 0:
                # first FIFO containing newest elements -> input comes from input reg
                code_gen_dict["$GENERATE_BUFFER_CONNECTION$"].append(
                    """assign reg_fifo_{fifo_id}_in = data_in;""".format(
                        fifo_id=i,
                    )
                )
            else:
                # other REG FIFOs -> input comes from connected BRAM FIFO (line buffer)
                input_fifo_id = i - 1
                code_gen_dict["$GENERATE_BUFFER_CONNECTION$"].append(
                    """assign reg_fifo_{fifo_id}_in = bram_fifo_{input_fifo_id}_out;
                    """.format(
                        fifo_id=i, input_fifo_id=input_fifo_id
                    )
                )
        for i in range(len(bram_fifos_depth)):
            input_fifo_id = i
            code_gen_dict["$GENERATE_BUFFER_CONNECTION$"].append(
                """assign bram_fifo_{fifo_id}_in = reg_fifo_{input_fifo_id}_out;
                """.format(
                    fifo_id=i, input_fifo_id=input_fifo_id
                )
            )

        return template_path, code_gen_dict

    def select_impl_style(self):
        """Selects implementation style based on folding configuration."""
        simd = self.get_nodeattr("SIMD")
        M = self.get_nodeattr("M")
        depthwise = self.get_nodeattr("depthwise")
        ifm_ch = self.get_nodeattr("IFMChannels")
        ifm_dim = self.get_nodeattr("IFMDim")
        stride = self.get_nodeattr("Stride")
        dilation = self.get_nodeattr("Dilation")
        k = self.get_nodeattr("ConvKernelDim")
        ifm_dim_h, ifm_dim_w = ifm_dim
        stride_h, stride_w = stride
        dilation_h, dilation_w = dilation
        k_h, k_w = k
        kernel_width = (k_w - 1) * dilation_w + 1  # incl. dilation
        kernel_height = (k_h - 1) * dilation_h + 1  # incl. dilation

        # check for valid configuration
        assert (
            kernel_height <= ifm_dim_h
            and kernel_width <= ifm_dim_w
            and stride_h <= ifm_dim_h
            and stride_w <= ifm_dim_w
        ), "Illegal conv configuration: kernel or stride > FM dimension"

        # init folding config
        if self.get_nodeattr("parallel_window"):
            # mmv_in = M * 1
            mmv_out = M * k_h * k_w
        else:
            # mmv_in = 1
            mmv_out = 1
            assert ifm_ch % simd == 0, "Constraint violated: SIMD must divide IFMChannels"

        # choose implementation style
        if mmv_out > 1 or (k_h == 1 and k_w == 1):
            impl_style = "parallel"
            if depthwise or (k_h == 1 and k_w == 1):
                # allow SIMD < IFM_CH in depthwise mode (VVAU supports the resulting data layout)
                # also allowed for 1x1 kernel since depthwise and non-depthwise are equivalent
                assert ifm_ch % simd == 0, "Constraint violated: SIMD must divide IFMChannels"
            else:
                assert ifm_ch == simd, "Constraint violated: SIMD must be equal to IFMChannels"
        else:
            impl_style = "default"

        return impl_style

    def generate_hdl(self):
        """Generates HDL code and wrapper for the IP, depending on required
        implementation style."""
        impl_style = self.select_impl_style()

        # prepare code generation by filling out dictionaries
        if impl_style == "default":
            template_path, code_gen_dict = self.prepare_codegen_default()
        elif impl_style == "parallel":
            template_path, code_gen_dict = self.prepare_codegen_parallel()
            if self.get_nodeattr("dynamic_mode"):
                raise Exception("Dynamic mode is not compatible with parallel_window")
        else:
            raise Exception("Requested impl. style not implemented")

        # add general parameters to dictionary
        code_gen_dict["$TOP_MODULE_NAME$"] = [self.get_verilog_top_module_name()]
        # save top module name so we can refer to it after this node has been renamed
        # (e.g. by GiveUniqueNodeNames(prefix) during MakeZynqProject)
        self.set_nodeattr("gen_top_module", self.get_verilog_top_module_name())
        code_gen_dict["$BIT_WIDTH$"] = [str(self.get_input_datatype().bitwidth())]
        ram_style = self.get_nodeattr("ram_style")
        code_gen_dict["$RAM_STYLE$"] = ['"{}"'.format(ram_style)]

        # apply code generation to templates
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        with open(template_path, "r") as f:
            template = f.read()
        if self.get_nodeattr("dynamic_mode"):
            template_select = "/finn-rtllib/swg/swg_template_wrapper_dynamic.v"
        else:
            template_select = "/finn-rtllib/swg/swg_template_wrapper.v"
        with open(os.environ["FINN_ROOT"] + template_select, "r") as f:
            template_wrapper = f.read()
        with open(os.environ["FINN_ROOT"] + "/finn-rtllib/swg/swg_template_axilite.v", "r") as f:
            template_axilite = f.read()
        for key in code_gen_dict:
            # transform list into long string separated by '\n'
            code_gen_line = "\n".join(code_gen_dict[key])
            template = template.replace(key, code_gen_line)
            template_wrapper = template_wrapper.replace(key, code_gen_line)
            template_axilite = template_axilite.replace(key, code_gen_line)
        with open(
            os.path.join(code_gen_dir, self.get_nodeattr("gen_top_module") + "_impl.sv"),
            "w",
        ) as f:
            f.write(template)
        with open(
            os.path.join(code_gen_dir, self.get_nodeattr("gen_top_module") + "_wrapper.v"),
            "w",
        ) as f:
            f.write(template_wrapper)

        # AXI-Lite reg. file component is only needed for dynamic mode
        if self.get_nodeattr("dynamic_mode"):
            with open(
                os.path.join(code_gen_dir, self.get_nodeattr("gen_top_module") + "_axilite.v"),
                "w",
            ) as f:
                f.write(template_axilite)

        # Copy static source file for common core components
        shutil.copy2(os.environ["FINN_ROOT"] + "/finn-rtllib/swg/swg_common.sv", code_gen_dir)
        shutil.copy2(os.environ["FINN_ROOT"] + "/finn-rtllib/swg/swg_pkg.sv", code_gen_dir)

        # set ipgen_path and ip_path so that HLS-Synth transformation
        # and stich_ip transformation do not complain
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def prepare_rtlsim(self):
        """Creates a Verilator emulation library for the RTL code generated
        for this node, sets the rtlsim_so attribute to its path and returns
        a PyVerilator wrapper around it."""
        # Modified to use generated (System-)Verilog instead of HLS output products

        if PyVerilator is None:
            raise ImportError("Installation of PyVerilator is required.")

        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        verilog_paths = [code_gen_dir]
        verilog_files = [
            "swg_pkg.sv",
            self.get_nodeattr("gen_top_module") + "_wrapper.v",
            self.get_nodeattr("gen_top_module") + "_impl.sv",
            "swg_common.sv",
        ]
        if self.get_nodeattr("dynamic_mode"):
            verilog_files.append(self.get_nodeattr("gen_top_module") + "_axilite.v")

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
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

        sourcefiles = [
            "swg_pkg.sv",
            self.get_nodeattr("gen_top_module") + "_wrapper.v",
            self.get_nodeattr("gen_top_module") + "_impl.sv",
            "swg_common.sv",
        ]

        if self.get_nodeattr("dynamic_mode"):
            sourcefiles += [self.get_nodeattr("gen_top_module") + "_axilite.v"]

        sourcefiles = [os.path.join(code_gen_dir, f) for f in sourcefiles]

        cmd = []
        for f in sourcefiles:
            cmd += ["add_files -norecurse %s" % (f)]
        cmd += [
            "create_bd_cell -type module -reference %s %s"
            % (self.get_nodeattr("gen_top_module"), self.onnx_node.name)
        ]
        return cmd

    def get_verilog_top_module_intf_names(self):
        # Overload default HLSCustomOp implementation to add axilite control IF
        """Return a dict of names of input and output interfaces.
        The keys reflect the protocols each interface implements:
        'clk', 'rst', 'm_axis', 's_axis', 'aximm', 'axilite'.
        Values are lists of tuples (axis, aximm) or names (axilite):
        'axis' tuples correspond to the list of node inputs in order,
        each tuple is (interface_name, interface_width_bits).
        axilite always assumed to be 32 bits and is not tuple (name only).
        Each block must have at most one aximm and one axilite."""
        intf_names = super().get_verilog_top_module_intf_names()
        if self.get_nodeattr("dynamic_mode"):
            intf_names["axilite"] = ["s_axilite"]
        return intf_names

    def get_dynamic_config(self, ifm_dim=None, stride=None, dilation=None):
        """Returns a configuration dict to re-configure FM dimension during
        runtime. Stride and dilation can also be changed. Certain restrictions
        apply (e.g. component must be synthesized for largest buffer size)."""
        # NOTE: For better driver integration, this functionality could be packaged
        # as a standalone function in the future
        if self.select_impl_style() != "default":
            raise Exception("Impl. style is incompatible with dynamic mode")

        if ifm_dim is None:
            ifm_dim = self.get_nodeattr("IFMDim")
        k = self.get_nodeattr("ConvKernelDim")
        if stride is None:
            stride = self.get_nodeattr("Stride")
        if dilation is None:
            dilation = self.get_nodeattr("Dilation")

        k_h, k_w = k
        stride_h, stride_w = stride
        dilation_h, dilation_w = dilation
        ifm_dim_h, ifm_dim_w = ifm_dim
        ofm_dim_h = compute_conv_output_dim(ifm_dim_h, k_h, stride_h, 0, dilation_h)
        ofm_dim_w = compute_conv_output_dim(ifm_dim_w, k_w, stride_w, 0, dilation_w)
        ofm_dim = [ofm_dim_h, ofm_dim_w]

        # update attributes and perform sanity check
        original_buffer_depth = self.get_buffer_depth()
        self.set_nodeattr("IFMDim", ifm_dim)
        self.set_nodeattr("OFMDim", ofm_dim)
        self.set_nodeattr("Stride", stride)
        self.set_nodeattr("Dilation", dilation)
        assert (
            self.get_buffer_depth() <= original_buffer_depth
        ), """Error: requested
            dynamic configuration does not fit in generated buffer implementation."""

        # (re-)call codegen and extract new values
        # each setting is mapped to an axi-lite register address
        template_path, code_gen_dict = self.prepare_codegen_default()
        config = {
            "cfg_wren": (0 * 4, 1),
            "cfg_cntr_simd": (1 * 4, int(code_gen_dict["$LOOP_SIMD_ITERATIONS$"][0])),
            "cfg_cntr_kw": (2 * 4, int(code_gen_dict["$LOOP_KW_ITERATIONS$"][0])),
            "cfg_cntr_kh": (3 * 4, int(code_gen_dict["$LOOP_KH_ITERATIONS$"][0])),
            "cfg_cntr_w": (4 * 4, int(code_gen_dict["$LOOP_W_ITERATIONS$"][0])),
            "cfg_cntr_h": (5 * 4, int(code_gen_dict["$LOOP_H_ITERATIONS$"][0])),
            "cfg_incr_head_simd": (6 * 4, int(code_gen_dict["$HEAD_INCR_SIMD$"][0])),
            "cfg_incr_head_kw": (7 * 4, int(code_gen_dict["$HEAD_INCR_KW$"][0])),
            "cfg_incr_head_kh": (8 * 4, int(code_gen_dict["$HEAD_INCR_KH$"][0])),
            "cfg_incr_head_w": (9 * 4, int(code_gen_dict["$HEAD_INCR_W$"][0])),
            "cfg_incr_head_h": (10 * 4, int(code_gen_dict["$HEAD_INCR_H$"][0])),
            "cfg_incr_tail_w": (11 * 4, int(code_gen_dict["$TAIL_INCR_W$"][0])),
            "cfg_incr_tail_h": (12 * 4, int(code_gen_dict["$TAIL_INCR_H$"][0])),
            "cfg_incr_tail_last": (13 * 4, int(code_gen_dict["$TAIL_INCR_LAST$"][0])),
            "cfg_last_read": (14 * 4, int(code_gen_dict["$LAST_READ_ELEM$"][0])),
            "cfg_last_write": (15 * 4, int(code_gen_dict["$LAST_WRITE_ELEM$"][0])),
        }
        return config
