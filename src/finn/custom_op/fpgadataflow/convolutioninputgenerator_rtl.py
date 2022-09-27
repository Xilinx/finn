# Copyright (C) 2022, Advanced Micro Devices, Inc.
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
from math import copysign
from qonnx.core.datatype import DataType
from qonnx.custom_op.general import im2col
from qonnx.custom_op.general.im2col import compute_conv_output_dim

from finn.custom_op.fpgadataflow.hlscustomop import HLSCustomOp
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


class ConvolutionInputGenerator_rtl(HLSCustomOp):
    """Class that does not correspond to one of the finn-hlslib ConvolutionInputGenerator
    (sliding window) function variants. Generates an RTL ConvolutionInputGenerator
    implementation based on (System-)Verilog templates, defined in finn-rtllib/swg."""

    def __init__(self, onnx_node):
        super().__init__(onnx_node)

    def get_nodeattr_types(self):
        my_attrs = {
            "ConvKernelDim": ("ints", True, []),  # [H, W] = [Y, X]
            "IFMChannels": ("i", True, 0),
            "IFMDim": ("ints", True, []),  # [H, W] = [Y, X]
            "OFMDim": ("ints", True, []),  # [H, W] = [Y, X]
            "SIMD": ("i", True, 0),
            # additional parallelization parameter - not yet implemented
            "M": ("i", False, 1),
            # alternative implementation style - not yet implemented
            "parallel_window": ("i", False, 0, {0}),
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
            # attribute to save top module name - not user configurable
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
        if self.get_nodeattr("parallel_window"):
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
        if self.get_nodeattr("parallel_window"):
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

    def get_1d_conv_attrs_normalized(self):
        # normalize FM dimensions so that:
        # [H, W] = [Y, X] = [1, D] or [D, 1] are always mapped to [1, D].
        # The dummy ('1') dimension is the Y-dimension.
        ifm_ch = self.get_nodeattr("IFMChannels")
        k = self.get_nodeattr("ConvKernelDim")
        ifm_dim = self.get_nodeattr("IFMDim")
        ofm_dim = self.get_nodeattr("OFMDim")
        stride = self.get_nodeattr("Stride")
        dilation = self.get_nodeattr("Dilation")

        if ifm_dim[1] == 1:
            ifm_dim = ifm_dim[::-1]
            ofm_dim = ofm_dim[::-1]
            k = k[::-1]
            stride = stride[::-1]
            dilation = dilation[::-1]

        return (ifm_ch, ifm_dim, ofm_dim, k, stride, dilation)

    def get_buffer_depth(self):
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
            # compute minimal buffer length (assuming it holds 1 complete window)
            buffer_min_size = (
                (k_h - 1) * dilation_h * w + (k_w - 1) * dilation_w + 1
            ) * channel_factor

            # add additional buffer space in case of stride > 1
            # this minimizes cycle count as it allows an earlier pre-load of inputs
            buffer_depth = (
                buffer_min_size
                + max(
                    0,
                    ((stride_w - 1) - (int(mmv_out * k_h * k_w / mmv_in)))
                    * channel_factor,
                )
                + max(
                    0,
                    ((stride_h - 1) * w - (int(mmv_out * k_h * k_w / mmv_in)))
                    * channel_factor,
                )
            )
        else:
            buffer_depth = 0
            raise Exception("Requested impl. style not implemented")
        return buffer_depth

    def get_exp_cycles(self):
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
            exp_cycles = buffer_min_size + ofm_dim_h * max_cycles  # initial buffering
            if depthwise:
                exp_cycles += (stride_h - 1) * ifm_dim_w * channel_factor

        return int(exp_cycles)

    def bram_estimation(self):
        simd = self.get_nodeattr("SIMD")
        ram_style = self.get_nodeattr("ram_style")

        # NOTE: Actual BRAM usage might be lower in some cases.
        # This does not account for the exact Vivado behavior yet.
        buffer_width = simd * self.get_input_datatype().bitwidth()
        buffer_depth = self.get_buffer_depth()
        if ram_style == "block" or ram_style == "auto":
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

            return int(ram_cascade_depth * ram_cascade_width - cascade_savings)
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
        buffer_width = simd * self.get_input_datatype().bitwidth()
        buffer_depth = self.get_buffer_depth()

        if ram_style == "ultra":
            ram_depth = 4096
            ram_width = 72
            ram_cascade_depth = math.ceil(buffer_depth / ram_depth)
            ram_cascade_width = math.ceil(buffer_width / ram_width)
            return int(ram_cascade_depth * ram_cascade_width)
        else:
            return 0

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node
        exp_ishape = self.get_normal_input_shape()
        exp_oshape = self.get_normal_output_shape()
        folded_ishape = self.get_folded_input_shape()

        if mode == "cppsim":
            raise Exception(
                "cppsim not possible for RTL SWG, please set exec_mode to rtlsim"
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

    def prepare_codegen_default(self):
        # Default implementation style for MMV_out = 1: addressable cyclic buffer
        # Computing incremental addressing scheme directly..
        template_path = (
            os.environ["FINN_ROOT"] + "/finn-rtllib/swg/swg_template_default.sv"
        )
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
        buffer_min_size = (
            (k_h - 1) * dilation_h * w + (k_w - 1) * dilation_w + 1
        ) * channel_factor

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

        # sanity check
        assert not (
            abs(addr_incr_end_window) > buffer_actual_size
        ), "ERROR: W increment > buffer size, wrap logic doesn't account for this"
        assert not (
            abs(addr_incr_end_row) > buffer_actual_size
        ), "ERROR: H increment > buffer size, wrap logic doesn't account for this"

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

        code_gen_dict["$TAIL_INCR_W$"] = [str(tail_incr_w)]
        code_gen_dict["$TAIL_INCR_H$"] = [str(tail_incr_h)]
        code_gen_dict["$TAIL_INCR_LAST$"] = [str(tail_incr_last_window)]

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

        code_gen_dict["$LOOP_H_ITERATIONS$"] = [str(loop_h_iterations - 1)]
        code_gen_dict["$LOOP_W_ITERATIONS$"] = [str(loop_w_iterations - 1)]
        code_gen_dict["$LOOP_KH_ITERATIONS$"] = [str(loop_kh_iterations - 1)]
        code_gen_dict["$LOOP_KW_ITERATIONS$"] = [str(loop_kw_iterations - 1)]
        code_gen_dict["$LOOP_SIMD_ITERATIONS$"] = [str(loop_simd_iterations - 1)]

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
        code_gen_dict["$ADDR_INCREMENT_MAP$"] = [
            "'{{ {}'d0, {}'d{}, {}'d{}, {}'d{}, {}'d{}, {}'d{}}}".format(
                incr_bitwidth,
                int(copysign(incr_bitwidth, addr_incr_end_simd)),
                abs(addr_incr_end_simd),
                int(copysign(incr_bitwidth, addr_incr_end_window_elem)),
                abs(addr_incr_end_window_elem),
                int(copysign(incr_bitwidth, addr_incr_end_window_row)),
                abs(addr_incr_end_window_row),
                int(copysign(incr_bitwidth, addr_incr_end_window)),
                abs(addr_incr_end_window),
                int(copysign(incr_bitwidth, addr_incr_end_row)),
                abs(addr_incr_end_row),
            )
        ]

        code_gen_dict["$ELEM_PER_WINDOW$"] = [str(elem_per_window)]
        code_gen_dict["$SIMD$"] = [str(simd)]
        code_gen_dict["$MMV_IN$"] = [str(mmv_in)]
        code_gen_dict["$MMV_OUT$"] = [str(mmv_out)]

        return template_path, code_gen_dict

    def select_impl_style(self):
        simd = self.get_nodeattr("SIMD")
        M = self.get_nodeattr("M")
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
            assert (
                ifm_ch == simd
            ), "Constraint violated: SIMD must be equal to IFMChannels"
        else:
            # mmv_in = 1
            mmv_out = 1
            assert (
                ifm_ch % simd == 0
            ), "Constraint violated: SIMD must divide IFMChannels"

        # choose implementation style
        if mmv_out > 1 or (k_h == 1 and k_w == 1):
            impl_style = "parallel"
            assert (
                ifm_ch == simd
            ), "Constraint violated: SIMD must be equal to IFMChannels"
        else:
            impl_style = "default"

        assert (
            impl_style == "default"
        ), "ERROR: Parallel window mode not yet implemented"
        return impl_style

    def generate_hdl(self):
        impl_style = self.select_impl_style()

        # prepare code generation by filling out dictionaries
        if impl_style == "default":
            template_path, code_gen_dict = self.prepare_codegen_default()
        else:
            raise Exception("Requested impl. style not implemented")

        # add general parameters to dictionary
        code_gen_dict["$TOP_MODULE_NAME$"] = [self.get_verilog_top_module_name()]
        # save top module name so we can refer to it after this node has been renamed
        # (e.g. by GiveUniqueNodeNames(prefix) during MakeZynqProject)
        self.set_nodeattr("gen_top_module", self.get_verilog_top_module_name())
        code_gen_dict["$BIT_WIDTH$"] = [str(self.get_input_datatype().bitwidth())]
        ram_style = self.get_nodeattr("ram_style")
        if ram_style == "auto":
            code_gen_dict["$RAM_STYLE$"] = [""]
        else:
            code_gen_dict["$RAM_STYLE$"] = ['(* ram_style = "{}" *)'.format(ram_style)]

        # apply code generation to templates
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        with open(template_path, "r") as f:
            template = f.read()
        with open(
            os.environ["FINN_ROOT"] + "/finn-rtllib/swg/swg_template_wrapper.v", "r"
        ) as f:
            template_wrapper = f.read()
        for key in code_gen_dict:
            # transform list into long string separated by '\n'
            code_gen_line = "\n".join(code_gen_dict[key])
            template = template.replace(key, code_gen_line)
            template_wrapper = template_wrapper.replace(key, code_gen_line)
        with open(
            os.path.join(
                code_gen_dir, self.get_nodeattr("gen_top_module") + "_impl.sv"
            ),
            "w",
        ) as f:
            f.write(template)
        with open(
            os.path.join(
                code_gen_dir, self.get_nodeattr("gen_top_module") + "_wrapper.v"
            ),
            "w",
        ) as f:
            f.write(template_wrapper)

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
            self.get_nodeattr("gen_top_module") + "_wrapper.v",
            self.get_nodeattr("gen_top_module") + "_impl.sv",
        ]

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

        cmd = [
            "add_files -norecurse %s"
            % (
                os.path.join(
                    code_gen_dir, self.get_nodeattr("gen_top_module") + "_wrapper.v"
                )
            ),
            "add_files -norecurse %s"
            % (
                os.path.join(
                    code_gen_dir, self.get_nodeattr("gen_top_module") + "_impl.sv"
                )
            ),
            "create_bd_cell -type module -reference %s %s"
            % (self.get_nodeattr("gen_top_module"), self.onnx_node.name),
        ]

        return cmd

    def code_generation_ipgen(self, model, fpgapart, clk):
        """Normally: Generates C++ code and tcl script for IP generation.
        Here: Generates (System-)Verilog code for IP generation."""
        self.generate_hdl()

    def ipgen_singlenode_code(self):
        """Normally: Builds the bash script for IP generation."""
        pass

    def code_generation_cppsim(self, model):
        """Normally: Generates C++ code for simulation (cppsim)."""
        pass

    def compile_singlenode_code(self):
        pass

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
