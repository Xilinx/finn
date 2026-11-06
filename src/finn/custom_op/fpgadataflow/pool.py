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

import numpy as np
import warnings
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class Pool(HWCustomOp):
    """Abstraction layer for HW implementation of Pool.

    Consolidated pooling node that fuses sliding-window generation (input_gen),
    optional padding (pad), and the pooling reduction (Pool_batch) into a single
    HLS kernel.

    Input shape (BatchSize, IFMDim_H, IFMDim_W, Channels)  — raw feature map
    Output shape (BatchSize, OutImgDim_H, OutImgDim_W, Channels)

    Channels can be folded using PE (SIMD from the input perspective).
    """

    def get_nodeattr_types(self):
        my_attrs = {
            "Channels": ("i", True, 0),
            "PE": ("i", True, 1),
            "KernelSize": ("ints", True, []),
            # Function:
            #  - MaxPool
            #  - QuantAvgPool
            "Function": ("s", True, "", {"MaxPool", "QuantAvgPool"}),
            # Input feature map spatial dimensions [H, W]
            "IFMDim": ("ints", True, []),
            # Output feature map spatial dimensions [H, W]
            "OutImgDims": ("ints", True, []),
            # Stride [SH, SW]
            "Stride": ("ints", True, []),
            # Padding [H_begin, W_begin, H_end, W_end]
            "Padding": ("ints", False, [0, 0, 0, 0]),
            # Pad value (datatype min for MaxPool, 0 otherwise)
            "PadValue": ("i", False, 0),
            # FINN DataTypes for inputs/outputs
            "InputDataType": ("s", True, ""),
            "OutputDataType": ("s", True, ""),
            "AccumBits": ("i", False, 0),
            "Size": ("i", False, 1),
            "BatchSize": ("i", False, 1),
        }

        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        return DataType[self.get_nodeattr("InputDataType")]

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output."""
        fxn = self.get_nodeattr("Function")
        odt = DataType[self.get_nodeattr("OutputDataType")]

        if fxn == "MaxPool":
            # Same as input
            idt = DataType[self.get_nodeattr("InputDataType")]
            assert odt == idt, "In datatype must be equal to out datatype for Maxpool"
        elif fxn == "QuantAvgPool":
            idt = DataType[self.get_nodeattr("InputDataType")]
            assert (
                idt.signed() == odt.signed()
            ), """QuantAvgPool: Can't mix signed
            and unsigned datatypes"""
        else:
            raise Exception("Pool_Batch doesn't currently support " + fxn)

        return odt

    def get_normal_input_shape(self, ind=0):
        ifm_ch = self.get_nodeattr("Channels")
        ifm_dims = self.get_nodeattr("IFMDim")
        batch_size = self.get_nodeattr("BatchSize")
        ishape = (batch_size, *ifm_dims, ifm_ch)
        return ishape

    def get_folded_input_shape(self, ind=0):
        normal_ishape = list(self.get_normal_input_shape())
        ifm_ch = self.get_nodeattr("Channels")
        pe = self.get_nodeattr("PE")
        assert ifm_ch % pe == 0, "PE must divide input channels"
        fold = int(ifm_ch / pe)
        folded_ishape = normal_ishape[:-1] + [fold, pe]
        return tuple(folded_ishape)

    def get_normal_output_shape(self, ind=0):
        ofm_ch = self.get_nodeattr("Channels")
        odims = self.get_nodeattr("OutImgDims")
        batch_size = self.get_nodeattr("BatchSize")
        oshape = (batch_size, *odims, ofm_ch)
        return oshape

    def get_folded_output_shape(self, ind=0):
        normal_oshape = list(self.get_normal_output_shape())
        ifm_ch = self.get_nodeattr("Channels")
        pe = self.get_nodeattr("PE")
        assert ifm_ch % pe == 0, "PE must divide input channels"
        fold = int(ifm_ch / pe)
        folded_oshape = normal_oshape[:-1] + [fold, pe]
        return tuple(folded_oshape)

    def get_exp_cycles(self):
        # Fused kernel: [pad] -> input_gen -> Pool_batch in HLS dataflow.
        #
        # input_gen reads elements from a circular buffer driven by a Nest<>
        # loop nest.  Output i reads buffer address addrs[i].  Data at
        # address a becomes available WP_DELAY+1 cycles after it enters the
        # buffer.  When pad<> precedes input_gen, an additional PAD_LATENCY
        # accounts for the HLS dataflow pipeline between the two stages.
        #
        # Output i cannot be produced before cycle:
        #   addrs[i] + WP_DELAY + 1 + PAD_LATENCY
        # and outputs are serialised at 1/cycle, so:
        #   earliest[i] = max(earliest[i-1]+1, addrs[i] + base_delay)
        #
        # The last output cycle is:
        #   earliest[last] = max_slack + base_delay + total_out - 1
        # where max_slack = max over i of (addrs[i] - i).
        #
        # For the pooling loop nest (oh, ow, cf, kh, kw), max_slack is at
        # (oh=0, ow=0, cf=0, kh=KH-1, kw=KW-1):
        #   max_slack = (KH-1)*(W*CF - KW) + (KW-1)*(CF-1)
        #
        # Total = max_slack + base_delay + total_out - 1 + PIPELINE_FLUSH
        ifm_ch = self.get_nodeattr("Channels")
        pe = self.get_nodeattr("PE")
        cf = ifm_ch // pe
        k_h, k_w = self.get_nodeattr("KernelSize")
        ifm_dim_h, ifm_dim_w = self.get_nodeattr("IFMDim")
        ofm_dim_h, ofm_dim_w = self.get_nodeattr("OutImgDims")
        pad = self.get_nodeattr("Padding")
        batch_size = self.get_nodeattr("BatchSize")
        has_padding = any(p != 0 for p in pad)

        # Padded spatial dimensions (what input_gen sees)
        w_padded = ifm_dim_w + pad[1] + pad[3]

        # Total output elements from input_gen
        total_out = ofm_dim_h * ofm_dim_w * k_h * k_w * cf

        # Max data-dependency slack in the address sequence
        max_slack = (k_h - 1) * (w_padded * cf - k_w) + (k_w - 1) * (cf - 1)

        # HLS pipeline constants
        WP_DELAY = 4
        PIPELINE_FLUSH = 3
        PAD_LATENCY = 4 if has_padding else 0

        base_delay = WP_DELAY + 1 + PAD_LATENCY
        exp_cycles = max_slack + base_delay + total_out - 1 + PIPELINE_FLUSH
        exp_cycles *= batch_size
        return int(exp_cycles)

    def get_instream_width(self, ind=0):
        dt_bits = self.get_input_datatype().bitwidth()
        pe = self.get_nodeattr("PE")
        in_width = int(dt_bits * pe)
        return in_width

    def get_outstream_width(self, ind=0):
        dt_bits = self.get_output_datatype().bitwidth()
        pe = self.get_nodeattr("PE")
        out_width = int(dt_bits * pe)
        return out_width

    def infer_node_datatype(self, model):
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            warn_str = "InputDataType changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype()),
                str(idt),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("InputDataType", idt.name)
        if self.get_nodeattr("Function") == "MaxPool":
            # output = input by definition for MaxPool
            self.set_nodeattr("OutputDataType", idt.name)
        # set output tensor annotation from property
        odt = self.get_output_datatype()
        model.set_tensor_datatype(node.output[0], odt)

    def minimize_accumulator_width(self, model, datatype_only=False):
        """Minimize accumulator width for QuantAvgPool.

        Parameters
        ----------
        datatype_only : bool
            Not used - Pool has no weights, always uses datatype bounds.
        """
        fxn = self.get_nodeattr("Function")
        if fxn != "QuantAvgPool":
            return  # MaxPool doesn't need accumulator

        idt = self.get_input_datatype()
        k = self.get_nodeattr("KernelSize")
        kernel_elements = np.prod(k)

        # Compute worst-case accumulator range
        if idt.signed():
            acc_min = kernel_elements * idt.min()
            acc_max = kernel_elements * idt.max()
        else:
            acc_min = 0
            acc_max = kernel_elements * idt.max()

        # Get smallest datatype that fits
        if acc_min < 0:
            adt = DataType.get_smallest_possible(min(acc_min, -acc_max - 1))
        else:
            adt = DataType.get_smallest_possible(acc_max)

        self.set_nodeattr("AccumBits", adt.bitwidth())
        return adt

    def verify_node(self):
        info_messages = []
        # verify that "backend" is set to "fpgadataflow"
        backend_value = self.get_nodeattr("backend")
        if backend_value == "fpgadataflow":
            info_messages.append("Attribute backend is set correctly")
        else:
            info_messages.append('Attribute backend should be set to "fpgadataflow"')

        # verify the number of inputs
        if len(self.onnx_node.input) == 1:
            info_messages.append("The number of inputs is correct")
        else:
            info_messages.append("""Pool_Batch needs 1 data input""")

        # check supported function
        fnx = self.get_nodeattr("Function")
        if fnx in ["MaxPool", "QuantAvgPool"]:
            info_messages.append("Attribute Function contains a supported pool function")
        else:
            info_messages.append("Attribute Function contains an unsupported pool function")
        return info_messages

    def execute_node(self, context, graph):
        # simulate behavior with Python functionality
        node = self.onnx_node
        fnx = self.get_nodeattr("Function")
        kh, kw = self.get_nodeattr("KernelSize")
        sh, sw = self.get_nodeattr("Stride")
        ch = self.get_nodeattr("Channels")
        pad = self.get_nodeattr("Padding")
        pad_value = self.get_nodeattr("PadValue")
        k2 = kh * kw

        inp_values = context[node.input[0]]
        # inp_values shape: (batch, IFM_H, IFM_W, Channels)

        # Apply padding if needed
        if any(p != 0 for p in pad):
            inp_values = np.pad(
                inp_values,
                ((0, 0), (pad[0], pad[2]), (pad[1], pad[3]), (0, 0)),
                mode="constant",
                constant_values=pad_value,
            )

        batch, ph, pw, _ = inp_values.shape
        ofm_h, ofm_w = self.get_nodeattr("OutImgDims")

        # Extract sliding windows and reduce
        # result shape: (batch, ofm_h, ofm_w, ch)
        result = np.zeros((batch, ofm_h, ofm_w, ch), dtype=inp_values.dtype)
        for b in range(batch):
            for oh in range(ofm_h):
                for ow in range(ofm_w):
                    window = inp_values[b, oh * sh : oh * sh + kh, ow * sw : ow * sw + kw, :]
                    if fnx == "MaxPool":
                        result[b, oh, ow, :] = np.max(window, axis=(0, 1))
                    elif fnx == "QuantAvgPool":
                        ibits = self.get_input_datatype().bitwidth()
                        obits = self.get_output_datatype().bitwidth()
                        max_value = (2**ibits - 1) * k2
                        max_bit_width = int(max_value).bit_length()
                        shift_bits = max(0, max_bit_width - obits)
                        acc = np.sum(window, axis=(0, 1))
                        result[b, oh, ow, :] = np.right_shift(acc.astype(int), shift_bits)

        oshape = context[node.output[0]].shape
        context[node.output[0]] = np.asarray(result, dtype=np.float32).reshape(oshape)
