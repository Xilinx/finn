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
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class Pool(HWCustomOp):
    """Abstraction layer for HW implementation of Pool.
    Requires ConvolutionInputGenerator(depthwise == 1) to format its input

    Input shape (BatchSize,OutImgDim,OutImgDim,TotalKernelSize*Channels)
    Output shape (BatchSize,OutImgDim,OutImgDim,Channels)

    Notes:

    * The input shape was chosen to be compatible with im2col (only true when there
      is not folding).
    * The actual data layout produced by the hlslib kernels is different
      for depthwise ops.

        * depthwise SWG: (1, OFMDim, OFMDim, IFMChannels/PE, K, K, PE)

    Channels can be folded using PE (SIMD from the input perspective)
    """

    def get_nodeattr_types(self):
        my_attrs = {
            "Channels": ("i", True, 0),
            "PE": ("i", True, 1),
            "KernelSize": ("ints", True, []),
            # Function:
            #  - MaxPool
            #  - QuantAvgPool
            # TODO add support for AvgPool and AccPool
            "Function": ("s", True, "", {"MaxPool", "QuantAvgPool"}),
            "OutImgDims": ("ints", True, []),
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
        odims = self.get_nodeattr("OutImgDims")
        batch_size = self.get_nodeattr("BatchSize")
        k = self.get_nodeattr("KernelSize")
        k_prod = int(np.prod(k))
        ishape = (batch_size, *odims, k_prod * ifm_ch)
        return ishape

    def get_folded_input_shape(self, ind=0):
        normal_ishape = list(self.get_normal_input_shape())
        ifm_ch = self.get_nodeattr("Channels")
        pe = self.get_nodeattr("PE")
        assert ifm_ch % pe == 0, "PE must divide input channels"
        fold = int(normal_ishape[-1] / pe)
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

    def get_number_output_values(self):
        folded_oshape = self.get_folded_output_shape()
        return np.prod(folded_oshape[1:-1])

    def get_exp_cycles(self):
        # (Channels * kernel * kernel) / PE * odim * odim * batch_size
        ifm_ch = self.get_nodeattr("Channels")
        pe = self.get_nodeattr("PE")
        k = self.get_nodeattr("KernelSize")
        k_prod = int(np.prod(k))
        odims = self.get_nodeattr("OutImgDims")
        batch_size = self.get_nodeattr("BatchSize")
        exp_cycles = ((ifm_ch * k_prod) / pe) * np.prod(odims) * batch_size
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
        # data type stays the same
        dtype = self.get_output_datatype()
        model.set_tensor_datatype(node.output[0], dtype)

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
        k = self.get_nodeattr("KernelSize")
        ch = self.get_nodeattr("Channels")
        k2 = k[0] * k[1]

        inp_values = context[node.input[0]]
        ishape = inp_values.shape
        # reshape array to apply max or avg function only on kernel
        tmp_shape = tuple(list(ishape)[:-1] + [k2, ch])
        tmp_values = inp_values.reshape(tmp_shape)
        if fnx == "MaxPool":
            result = np.max(tmp_values, axis=3)
        elif fnx == "QuantAvgPool":
            # determine bits to shift
            ibits = self.get_input_datatype().bitwidth()
            obits = self.get_output_datatype().bitwidth()
            max_value = 2**ibits - 1
            max_value = max_value * k2
            max_bit_width = int(max_value).bit_length()
            shift_bits = max_bit_width - obits
            shift_bits = shift_bits if shift_bits >= 0 else 0
            result = np.sum(tmp_values, axis=3)
            result = np.right_shift(result.astype(int), shift_bits)
        oshape = context[node.output[0]].shape
        context[node.output[0]] = np.asarray(result, dtype=np.float32).reshape(oshape)
