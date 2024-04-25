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


class Deconvolution(HWCustomOp):
    """Abstraction layer for HW implementation of Deconvolution"""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            "KernelDim": ("ints", True, []),  # [H, W] = [Y, X]
            "IFMChannels": ("i", True, 0),
            "OFMChannels": ("i", True, 0),
            "IFMDim": ("ints", True, []),  # [H, W] = [Y, X]
            "PE": ("i", True, 0),
            "SIMD": ("i", True, 0),
            "Stride": ("ints", True, [1, 1]),  # [H, W] = [Y, X]
            "Padding": ("ints", True, []),  # [H, W] = [Y, X]
            # FINN DataTypes for inputs, weights, outputs
            "inputDataType": ("s", True, ""),
            "weightDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_normal_input_shape(self, ind=0):
        ifm_dim_h, ifm_dim_w = self.get_nodeattr("IFMDim")
        ifm_ch = self.get_nodeattr("IFMChannels")
        ishape = (1, ifm_dim_h, ifm_dim_w, ifm_ch)
        return ishape

    def get_folded_input_shape(self, ind=0):
        ifm_dim_h, ifm_dim_w = self.get_nodeattr("IFMDim")
        ifm_ch = self.get_nodeattr("IFMChannels")
        simd = self.get_nodeattr("SIMD")
        assert ifm_ch % simd == 0, "SIMD must divide IFMChannels"
        fold = int(ifm_ch / simd)
        folded_ishape = (1, ifm_dim_h, ifm_dim_w, fold, simd)
        return folded_ishape

    def get_normal_output_shape(self, ind=0):
        idim_h, idim_w = self.get_nodeattr("IFMDim")
        stride_h, stride_w = self.get_nodeattr("Stride")
        k_h, k_w = self.get_nodeattr("KernelDim")
        ofm_ch = self.get_nodeattr("OFMChannels")
        pad_h, pad_w = self.get_nodeattr("Padding")
        odim_h = (idim_h - 1) * stride_h - 2 * pad_h + (k_h - 1) + 1
        odim_w = (idim_w - 1) * stride_w - 2 * pad_w + (k_w - 1) + 1
        oshape = (1, odim_h, odim_w, ofm_ch)
        return oshape

    def get_folded_output_shape(self, ind=0):
        normal_oshape = self.get_normal_output_shape()
        odim_h = normal_oshape[1]
        odim_w = normal_oshape[2]
        ofm_ch = normal_oshape[3]
        pe = self.get_nodeattr("PE")
        fold = int(ofm_ch / pe)
        folded_oshape = (1, odim_h, odim_w, fold, pe)
        return folded_oshape

    def make_shape_compatible_op(self, model):
        exp_ishape = self.get_normal_input_shape()
        oshape = self.get_normal_output_shape()
        ishape = tuple(model.get_tensor_shape(self.onnx_node.input[0]))
        assert ishape == exp_ishape, "Unexpected input shape for Deconv."
        # implement tensor with correct shape
        return super().make_const_shape_op(oshape)

    def infer_node_datatype(self, model):
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            warn_str = "inputDataType changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype()),
                str(idt),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("inputDataType", idt.name)
        # set output datatype from property
        odt = self.get_output_datatype()
        model.set_tensor_datatype(node.output[0], odt)

    def verify_node(self):
        pass

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        return DataType[self.get_nodeattr("inputDataType")]

    def get_weight_datatype(self):
        """Returns FINN DataType of weights."""
        return DataType[self.get_nodeattr("weightDataType")]

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output."""
        return DataType[self.get_nodeattr("outputDataType")]

    def get_instream_width(self, ind=0):
        """Returns stream width, input and output stream width are equal for
        the sliding window function"""
        ibits = self.get_input_datatype().bitwidth()
        simd = self.get_nodeattr("SIMD")
        ifm_ch = self.get_nodeattr("IFMChannels")
        assert ifm_ch % simd == 0, "SIMD must divide IFMChannels"
        in_width = simd * ibits
        return in_width

    def get_outstream_width(self, ind=0):
        o_bits = self.get_output_datatype().bitwidth()
        out_width = o_bits * self.get_nodeattr("PE")
        return out_width

    def get_number_output_values(self):
        folded_oshape = self.get_folded_output_shape()
        num_output_elems = np.prod(folded_oshape[:-1])
        return num_output_elems

    def get_exp_cycles(self):
        return 0

    def bram_estimation(self):
        return 0

    def lut_estimation(self):
        return 0

    def uram_estimation(self):
        return 0

    def execute_node(self, context, graph):
        pass
