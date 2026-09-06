# Copyright (C) 2023, Advanced Micro Devices, Inc.
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


class FMPadding(HWCustomOp):
    """Abstraction layer for HW impplementation of FMPadding.
    Pads input image by given amount."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            # spatial size of input images
            "ImgDim": ("ints", True, []),  # [H, W] = [Y, X]
            # total padding (per dimension) to apply
            "Padding": (
                "ints",
                True,
                [1, 1, 1, 1],
            ),  # [H_begin, W_begin, H_end, W_end] = [Y_begin, X_begin, Y_end, X_end]
            # number of channels in input image
            "NumChannels": ("i", True, 0),
            # SIMD Input parallelism
            "SIMD": ("i", False, 1),
            # FINN input datatype
            "inputDataType": ("s", True, ""),
            # shape describing input vecs per execution
            "numInputVectors": ("i", False, 1),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_padded_odim(self):
        "Return the padded spatial size of the output."
        idim_h, idim_w = self.get_nodeattr("ImgDim")
        pad = self.get_nodeattr("Padding")
        pad_h = pad[0] + pad[2]
        pad_w = pad[1] + pad[3]
        odim_h = idim_h + pad_h
        odim_w = idim_w + pad_w
        return [odim_h, odim_w]

    def get_exp_cycles(self):
        odim_h, odim_w = self.get_padded_odim()
        channels = self.get_nodeattr("NumChannels")
        simd = self.get_nodeattr("SIMD")
        batch_size = self.get_nodeattr("numInputVectors")
        exp_cycles = (channels / simd) * batch_size * odim_h * odim_w
        return int(exp_cycles)

    def get_normal_input_shape(self, ind=0):
        idim_h, idim_w = self.get_nodeattr("ImgDim")
        num_ch = self.get_nodeattr("NumChannels")
        ishape = (1, idim_h, idim_w, num_ch)
        return ishape

    def get_normal_output_shape(self, ind=0):
        odim_h, odim_w = self.get_padded_odim()
        num_ch = self.get_nodeattr("NumChannels")

        oshape = (1, odim_h, odim_w, num_ch)
        return oshape

    def get_folded_input_shape(self, ind=0):
        normal_ishape = list(self.get_normal_input_shape())
        ifm_ch = self.get_nodeattr("NumChannels")
        simd = self.get_nodeattr("SIMD")
        assert ifm_ch % simd == 0, "SIMD must divide input channels"
        fold = int(normal_ishape[-1] / simd)
        folded_ishape = normal_ishape[:-1] + [fold, simd]
        return tuple(folded_ishape)

    def get_folded_output_shape(self, ind=0):
        normal_oshape = list(self.get_normal_output_shape())
        ifm_ch = self.get_nodeattr("NumChannels")
        simd = self.get_nodeattr("SIMD")
        assert ifm_ch % simd == 0, "SIMD must divide input channels"
        fold = int(normal_oshape[-1] / simd)
        folded_oshape = normal_oshape[:-1] + [fold, simd]
        return tuple(folded_oshape)

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
        model.set_tensor_datatype(node.output[0], idt)

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        ret = DataType[self.get_nodeattr("inputDataType")]
        # the hlslib op always pads with zeros, so ensure that the DataType
        # is able to represent zeros
        assert ret.allowed(0), "FMPadding_Batch DataType must support zero"
        return ret

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output. (Same as input datatype)"""
        return self.get_input_datatype()

    def get_instream_width(self, ind=0):
        ibits = self.get_input_datatype().bitwidth()
        simd = self.get_nodeattr("SIMD")
        return ibits * simd

    def get_outstream_width(self, ind=0):
        obits = self.get_output_datatype().bitwidth()
        simd = self.get_nodeattr("SIMD")
        return obits * simd

    def execute_node(self, context, graph):
        # simulate behavior with Python functionality
        node = self.onnx_node
        pad = self.get_nodeattr("Padding")
        inp_values = context[node.input[0]]
        oshape = context[node.output[0]].shape
        result = np.pad(
            inp_values, ((0, 0), (pad[0], pad[2]), (pad[1], pad[3]), (0, 0)), "constant"
        )
        context[node.output[0]] = np.asarray(result, dtype=np.float32).reshape(oshape)
