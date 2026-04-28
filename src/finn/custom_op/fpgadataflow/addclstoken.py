# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
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


class AddCLSToken(HWCustomOp):
    """Prepend a learned class token to a sequence of patch tokens."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = super().get_nodeattr_types()
        my_attrs.update(
            {
                "NumTokens": ("i", True, 0),
                "NumChannels": ("i", True, 0),
                "PadTokens": ("i", False, 0),
                "SIMD": ("i", False, 1),
                "inputDataType": ("s", True, ""),
                "outputDataType": ("s", False, ""),
            }
        )
        return my_attrs

    def get_normal_input_shape(self, ind=0):
        num_channels = self.get_nodeattr("NumChannels")
        if ind == 0:
            return (1, self.get_nodeattr("NumTokens"), num_channels)
        elif ind == 1:
            return (1, 1, num_channels)
        else:
            raise Exception("AddCLSToken only has two inputs")

    def get_folded_input_shape(self, ind=0):
        normal_shape = self.get_normal_input_shape(ind)
        simd = self.get_nodeattr("SIMD")
        num_channels = normal_shape[-1]
        assert num_channels % simd == 0, "SIMD must divide NumChannels"
        return normal_shape[:-1] + (num_channels // simd, simd)

    def get_normal_output_shape(self, ind=0):
        num_tokens = self.get_nodeattr("NumTokens")
        num_channels = self.get_nodeattr("NumChannels")
        pad_tokens = self.get_nodeattr("PadTokens")
        return (1, num_tokens + 1 + pad_tokens, num_channels)

    def get_folded_output_shape(self, ind=0):
        normal_shape = self.get_normal_output_shape(ind)
        simd = self.get_nodeattr("SIMD")
        num_channels = normal_shape[-1]
        assert num_channels % simd == 0, "SIMD must divide NumChannels"
        return normal_shape[:-1] + (num_channels // simd, simd)

    def make_shape_compatible_op(self, model):
        exp_ishape = self.get_normal_input_shape(0)
        ishape = tuple(model.get_tensor_shape(self.onnx_node.input[0]))
        assert ishape == exp_ishape, "Unexpected input shape for patch tokens."

        exp_wshape = self.get_normal_input_shape(1)
        wshape = tuple(model.get_tensor_shape(self.onnx_node.input[1]))
        assert wshape == exp_wshape, "Unexpected input shape for CLS token."

        return super().make_const_shape_op(self.get_normal_output_shape())

    def infer_node_datatype(self, model):
        node = self.onnx_node
        attr_idt = None
        if self.get_nodeattr("inputDataType") != "":
            attr_idt = self.get_input_datatype()

        idt = model.get_tensor_datatype(node.input[0])
        if idt is None:
            idt = attr_idt
        if idt is None:
            raise Exception("AddCLSToken input datatype is not set")

        if attr_idt is not None and attr_idt != idt:
            warnings.warn(
                "inputDataType changing for %s: %s -> %s" % (node.name, str(attr_idt), str(idt))
            )
        self.set_nodeattr("inputDataType", idt.name)

        cls_dt = model.get_tensor_datatype(node.input[1])
        if cls_dt is None:
            model.set_tensor_datatype(node.input[1], idt)
        else:
            assert cls_dt == idt, "CLS token datatype must match input datatype."

        self.set_nodeattr("outputDataType", idt.name)
        model.set_tensor_datatype(node.output[0], idt)

    def verify_node(self):
        pass

    def get_input_datatype(self, ind=0):
        return DataType[self.get_nodeattr("inputDataType")]

    def get_output_datatype(self, ind=0):
        odt = self.get_nodeattr("outputDataType")
        if odt == "":
            return self.get_input_datatype(ind)
        return DataType[odt]

    def get_instream_width(self, ind=0):
        if ind != 0:
            return 0
        return self.get_input_datatype().bitwidth() * self.get_nodeattr("SIMD")

    def get_outstream_width(self, ind=0):
        return self.get_output_datatype().bitwidth() * self.get_nodeattr("SIMD")

    def get_number_output_values(self):
        return int(np.prod(self.get_folded_output_shape()[:-1]))

    def get_exp_cycles(self):
        return int(np.prod(self.get_folded_output_shape()[:-1]))

    def execute_node(self, context, graph):
        node = self.onnx_node
        patches = context[node.input[0]]
        cls_token = context[node.input[1]]

        result = np.concatenate([cls_token, patches], axis=1)
        pad_tokens = self.get_nodeattr("PadTokens")
        if pad_tokens > 0:
            pad_shape = (1, pad_tokens, self.get_nodeattr("NumChannels"))
            padding = np.zeros(pad_shape, dtype=result.dtype)
            result = np.concatenate([result, padding], axis=1)

        oshape = self.get_normal_output_shape()
        context[node.output[0]] = np.asarray(result, dtype=np.float32).reshape(oshape)

    def bram_estimation(self):
        return 0

    def lut_estimation(self):
        return int(128 + self.get_nodeattr("NumChannels"))

    def get_op_and_param_counts(self):
        return {"param_cls_token": int(self.get_nodeattr("NumChannels"))}
