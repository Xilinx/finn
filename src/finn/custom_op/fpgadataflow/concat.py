# Copyright (c) 2021, Xilinx
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

import math
import numpy as np
import warnings
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class StreamingConcat(HWCustomOp):
    """Abstraction layer for HW implementation of Concat.
    Only supports concatenating along the last (channel) axis."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            "SIMD": ("i", True, 0),
            # number of elements from each stream to concat
            "ChannelsPerStream": ("ints", True, []),
            # FINN DataTypes for inputs; output datatype inferred from inputs
            "inputDataTypes": ("strings", True, [""]),
            # number of input vectors for non-concat axes, examples:
            # [1] is a single vector (like a FC layer with batch=1)
            # [4] is four vectors (like a FC layer with batch=4)
            # [1, 4, 4] is four * four vectors (like a conv layer with batch=1)
            "numInputVectors": ("ints", False, [1]),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_n_inputs(self):
        return len(self.get_nodeattr("ChannelsPerStream"))

    def get_total_elems(self):
        elems_per_stream = self.get_nodeattr("ChannelsPerStream")
        return int(np.sum(elems_per_stream))

    def get_normal_input_shape(self, ind=0):
        elems_per_stream = self.get_nodeattr("ChannelsPerStream")
        elems = elems_per_stream[ind]
        vecs = list(self.get_nodeattr("numInputVectors"))
        ishape = tuple(vecs + [elems])
        return ishape

    def get_folded_input_shape(self, ind=0):
        simd = self.get_nodeattr("SIMD")
        folds = self.get_nodeattr("ChannelsPerStream")[ind] // simd
        vecs = list(self.get_nodeattr("numInputVectors"))
        return tuple(vecs + [folds, simd])

    def get_normal_output_shape(self, ind=0):
        total_elems = self.get_total_elems()
        vecs = list(self.get_nodeattr("numInputVectors"))
        return tuple(vecs + [total_elems])

    def get_folded_output_shape(self, ind=0):
        total_elems = self.get_total_elems()
        simd = self.get_nodeattr("SIMD")
        folds = total_elems // simd
        vecs = list(self.get_nodeattr("numInputVectors"))
        return tuple(vecs + [folds, simd])

    def infer_node_datatype(self, model):
        # check all input datatypes
        for i, inp in enumerate(self.onnx_node.input):
            idt = model.get_tensor_datatype(inp)
            if idt != self.get_input_datatype(i):
                warn_str = "inputDataType changing for %s: %s -> %s " % (
                    self.onnx_node.name,
                    str(self.get_input_datatype(i)),
                    str(idt),
                )
                warnings.warn(warn_str)
                old_datatypes_attr = self.get_nodeattr("inputDataTypes")
                old_datatypes_attr[i] = idt.name
                self.set_nodeattr("inputDataTypes", old_datatypes_attr)
        odt = self.get_output_datatype()
        model.set_tensor_datatype(self.onnx_node.output[0], odt)

    def get_input_datatype(self, ind=0):
        # input dt identical for all inputs
        return DataType[self.get_nodeattr("inputDataTypes")[ind]]

    def get_output_datatype(self, ind=0):
        # infer output datatype from declared inputDataTypes
        min_input = 0
        max_input = 0
        for i in range(len(self.get_nodeattr("inputDataTypes"))):
            idt = self.get_input_datatype(i)
            if idt.min() < min_input:
                min_input = idt.min()
            if idt.max() > max_input:
                max_input = idt.max()
        # if the input range is always greater than 0, then acc_max <= 2^P - 1
        if min_input >= 0:
            out_bit_width = math.ceil(np.log2(max_input + 1))
            odt = DataType[f"UINT{out_bit_width}"]
        # if the input range is signed, then acc_min >= -2^{P-1} and acc_max <=
        # 2^{P - 1} - 1, which means 2^{P - 1} >= max(-acc_min, 1 + acc_max)
        else:
            max_abs_input = max(-min_input, 1 + max_input)
            out_bit_width = math.ceil(np.log2(max_abs_input) + 1)
            odt = DataType[f"INT{out_bit_width}"]
        return odt

    def get_instream_width(self, ind=0):
        ibits = self.get_input_datatype(ind).bitwidth()
        return ibits * self.get_nodeattr("SIMD")

    def get_outstream_width(self, ind=0):
        obits = self.get_output_datatype().bitwidth()
        out_width = obits * self.get_nodeattr("SIMD")
        return out_width

    def get_exp_cycles(self):
        return np.prod(self.get_folded_output_shape()[:-1])

    def execute_node(self, context, graph):
        node = self.onnx_node
        inp_values = []
        for inp in node.input:
            inp_values.append(context[inp])
        result = np.concatenate(inp_values, axis=-1)
        context[node.output[0]] = result
