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
from onnx import helper
from qonnx.core.datatype import DataType
from qonnx.util.basic import roundup_to_integer_multiple

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class StreamingSplit(HWCustomOp):
    """Abstraction layer for HW implementation of Split.
    Only supports splitting along the last (channel) axis."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            "SIMD": ("i", True, 0),
            # number of elements of each output streams
            "ChannelsPerStream": ("ints", True, []),
            # FINN DataTypes for input; output datatypes inferred from input
            "inputDataType": ("s", True, ""),
            # number of input vectors for non-split axes, examples:
            # [1] is a single vector (like a FC layer with batch=1)
            # [4] is four vectors (like a FC layer with batch=4)
            # [1, 4, 4] is four * four vectors (like a conv layer with batch=1)
            "numInputVectors": ("ints", False, [1]),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_n_outputs(self):
        return len(self.get_nodeattr("ChannelsPerStream"))

    def get_total_elems(self):
        elems_per_stream = self.get_nodeattr("ChannelsPerStream")
        return int(np.sum(elems_per_stream))

    def get_normal_input_shape(self, ind=0):
        total_elems = self.get_total_elems()
        vecs = list(self.get_nodeattr("numInputVectors"))
        ishape = tuple(vecs + [total_elems])
        return ishape

    def get_folded_input_shape(self, ind=0):
        simd = self.get_nodeattr("SIMD")
        folds = self.get_total_elems() // simd
        vecs = list(self.get_nodeattr("numInputVectors"))
        return tuple(vecs + [folds, simd])

    def get_normal_output_shape(self, ind=0):
        elems = self.get_nodeattr("ChannelsPerStream")[ind]
        vecs = list(self.get_nodeattr("numInputVectors"))
        return tuple(vecs + [elems])

    def get_folded_output_shape(self, ind=0):
        elems = self.get_nodeattr("ChannelsPerStream")[ind]
        simd = self.get_nodeattr("SIMD")
        folds = elems // simd
        vecs = list(self.get_nodeattr("numInputVectors"))
        return tuple(vecs + [folds, simd])

    def make_shape_compatible_op(self, model):
        # check input shape
        exp_ishape = self.get_normal_input_shape()
        ishape = tuple(model.get_tensor_shape(self.onnx_node.input[0]))
        assert ishape == exp_ishape, "Unexpected input shape"

        assert len(self.onnx_node.output) == self.get_n_outputs(), "Unexpected number of outputs"
        ret = helper.make_node("Split", self.onnx_node.input, self.onnx_node.output, axis=-1)
        return ret

    def infer_node_datatype(self, model):
        # check input datatype
        inp = self.onnx_node.input[0]
        idt = model.get_tensor_datatype(inp)
        if idt != self.get_input_datatype():
            warn_str = "inputDataType changing for %s: %s -> %s " % (
                self.onnx_node.name,
                str(self.get_input_datatype()),
                str(idt),
            )
            warnings.warn(warn_str)
            self.set_nodeattr("inputDataType", idt.name)
        odt = self.get_output_datatype()
        for out in self.onnx_node.output:
            model.set_tensor_datatype(out, odt)

    def verify_node(self):
        pass

    def get_input_datatype(self, ind=0):
        return DataType[self.get_nodeattr("inputDataType")]

    def get_output_datatype(self, ind=0):
        # all output datatypes are the same as the input datatype
        return self.get_input_datatype()

    def get_instream_width(self, ind=0):
        ibits = self.get_input_datatype().bitwidth()
        return ibits * self.get_nodeattr("SIMD")

    def get_outstream_width(self, ind=0):
        obits = self.get_output_datatype().bitwidth()
        out_width = obits * self.get_nodeattr("SIMD")
        return out_width

    def get_number_output_values(self):
        out_val = {}
        for i in range(len(self.onnx_node.output)):
            out_val["out_%s" % i] = np.prod(self.get_folded_output_shape(i)[1:-1])
        return out_val

    def get_exp_cycles(self):
        return np.prod(self.get_folded_input_shape()[:-1])

    def execute_node(self, context, graph):
        node = self.onnx_node
        split = self.get_nodeattr("ChannelsPerStream")
        np_split_param = np.cumsum(split[:-1])
        np_result = np.split(context[node.input[0]], np_split_param, axis=-1)
        for i, out in enumerate(node.output):
            context[out] = np_result[i]

    def get_instream_width_padded(self, ind=0):
        in_width = self.get_instream_width()
        return roundup_to_integer_multiple(in_width, 8)

    def get_verilog_top_module_intf_names(self):
        intf_names = super().get_verilog_top_module_intf_names()
        intf_names["s_axis"] = [("in0", self.get_instream_width_padded())]
        intf_names["m_axis"] = []
        for i in range(self.get_n_outputs()):
            intf_names["m_axis"].append(("out_%d" % i, self.get_instream_width_padded()))
        return intf_names
