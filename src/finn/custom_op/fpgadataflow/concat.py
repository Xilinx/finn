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

import numpy as np
from qonnx.core.datatype import DataType
from qonnx.util.basic import roundup_to_integer_multiple

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class StreamingConcat(HWCustomOp):
    """Abstraction layer for HW implementation of Concat.
    Only supports concatenating along the last axis."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            # number of elements from each stream to concat
            "ElemsPerStream": ("ints", True, []),
            # FINN DataTypes for inputs; output datatype inferred from input
            "inputDataType": ("s", True, ""),
            # number of input vectors for non-concat axes, examples:
            # [1] is a single vector (like a FC layer with batch=1)
            # [4] is four vectors (like a FC layer with batch=4)
            # [1, 4, 4] is four * four vectors (like a conv layer with batch=1)
            "numInputVectors": ("ints", False, [1]),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_n_inputs(self):
        return len(self.get_nodeattr("ElemsPerStream"))

    def get_total_elems(self):
        elems_per_stream = self.get_nodeattr("ElemsPerStream")
        return int(np.sum(elems_per_stream))

    def get_normal_input_shape(self, ind=0):
        elems_per_stream = self.get_nodeattr("ElemsPerStream")
        elems = elems_per_stream[ind]
        vecs = list(self.get_nodeattr("numInputVectors"))
        ishape = tuple(vecs + [elems])
        return ishape

    def get_folded_input_shape(self, ind=0):
        return self.get_normal_input_shape(ind)

    def get_normal_output_shape(self, ind=0):
        total_elems = self.get_total_elems()
        vecs = list(self.get_nodeattr("numInputVectors"))
        return tuple(vecs + [total_elems])

    def get_folded_output_shape(self, ind=0):
        return self.get_normal_output_shape()

    def make_shape_compatible_op(self, model):
        # check all input shapes
        for i, inp in enumerate(self.onnx_node.input):
            exp_ishape = self.get_normal_input_shape(i)
            ishape = tuple(model.get_tensor_shape(inp))
            assert ishape == exp_ishape, "Unexpected shape for " + inp
        oshape = self.get_normal_output_shape()
        return super().make_const_shape_op(oshape)

    def infer_node_datatype(self, model):
        # check all input datatypes
        for i, inp in enumerate(self.onnx_node.input):
            idt = model.get_tensor_datatype(inp)
            assert idt == self.get_input_datatype()
        odt = self.get_output_datatype()
        model.set_tensor_datatype(self.onnx_node.output[0], odt)

    def verify_node(self):
        pass

    def get_input_datatype(self, ind=0):
        # input dt identical for all inputs
        return DataType[self.get_nodeattr("inputDataType")]

    def get_output_datatype(self, ind=0):
        return self.get_input_datatype()

    def get_instream_width(self, ind=0):
        elems_per_stream = self.get_nodeattr("ElemsPerStream")
        elems = elems_per_stream[ind]
        ibits = self.get_input_datatype().bitwidth()
        return elems * ibits

    def get_outstream_width(self, ind=0):
        obits = self.get_output_datatype().bitwidth()
        total_elems = self.get_total_elems()
        out_width = total_elems * obits
        return out_width

    def get_number_output_values(self):
        return np.prod(self.get_folded_output_shape()[:-1])

    def get_exp_cycles(self):
        return np.prod(self.get_folded_output_shape()[:-1])

    def execute_node(self, context, graph):
        node = self.onnx_node
        inp_values = []
        for inp in node.input:
            inp_values.append(context[inp])
        result = np.concatenate(inp_values, axis=-1)
        context[node.output[0]] = result

    def get_instream_width_padded(self, ind=0):
        in_width = self.get_instream_width(ind)
        return roundup_to_integer_multiple(in_width, 8)

    def get_verilog_top_module_intf_names(self):
        intf_names = super().get_verilog_top_module_intf_names()
        n_inputs = self.get_n_inputs()
        sname = self.hls_sname()
        intf_names["s_axis"] = []
        for i in range(n_inputs):
            intf_names["s_axis"].append(("in%d_%s" % (i, sname), self.get_instream_width_padded(i)))
        return intf_names
