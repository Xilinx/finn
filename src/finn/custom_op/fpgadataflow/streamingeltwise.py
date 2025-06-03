# Copyright (c) 2022, Xilinx
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


class StreamingEltwise(HWCustomOp):
    """Abstraction layer for HW implementation of StreamingEltwise"""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = super().get_nodeattr_types()
        my_attrs.update(
            {
                "NumChannels": ("i", True, ""),
                "PE": ("i", True, ""),
                # FINN DataTypes for inputs; output datatype inferred from input
                "inputDataType0": ("s", True, ""),
                "inputDataType1": ("s", True, ""),
                # type of EltwiseFunction for the operation
                "eltwiseOp": ("s", True, "", ["Add", "Sub", "AbsDiff"]),
                # number of input vectors, examples:
                # [1] is a single vector (like a FC layer with batch=1)
                # [4] is four vectors (like a FC layer with batch=4)
                # [1, 4, 4] is four * four vectors (like a conv layer with batch=1)
                "numInputVectors": ("ints", False, [1]),
                "inFIFODepths": ("ints", False, [2, 2]),
            }
        )
        return my_attrs

    def get_eltwise_op_lambda(self):
        eltwise_op = self.get_nodeattr("eltwiseOp")
        idt0 = self.get_input_datatype(0)
        idt1 = self.get_input_datatype(1)
        odt = self.get_output_datatype()
        tin0 = idt0.get_hls_datatype_str()
        tin1 = idt1.get_hls_datatype_str()
        tout = odt.get_hls_datatype_str()
        eltwise_ops = {
            # "Add": "[](auto a, auto b) { return  a + b; }",
            # "Sub": "[](auto a, auto b) { return  a - b; }",
            # "AbsDiff": "[](auto a, auto b) { return  a>b? a-b : b-a; }",
            "Add": f"add<{tin0}, {tin1}, {tout}>()",
            "Sub": f"sub<{tin0}, {tin1}, {tout}>()",
            "AbsDiff": f"absdiff<{tin0}, {tin1}, {tout}>()",
        }
        return eltwise_ops[eltwise_op]

    def get_normal_input_shape(self, ind=0):
        ich = self.get_nodeattr("NumChannels")
        vecs = list(self.get_nodeattr("numInputVectors"))
        ishape = tuple(vecs + [ich])
        return ishape

    def get_folded_input_shape(self, ind=0):
        ich = self.get_nodeattr("NumChannels")
        pe = self.get_nodeattr("PE")
        assert ich % pe == 0, "PE must divide NumChannels"
        vecs = list(self.get_nodeattr("numInputVectors"))
        ishape = tuple(vecs + [ich // pe, pe])
        return ishape

    def get_normal_output_shape(self, ind=0):
        return self.get_normal_input_shape()

    def get_folded_output_shape(self, ind=0):
        return self.get_folded_input_shape()

    def infer_node_datatype(self, model):
        node = self.onnx_node
        idt0 = model.get_tensor_datatype(node.input[0])
        if idt0 != self.get_input_datatype(0):
            warn_str = "inputDataType0 changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype(0)),
                str(idt0),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("inputDataType0", idt0.name)
        idt1 = model.get_tensor_datatype(node.input[1])
        if idt1 != self.get_input_datatype(1):
            warn_str = "inputDataType1 changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype(1)),
                str(idt1),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("inputDataType1", idt1.name)
        # enforce output data type (calculated based on idt)
        odt = self.get_output_datatype()
        model.set_tensor_datatype(self.onnx_node.output[0], odt)

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        return DataType[self.get_nodeattr("inputDataType" + str(ind))]

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output."""
        op = self.get_nodeattr("eltwiseOp")
        idt0 = self.get_input_datatype(0)
        idt1 = self.get_input_datatype(1)
        assert idt0.signed() == idt1.signed(), (
            "%s: Inputs must have same signedness" % self.onnx_node.name
        )
        idt0_min, idt0_max = idt0.min(), idt0.max()
        idt1_min, idt1_max = idt1.min(), idt1.max()
        cands = [
            idt0_min - idt1_min,
            idt0_min - idt1_max,
            idt0_max - idt1_min,
            idt0_max - idt1_max,
        ]
        largest_magnitude = max(map(abs, cands))
        if op == "Add":
            if idt0.signed():
                return DataType.get_smallest_possible(idt0.min() + idt1.min())
            else:
                return DataType.get_smallest_possible(idt0.max() + idt1.max())
        elif op == "Sub":
            return DataType.get_smallest_possible(-largest_magnitude)
        elif op == "AbsDiff":
            return DataType.get_smallest_possible(largest_magnitude)
        else:
            raise Exception("%s: Unknown eltWiseOp = %s" % (self.onnx_node.name, op))

    def get_instream_width(self, ind=0):
        """Returns input stream width."""
        ibits = self.get_input_datatype(ind).bitwidth()
        pe = self.get_nodeattr("PE")
        in_width = pe * ibits
        return in_width

    def get_outstream_width(self, ind=0):
        """Returns output stream width."""
        obits = self.get_output_datatype().bitwidth()
        pe = self.get_nodeattr("PE")
        out_width = pe * obits
        return out_width

    def get_exp_cycles(self):
        # Channels/PE * batch size * fmdim * fmdim
        return np.prod(self.get_folded_output_shape()[:-1])

    def execute_node(self, context, graph):
        # simulate behavior using Python
        node = self.onnx_node
        inp0_values = context[node.input[0]]
        inp1_values = context[node.input[1]]
        eltwiseOp = self.get_nodeattr("eltwiseOp")
        oshape = context[node.output[0]].shape
        ishape0 = inp0_values.shape
        ishape1 = inp1_values.shape
        assert ishape0 == ishape1, "Shapes of inputs should be the same for Streamingeltwise"
        # subtraction
        result = inp0_values - inp1_values
        if eltwiseOp == "Sub":
            context[node.output[0]] = np.asarray(result, dtype=np.float32).reshape(oshape)
        elif eltwiseOp == "AbsDiff":
            context[node.output[0]] = np.abs(np.asarray(result, dtype=np.float32)).reshape(oshape)
        else:
            raise Exception("%s: Unknown eltWiseOp = %s" % (node.name, eltwiseOp))

    def get_verilog_top_module_intf_names(self):
        intf_names = super().get_verilog_top_module_intf_names()
        swidth = self.get_instream_width_padded()
        intf_names["s_axis"] = [(x + "_V", swidth) for x in ["in0", "in1"]]
        return intf_names
