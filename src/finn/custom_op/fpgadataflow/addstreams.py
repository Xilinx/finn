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


class AddStreams(HWCustomOp):
    """Abstraction layer for HW implementation of AddStreams."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = super().get_nodeattr_types()
        my_attrs.update(
            {
                "NumChannels": ("i", True, ""),
                "PE": ("i", True, ""),
                # FINN DataTypes for inputs; output datatype inferred from input
                "inputDataTypes": ("strings", True, [""]),
                # number of input vectors, examples:
                # [1] is a single vector (like a FC layer with batch=1)
                # [4] is four vectors (like a FC layer with batch=4)
                # [1, 4, 4] is four * four vectors (like a conv layer with batch=1)
                "numInputVectors": ("ints", False, [1]),
                "inFIFODepths": ("ints", False, [2, 2]),
            }
        )
        return my_attrs

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
        # enforce output data type (calculated based on idt)
        odt = self.get_output_datatype()
        model.set_tensor_datatype(self.onnx_node.output[0], odt)

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        return DataType[self.get_nodeattr("inputDataTypes")[ind]]

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output."""
        min_input = 0
        max_input = 0
        for i in range(len(self.get_nodeattr("inputDataTypes"))):
            idt = self.get_input_datatype(i)
            if idt.min() < min_input:
                min_input = idt.min()
            if idt.max() > max_input:
                max_input = idt.max()
        if min_input >= 0:
            out_bit_width = math.ceil(np.log2(max_input + 1))
            odt = DataType[f"UINT{out_bit_width + 1}"]
        else:
            max_abs_input = max(-min_input, 1 + max_input)
            out_bit_width = math.ceil(np.log2(max_abs_input) + 1)
            odt = DataType[f"INT{out_bit_width + 1}"]
        return odt

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
        oshape = context[node.output[0]].shape
        ishape0 = inp0_values.shape
        ishape1 = inp1_values.shape
        assert ishape0 == ishape1, "Shapes of inputs should be the same for Addstreams"
        result = inp0_values + inp1_values
        context[node.output[0]] = np.asarray(result, dtype=np.float32).reshape(oshape)

    def derive_characteristic_fxns(self, period):
        n_inps = np.prod(self.get_folded_input_shape()[:-1])
        io_dict = {
            "inputs": {
                "in0": [0 for i in range(n_inps)],
                "in1": [0 for i in range(n_inps)],
            },
            "outputs": {"out0": []},
        }
        super().derive_characteristic_fxns(period, override_rtlsim_dict=io_dict)
