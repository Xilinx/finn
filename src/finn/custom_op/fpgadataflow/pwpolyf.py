# Copyright (C) 2026, Advanced Micro Devices, Inc.
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

# Piecewise polynomial constants matching the RTL module
_NUM_OCTAVES = 5
_SUPPORTED_FUNCS = {"gelu", "silu", "sigmoid", "tanh"}


class PWPolyF(HWCustomOp):
    """
    HW op for piecewise polynomial activations (GELU, SiLU, Sigmoid, Tanh).

    Element-wise FP32, coefficients baked into RTL.  No weights or BRAM.
    """

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            # activation function: gelu, silu, sigmoid, tanh
            "func": ("s", True, ""),
            # top-mantissa subdivision bits (K=3 gives 81 segments)
            "K": ("i", False, 3),
            # parallelism; elements processed per cycle
            "PE": ("i", True, 0),
            # number of channels (last dimension of input tensor)
            "NumChannels": ("i", True, 0),
            # FINN DataTypes for inputs, outputs (always FLOAT32)
            "inputDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
            # polynomial degree (number of FMA stages per PE)
            "degree": ("i", False, 2),
            # number of input vectors, examples:
            # [1] is a single vector (like a FC layer with batch=1)
            # [4] is four vectors (like a FC layer with batch=4)
            # [1, 4, 4] is four * four vectors (like a conv layer with batch=1)
            "numInputVectors": ("ints", False, [1]),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_num_segments(self):
        K = self.get_nodeattr("K")
        return 1 + 2 * _NUM_OCTAVES * (1 << K)

    def make_shape_compatible_op(self, model):
        oshape = self.get_normal_output_shape()
        return super().make_const_shape_op(oshape)

    def infer_node_datatype(self, model):
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            self.set_nodeattr("inputDataType", idt.name)
        odt = self.get_output_datatype()
        model.set_tensor_datatype(node.output[0], odt)

    def verify_node(self):
        info_messages = []

        backend_value = self.get_nodeattr("backend")
        if backend_value == "fpgadataflow":
            info_messages.append("Attribute backend is set correctly")
        else:
            info_messages.append('Attribute backend should be set to "fpgadataflow"')

        func = self.get_nodeattr("func")
        if func in _SUPPORTED_FUNCS:
            info_messages.append("Attribute func is set correctly")
        else:
            info_messages.append(
                "Attribute func must be one of %s, got %s" % (_SUPPORTED_FUNCS, func)
            )

        pe = self.get_nodeattr("PE")
        nch = self.get_nodeattr("NumChannels")
        if pe > 0 and nch > 0 and nch % pe == 0:
            info_messages.append("PE divides NumChannels")
        else:
            info_messages.append("PE must divide NumChannels evenly")

        idt = self.get_nodeattr("inputDataType")
        if idt != "FLOAT32":
            info_messages.append("PWPolyF requires FLOAT32 input, got %s" % idt)

        return info_messages

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        return DataType[self.get_nodeattr("inputDataType")]

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output."""
        return DataType[self.get_nodeattr("outputDataType")]

    def get_instream_width(self, ind=0):
        return self.get_input_datatype().bitwidth() * self.get_nodeattr("PE")

    def get_outstream_width(self, ind=0):
        return self.get_output_datatype().bitwidth() * self.get_nodeattr("PE")

    def get_folded_input_shape(self, ind=0):
        pe = self.get_nodeattr("PE")
        nch = self.get_nodeattr("NumChannels")
        fold = nch // pe
        vecs = list(self.get_nodeattr("numInputVectors"))
        return tuple(vecs + [fold, pe])

    def get_folded_output_shape(self, ind=0):
        return self.get_folded_input_shape()

    def get_normal_input_shape(self, ind=0):
        nch = self.get_nodeattr("NumChannels")
        vecs = list(self.get_nodeattr("numInputVectors"))
        return tuple(vecs + [nch])

    def get_normal_output_shape(self, ind=0):
        return self.get_normal_input_shape()

    def get_number_output_values(self):
        return np.prod(self.get_folded_output_shape()[:-1])

    def get_exp_cycles(self):
        # II=1, latency amortised over stream length
        return np.prod(self.get_folded_output_shape()[:-1])

    def lut_estimation(self):
        pe = self.get_nodeattr("PE")
        degree = self.get_nodeattr("degree")
        return 100 * degree * pe

    def bram_estimation(self):
        # coefficients stored in LUT ROM, not BRAM
        return 0

    def uram_estimation(self):
        return 0

    def dsp_estimation(self, fpgapart=None):
        pe = self.get_nodeattr("PE")
        degree = self.get_nodeattr("degree")
        return degree * pe

    def execute_node(self, context, graph):
        node = self.onnx_node
        inp = context[node.input[0]]

        func = self.get_nodeattr("func")
        K = self.get_nodeattr("K")

        # lazy import to avoid hard dependency on torch at module level
        import torch  # noqa: PLC0415

        from finn.util.pwpolyf import PiecewisePolyActivation  # noqa: PLC0415

        degree = self.get_nodeattr("degree")
        mod = PiecewisePolyActivation(func, K=K, degree=degree)
        with torch.no_grad():
            x = torch.from_numpy(inp.astype(np.float32))
            y = mod(x)
        context[node.output[0]] = y.numpy()
