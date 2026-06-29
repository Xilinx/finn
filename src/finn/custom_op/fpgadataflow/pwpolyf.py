# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import math
import numpy as np
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp

# NUM_OCTAVES is fixed by the RTL segment decode and clamp range. K controls
# the number of mantissa subdivisions inside each of these fixed octaves.
_NUM_OCTAVES = 5
_SUPPORTED_FUNCS = {"gelu", "silu", "sigmoid", "tanh"}


class PWPolyF(HWCustomOp):
    """
    HW op for piecewise polynomial activations (GELU, SiLU, Sigmoid, Tanh).

    Element-wise FP32, coefficients baked into RTL.  No weights.
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
        assert idt == DataType["FLOAT32"], "%s: PWPolyF requires FLOAT32 input, got %s" % (
            node.name,
            idt,
        )
        self.set_nodeattr("inputDataType", idt.name)
        self.set_nodeattr("outputDataType", idt.name)
        model.set_tensor_datatype(node.output[0], idt)

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
        odt = self.get_nodeattr("outputDataType")
        if odt != "FLOAT32":
            info_messages.append("PWPolyF requires FLOAT32 output, got %s" % odt)

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

    def get_exp_cycles(self):
        # II=1, latency amortised over stream length
        return np.prod(self.get_folded_output_shape()[:-1])

    def lut_estimation(self):
        pe = self.get_nodeattr("PE")
        degree = self.get_nodeattr("degree")
        return 100 * degree * pe

    def bram_estimation(self):
        pe = self.get_nodeattr("PE")
        degree = self.get_nodeattr("degree")
        num_segs = self.get_num_segments()

        if degree <= 1:
            return 0

        # Stages after the first use a registered dynamic coefficient lookup
        # for the DSP C input. Vivado infers this as one 32-bit wide ROM per
        # stage and PE, backed by RAMB18 for the default K=3 table depth.
        coeff_width = 32
        if coeff_width <= 18 or num_segs > 512:
            bram18_per_coeff_rom = math.ceil(num_segs / 1024) * math.ceil(coeff_width / 18)
        else:
            bram18_per_coeff_rom = math.ceil(num_segs / 512) * math.ceil(coeff_width / 36)
        return pe * (degree - 1) * bram18_per_coeff_rom

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

        from finn.util.torch_hw_modules import (  # noqa: PLC0415
            PWPolyFActivation,
        )

        degree = self.get_nodeattr("degree")
        mod = PWPolyFActivation(func, K=K, degree=degree)
        with torch.no_grad():
            x = torch.from_numpy(inp.astype(np.float32))
            y = mod(x)
        context[node.output[0]] = y.numpy()
