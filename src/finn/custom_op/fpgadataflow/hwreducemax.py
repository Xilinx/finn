############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

import warnings
import numpy as np
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class HWReduceMax(HWCustomOp):

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            "ifm_dim": ("ints", True, []),
            "SIMD": ("i", False, 1),
            "input_data_type": ("s", True, ""),
            "NumChannels": ("i", False, 128),
            "axis": ("i", False, -1),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_normal_input_shape(self, ind=0):
        return self.get_nodeattr("ifm_dim")

    def get_normal_output_shape(self, ind=0):
        input_shape = list(self.get_normal_input_shape())
        axis = self.get_nodeattr("axis")
        if axis < 0:
            axis = len(input_shape) + axis
        output_shape = input_shape.copy()
        output_shape[axis] = 1
        return output_shape

    def execute_node(self, context, graph):
        node = self.onnx_node
        input_data = context[node.input[0]]
        axis = self.get_nodeattr("axis")
        output_data = np.max(input_data, axis=axis, keepdims=True)
        context[node.output[0]] = output_data

    def get_input_datatype(self, ind=0):
        data_type = DataType[self.get_nodeattr("input_data_type")]
        return data_type

    def infer_node_datatype(self, model):
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            warn_str = "input_data_type changing for %s: %s -> %s " % (
                node.name, str(self.get_input_datatype()), str(idt))
            warnings.warn(warn_str)
        self.set_nodeattr("input_data_type", idt.name)
        odt = self.get_output_datatype()
        model.set_tensor_datatype(node.output[0], odt)

    def get_instream_width(self, ind=0):
        ibits = self.get_input_datatype().bitwidth()
        simd = self.get_nodeattr("SIMD")
        return ibits * simd

    def get_outstream_width(self, ind=0):
        obits = self.get_output_datatype().bitwidth()
        simd = self.get_nodeattr("SIMD")
        return obits * simd

    def get_output_datatype(self, ind=0):
        return self.get_input_datatype()

    def get_folded_output_shape(self, ind=0):
        return self.get_normal_output_shape()

    def get_folded_input_shape(self, ind=0):
        normal_ishape = list(self.get_normal_input_shape())
        simd = self.get_nodeattr("SIMD")
        assert normal_ishape[-1] % simd == 0, "SIMD must divide into input dimension"
        fold = int(normal_ishape[-1] / simd)
        folded_ishape = normal_ishape[:-1] + [fold, simd]
        return tuple(folded_ishape)

    def get_exp_cycles(self):
        ifm_dim = self.get_nodeattr("ifm_dim")
        simd = self.get_nodeattr("SIMD")
        total_elements = np.prod(ifm_dim)
        cycles = total_elements // simd
        return cycles