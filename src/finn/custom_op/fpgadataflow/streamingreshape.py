# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Hardware layer for the ONNX shape operators Reshape, Flatten, Squeeze and
Unsqueeze."""

import numpy as np
import warnings
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class StreamingReshape(HWCustomOp):
    """Re-interprets the shape of a stream without touching its data.

    A FINN stream carries a tensor's elements in row-major order of its shape.
    Reshape, Flatten, Squeeze and Unsqueeze keep that order and only change the
    shape, so the stream itself passes through unchanged. The layer exists so
    that the tensors before and after it keep their own shapes in the graph:
    its input is annotated with the shape of the producer, its output with the
    shape the consumer expects.

    Both streams carry PE elements per cycle, so PE must divide the innermost
    dimension of both shapes.
    """

    def get_nodeattr_types(self):
        my_attrs = {
            "DataType": ("s", True, ""),
            "in_shape": ("ints", True, []),
            "out_shape": ("ints", True, []),
            "PE": ("i", False, 1),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_normal_input_shape(self, ind=0):
        return list(self.get_nodeattr("in_shape"))

    def get_normal_output_shape(self, ind=0):
        return list(self.get_nodeattr("out_shape"))

    def _folded(self, shape):
        pe = self.get_nodeattr("PE")
        assert shape[-1] % pe == 0, "PE must divide the innermost dimension %s" % str(shape)
        return tuple(shape[:-1] + [shape[-1] // pe, pe])

    def get_folded_input_shape(self, ind=0):
        return self._folded(self.get_normal_input_shape())

    def get_folded_output_shape(self, ind=0):
        return self._folded(self.get_normal_output_shape())

    def get_input_datatype(self, ind=0):
        return DataType[self.get_nodeattr("DataType")]

    def get_output_datatype(self, ind=0):
        return DataType[self.get_nodeattr("DataType")]

    def infer_node_datatype(self, model):
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            warnings.warn(
                "DataType changing for %s: %s -> %s"
                % (node.name, str(self.get_input_datatype()), str(idt))
            )
            self.set_nodeattr("DataType", idt.name)
        model.set_tensor_datatype(node.output[0], idt)

    def get_instream_width(self, ind=0):
        return self.get_input_datatype().bitwidth() * self.get_nodeattr("PE")

    def get_outstream_width(self, ind=0):
        return self.get_output_datatype().bitwidth() * self.get_nodeattr("PE")

    def get_exp_cycles(self):
        return int(np.prod(self.get_folded_output_shape()[:-1]))

    def execute_node(self, context, graph):
        node = self.onnx_node
        inp = context[node.input[0]]
        assert list(inp.shape) == self.get_normal_input_shape(), "Input shape mismatch"
        context[node.output[0]] = inp.reshape(self.get_normal_output_shape())
