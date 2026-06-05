# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.util.basic import get_by_name


class RemoveStreamOnlyShapeOps(Transformation):
    """Remove stream-shape-only ONNX shape ops between HW nodes.

    This transformation is intended to run after fpgadataflow HW layer inference
    has consumed tensor shape metadata, but before dataflow partitioning. It
    removes standalone ONNX Reshape/Squeeze/Unsqueeze nodes whose data stream
    length is unchanged and which sit between fpgadataflow paths.

    For Squeeze/Unsqueeze, require the final dimension to be unchanged and the
    inserted/removed axes to be size 1. For Reshape, only require static shape
    metadata and equal element count, since reshape can intentionally regroup
    channel vectors after HW conversion.
    """

    def _is_standard_onnx_node(self, node):
        return node.domain in {"", "ai.onnx"}

    def _is_hw_node(self, node):
        backend = get_by_name(node.attribute, "backend")
        return backend is not None and backend.s.decode("UTF-8") == "fpgadataflow"

    def _has_hw_predecessor(self, model, tensor_name, visited=None):
        if visited is None:
            visited = set()
        if tensor_name in visited:
            return False
        visited.add(tensor_name)

        producer = model.find_producer(tensor_name)
        if producer is None:
            return False
        if self._is_hw_node(producer):
            return True

        for inp in producer.input:
            if model.get_initializer(inp) is None and self._has_hw_predecessor(
                model, inp, visited
            ):
                return True
        return False

    def _has_hw_successor(self, model, tensor_name, visited=None):
        if visited is None:
            visited = set()
        if tensor_name in visited:
            return False
        visited.add(tensor_name)

        consumers = model.find_consumers(tensor_name)
        if consumers is None:
            return False

        for consumer in consumers:
            if self._is_hw_node(consumer):
                return True
            for outp in consumer.output:
                if self._has_hw_successor(model, outp, visited):
                    return True
        return False

    def _get_axes(self, model, node, rank):
        if len(node.input) > 1:
            axes = model.get_initializer(node.input[1])
            if axes is None:
                return None
            axes = np.asarray(axes, dtype=np.int64).flatten().tolist()
        else:
            axes_attr = get_by_name(node.attribute, "axes")
            if axes_attr is None:
                return None
            axes = list(axes_attr.ints)

        norm_axes = []
        for axis in axes:
            axis = int(axis)
            if axis < 0:
                axis += rank
            if axis < 0 or axis >= rank:
                return None
            norm_axes.append(axis)
        if len(set(norm_axes)) != len(norm_axes):
            return None
        return sorted(norm_axes)

    def _get_shape(self, model, tensor_name):
        shape = model.get_tensor_shape(tensor_name)
        if shape is None or len(shape) == 0:
            return None
        if any(x is None for x in shape):
            return None
        return shape

    def _same_element_count(self, model, node):
        inp_shape = model.get_tensor_shape(node.input[0])
        out_shape = model.get_tensor_shape(node.output[0])
        if inp_shape is None or out_shape is None:
            return False
        if len(inp_shape) == 0 or len(out_shape) == 0:
            return False
        if any(x is None for x in inp_shape + out_shape):
            return False
        if int(np.prod(inp_shape)) != int(np.prod(out_shape)):
            return False
        return True

    def _safe_squeeze_or_unsqueeze(self, model, node):
        inp_shape = self._get_shape(model, node.input[0])
        out_shape = self._get_shape(model, node.output[0])
        if inp_shape is None or out_shape is None:
            return False
        if not self._same_element_count(model, node):
            return False
        if inp_shape[-1] != out_shape[-1]:
            return False

        if node.op_type == "Squeeze":
            axes = self._get_axes(model, node, len(inp_shape))
            if axes is None:
                return False
            return all(inp_shape[axis] == 1 for axis in axes)

        if node.op_type == "Unsqueeze":
            axes = self._get_axes(model, node, len(out_shape))
            if axes is None:
                return False
            return all(out_shape[axis] == 1 for axis in axes)

        return False

    def _safe_reshape(self, model, node):
        if len(node.input) < 2:
            return False
        if model.get_initializer(node.input[1]) is None:
            return False
        return self._same_element_count(model, node)

    def _safe_stream_shape_op(self, model, node):
        if not self._is_standard_onnx_node(node):
            return False
        if node.op_type in {"Squeeze", "Unsqueeze"}:
            return self._safe_squeeze_or_unsqueeze(model, node)
        if node.op_type == "Reshape":
            return self._safe_reshape(model, node)
        return False

    def _can_remove(self, model, node):
        if node.op_type not in {"Reshape", "Squeeze", "Unsqueeze"}:
            return False
        if node.output[0] in {x.name for x in model.graph.output}:
            return False
        if not self._safe_stream_shape_op(model, node):
            return False
        if not self._has_hw_predecessor(model, node.input[0]):
            return False
        if not self._has_hw_successor(model, node.output[0]):
            return False
        return True

    def apply(self, model: ModelWrapper):
        # Prefer removing Squeeze first. In an Unsqueeze -> Squeeze chain, removing
        # the Squeeze first preserves a valid graph for the next iteration. Then
        # remove Unsqueeze and standalone Reshape nodes.
        for op_type in ["Squeeze", "Unsqueeze", "Reshape"]:
            for node in list(model.graph.node):
                if node.op_type != op_type or not self._can_remove(model, node):
                    continue

                data_input = node.input[0]
                data_output = node.output[0]
                consumers = model.find_consumers(data_output)
                if consumers is None:
                    continue
                for consumer in consumers:
                    for i, inp in enumerate(consumer.input):
                        if inp == data_output:
                            consumer.input[i] = data_input
                model.graph.node.remove(node)
                return (model, True)

        return (model, False)
