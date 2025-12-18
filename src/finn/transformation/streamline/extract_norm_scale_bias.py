############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright for portions of this file is held by AMD and Microsoft under
# MIT license as part of project Brainsmith.
# All other copyright is held by AMD and is provided under BSD-3-Clause license.
#
# Note: This transform was originally written by Thomas Keller (ExpandNorms)
# and was adjusted.
#
############################################################################

import numpy as np
from onnx import TensorProto
from onnx import helper as oh
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes


class ExtractNormScaleBias(Transformation):
    """Extract LayerNormalization scale and bias into separate nodes
    and set initializers to 1 or 0 respectively."""

    def __init__(self):
        super().__init__()

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False

        for node in graph.node:
            node_ind += 1
            if node.op_type == "LayerNormalization":
                scale = model.get_initializer(node.input[1])
                if len(node.input) > 2:
                    bias = model.get_initializer(node.input[2])
                else:
                    bias = None
                scale_is_one = (scale == 1).all()
                bias_is_zero = not np.any(bias)
                if scale_is_one and (bias_is_zero or bias is None):
                    continue
                act_shape = model.get_tensor_shape(node.input[0])
                act_out = node.output[0]
                if not scale_is_one:
                    # extract scale into separate Mul node
                    scale_dt = model.get_tensor_datatype(node.input[1])
                    # Create new tensors
                    scale_act_in = oh.make_tensor_value_info(
                        model.make_new_valueinfo_name(), TensorProto.FLOAT, act_shape
                    )
                    scale_value = oh.make_tensor_value_info(
                        model.make_new_valueinfo_name(), TensorProto.FLOAT, [act_shape[-1]]
                    )
                    graph.value_info.append(scale_act_in)
                    graph.value_info.append(scale_value)

                    # Update previous output tensor
                    node.output[0] = scale_act_in.name
                    # Create Mul node to replace scale
                    mul_node = oh.make_node("Mul", [scale_act_in.name, scale_value.name], [act_out])

                    # set scale to all ones in LayerNormalization
                    model.set_initializer(node.input[1], np.ones(act_shape[-1], dtype=np.float32))

                    graph_modified = True

                if not bias_is_zero or bias is not None:
                    # extract bias into separate Add node
                    bias_dt = model.get_tensor_datatype(node.input[2])
                    # Create new input tensor
                    bias_act_in = oh.make_tensor_value_info(
                        model.make_new_valueinfo_name(), TensorProto.FLOAT, act_shape
                    )
                    bias_value = oh.make_tensor_value_info(
                        model.make_new_valueinfo_name(), TensorProto.FLOAT, [act_shape[-1]]
                    )
                    graph.value_info.append(bias_act_in)
                    graph.value_info.append(bias_value)
                    # Update previous output tensor
                    if not scale_is_one:
                        mul_node.output[0] = bias_act_in.name
                    else:
                        node.output[0] = bias_act_in.name

                    # Create Add node to replace bias
                    add_node = oh.make_node("Add", [bias_act_in.name, bias_value.name], [act_out])

                    # set bias to all zeros in LayerNormalization
                    model.set_initializer(node.input[2], np.zeros(act_shape[-1], dtype=np.float32))

                    graph_modified = True

                # insert new nodes
                insert_point = node_ind
                if not scale_is_one:
                    insert_point += 1
                    graph.node.insert(insert_point, mul_node)
                    model.set_initializer(mul_node.input[1], scale)
                    model.set_tensor_datatype(mul_node.input[1], scale_dt)
                if not bias_is_zero or bias is not None:
                    insert_point += 1
                    graph.node.insert(insert_point, add_node)
                    model.set_initializer(add_node.input[1], bias)
                    model.set_tensor_datatype(add_node.input[1], bias_dt)
                model = model.transform(InferShapes())
                model = model.transform(InferDataTypes())

        return (model, graph_modified)
