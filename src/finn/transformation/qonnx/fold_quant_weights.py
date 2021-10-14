# Copyright (c) 2021, Xilinx
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
from onnx import TensorProto, helper

import finn.core.onnx_exec as oxe
from finn.custom_op.registry import getCustomOp
from finn.transformation.base import Transformation
from finn.transformation.infer_shapes import InferShapes


class FoldQuantWeights(Transformation):
    """Merges Quant nodes, which are used as weights into the initializer
    of the weight tensor.
    """

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        execution_context = model.make_empty_exec_context()
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Quant" or n.op_type == "BinaryQuant":
                node_inp_inits = list(map(lambda x: model.get_initializer(x), n.input))
                node_inp_dyn = list(filter(lambda x: x is None, node_inp_inits))
                node_out = n.output[0]
                is_all_constant_inputs = len(node_inp_dyn) == 0
                ishape = model.get_tensor_shape(n.input[0])
                is_const_shape = (n.op_type == "Shape") and (ishape is not None)
                if is_all_constant_inputs or is_const_shape:
                    if (
                        n.op_type == "Quant"
                        and not model.get_initializer(n.input[2]) == 0
                    ):
                        raise ValueError(
                            "Only Quant nodes with zero-point == 0 "
                            "are currently supported."
                        )
                    scale = model.get_initializer(n.input[1])
                    unity_scale = (scale.flatten() == 1.0).all()
                    # this node has no dynamic inputs, only constant ones -- so we can
                    # do constant folding.
                    oxe.execute_node(n, execution_context, graph)
                    q_node_output = execution_context[node_out]
                    # Check we can directly constant fold
                    if unity_scale:
                        # use the execution result as an initializer
                        model.set_initializer(node_out, q_node_output)
                    else:
                        # Reshape scale for Conv if required
                        if model.is_fork_node(n):
                            raise RuntimeError(
                                "Weights quantized with the Quant node are not "
                                "allowed to be join nodes node."
                            )
                        target_node = model.find_direct_successors(n)
                        if target_node is None:
                            raise RuntimeError(
                                "Weights quantized with the Quant node must have "
                                "a successor node."
                            )
                        else:
                            target_node = target_node[0]

                        # Check next operator type
                        mul_like_nodes = ["Mul", "Div", "Conv", "MatMul"]
                        add_like_nodes = ["Add", "Sub"]
                        all_supported_ops = mul_like_nodes.copy()
                        all_supported_ops.extend(add_like_nodes)

                        if target_node.op_type not in all_supported_ops:
                            raise ValueError(
                                f"Can't constant fold Quant weight node "
                                f"into node type {target_node.op_type} "
                                f"at node: {target_node}."
                            )

                        # For both mul and Add:
                        # Move the scale factor behind the next operator
                        scale = model.get_initializer(n.input[1])
                        new_initializer = q_node_output / scale
                        # Round, to correct for floating point errors
                        new_initializer = np.round(new_initializer)
                        model.set_initializer(node_out, new_initializer)
                        q_inst = getCustomOp(n)
                        new_dtype = q_inst.get_internal_dtype(model)
                        model.set_tensor_datatype(node_out, new_dtype)

                        if target_node.op_type == "Conv" and len(scale.shape) > 0:
                            bias_shape = [1] * len(scale.shape)
                            bias_shape[1] = -1
                            scale = scale.reshape(bias_shape)

                        if scale.shape == (1,):
                            scale = scale[0]
                            mul_shape = tuple()
                        else:
                            mul_shape = scale.shape
                        mul_tensor = helper.make_tensor_value_info(
                            model.make_new_valueinfo_name(),
                            TensorProto.FLOAT,
                            mul_shape,
                        )
                        graph.value_info.append(mul_tensor)
                        model.set_initializer(mul_tensor.name, scale)

                        successor = model.find_consumers(node_out)
                        if successor is None:
                            raise RuntimeError(
                                "Can only constant fold scaled Quant weights "
                                "if a successor exists."
                            )
                        successor = successor[0]
                        succ_output_name = successor.output[0]

                        output_shape = model.get_tensor_shape(successor.output[0])
                        act_mul_tensor = helper.make_tensor_value_info(
                            model.make_new_valueinfo_name(),
                            TensorProto.FLOAT,
                            output_shape,
                        )
                        graph.value_info.append(act_mul_tensor)
                        successor.output[0] = act_mul_tensor.name

                        mul_node = helper.make_node(
                            "Mul",
                            [act_mul_tensor.name, mul_tensor.name],
                            [succ_output_name],
                        )
                        graph.node.insert(node_ind, mul_node)

                        if target_node.op_type in add_like_nodes:
                            # Move the scale factor behind also in-front of
                            # the next operator
                            div_tensor = helper.make_tensor_value_info(
                                model.make_new_valueinfo_name(),
                                TensorProto.FLOAT,
                                mul_shape,
                            )
                            graph.value_info.append(div_tensor)
                            model.set_initializer(div_tensor.name, scale)

                            succ_input_name = successor.input[0]
                            act_mul_tensor = helper.make_tensor_value_info(
                                model.make_new_valueinfo_name(),
                                TensorProto.FLOAT,
                                output_shape,
                            )
                            graph.value_info.append(act_mul_tensor)
                            successor.input[0] = act_mul_tensor.name

                            div_node = helper.make_node(
                                "Div",
                                [succ_input_name, div_tensor.name],
                                [act_mul_tensor.name],
                            )
                            graph.node.insert(node_ind, div_node)

                    # remove old node
                    graph.node.remove(n)
                    graph_modified = True
                    model = model.transform(InferShapes())
                    return (model, graph_modified)
        return (model, graph_modified)
