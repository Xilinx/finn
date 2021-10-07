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
from qonnx.transformation.quant_constant_folding import FoldTransposeIntoQuantInit

import finn.core.onnx_exec as oxe
from finn.core.datatype import DataType
from finn.transformation.base import Transformation
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.qonnx.gemm_to_matmul import GemmToMatMul
from finn.transformation.qonnx.qonnx_activation_handlers import QuantActBaseHandler
from finn.util.basic import get_by_name


class ConvertQONNXtoFINN(Transformation):
    """Converts QONNX dialect to FINN ONNX dialect.
    First the weights are converted using the FoldQuantWeights transformation,
    then the ConvertQuantActToMultiThreshold transformation is used to convert
    the activations.
    If incompatibilities are found a ValueError or RuntimeError is raised.
    """

    def apply(self, model):
        # Gemm operations are not supported by FINN, so we convert them to MatMul
        model = model.transform(GemmToMatMul())
        model = model.transform(FoldTransposeIntoQuantInit())
        # Make sure the datatypes exist, these are required for folding the weights
        model = model.transform(InferDataTypes())
        # Fold weights
        model = model.transform(FoldQuantWeights())
        # Convert activations
        model = model.transform(ConvertQuantActToMultiThreshold())

        # Unset FINN datatypes from MultiThreshold node output tensors to avoid warnings
        mt_nodes = model.get_nodes_by_op_type("MultiThreshold")
        qnt_annotations = model._model_proto.graph.quantization_annotation
        for n in mt_nodes:
            ret = get_by_name(qnt_annotations, n.output[0], "tensor_name")
            if ret is not None:
                ret_dt = get_by_name(
                    ret.quant_parameter_tensor_names, "finn_datatype", "key"
                )
                if ret_dt is not None:
                    ret_dt.Clear()
        # ToDo: This might be supported by finn-base in the future,
        #  by calling the following:
        # model.set_tensor_datatype(n.output[0], None)

        return (model, False)


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
            if n.op_type == "Quant":
                node_inp_inits = list(map(lambda x: model.get_initializer(x), n.input))
                node_inp_dyn = list(filter(lambda x: x is None, node_inp_inits))
                node_out = n.output[0]
                is_all_constant_inputs = len(node_inp_dyn) == 0
                ishape = model.get_tensor_shape(n.input[0])
                is_const_shape = (n.op_type == "Shape") and (ishape is not None)
                if is_all_constant_inputs or is_const_shape:
                    if not model.get_initializer(n.input[2]) == 0:
                        raise ValueError(
                            "Only Quant nodes with zero-point == 0 "
                            "are currently supported."
                        )
                    # this node has no dynamic inputs, only constant ones -- so we can
                    # do constant folding.
                    oxe.execute_node(n, execution_context, graph)
                    q_node_output = execution_context[node_out]
                    # Check if the datatype can be directly constant folded
                    dtype = model.get_tensor_datatype(n.output[0])
                    if "SCALED" in dtype.name:
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

                        # For buth mul and Add:
                        # Move the scale factor behind the next operator
                        scale = model.get_initializer(n.input[1])
                        new_initializer = q_node_output / scale
                        # Round, to correct for floating point errors
                        new_initializer = np.round(new_initializer)
                        model.set_initializer(node_out, new_initializer)
                        new_dtype = DataType[dtype.name.replace("SCALED", "")]
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

                    else:
                        # use the execution result as an initializer
                        model.set_initializer(node_out, q_node_output)
                    # remove old node
                    graph.node.remove(n)
                    graph_modified = True
                    model = model.transform(InferShapes())
                    return (model, graph_modified)
        return (model, graph_modified)


class ConvertQuantActToMultiThreshold(Transformation):
    """Converts Quant nodes in the activation path to MultiThreshold nodes."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False

        for n in graph.node:
            node_ind += 1
            if n.op_type == "Quant":
                # Check that the node is in the activation path
                inp = model.get_initializer(n.input[0])
                out = model.get_initializer(n.output[0])
                if not (inp is None and out is None):
                    continue
                predecessor = model.find_direct_predecessors(n)
                if predecessor is not None:
                    predecessor_op_type = predecessor[0].op_type
                else:
                    predecessor_op_type = predecessor
                if model.is_fork_node(n):
                    raise ValueError(
                        "Forking Quant nodes are not currently supported by FINN."
                    )
                if not model.get_initializer(n.input[2]) == 0:
                    raise ValueError(
                        "Only Quant nodes with zero-point == 0 are currently supported."
                    )

                # Check for possible ambiguity in handler selection
                valid_predecessors = []
                for cls in QuantActBaseHandler.__subclasses__():
                    valid_predecessors.extend(cls.valid_predecessor_op_types)
                if len(valid_predecessors) != len(set(valid_predecessors)):
                    raise RuntimeError(
                        "Two or more activation handlers declare the same "
                        "type of valid predecessor node. "
                        "This leads to ambiguity in the handler selection "
                        "and must thus be avoided."
                    )

                # Try to find a fitting handler for this Quant activation node
                for handler_cls in QuantActBaseHandler.__subclasses__():
                    if predecessor_op_type in handler_cls.valid_predecessor_op_types:
                        handler = handler_cls(model, n, node_ind)
                        break
                else:
                    raise ValueError(
                        f"Quant nodes in the activation path and with predecessor "
                        f"nodes of type {predecessor_op_type} are currently not "
                        f"supported by FINN and can not be converted to "
                        f"MultiThreshold nodes."
                    )
                model = handler.replace_quant_node()
                graph_modified = True
                return (model, graph_modified)

        return (model, graph_modified)
