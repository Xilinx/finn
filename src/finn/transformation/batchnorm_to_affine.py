import copy

import numpy as np
import onnx.shape_inference as si
from onnx import TensorProto
from onnx import helper as oh

import finn.transformation.general as tg


def batchnorm_to_affine(model):
    """Replaces any test-time BatchNorm layers with Mul-Add layers."""
    new_model = copy.deepcopy(model)
    graph = new_model.graph
    nodes_to_remove = []
    node_ind = 0
    for n in graph.node:
        node_ind += 1
        if n.op_type == "BatchNormalization":
            bn_input = n.input[0]
            bn_output = n.output[0]
            # extract batchnorm parameters as numpy arrays
            scale = tg.get_initializer(new_model, n.input[1])
            bias = tg.get_initializer(new_model, n.input[2])
            mean = tg.get_initializer(new_model, n.input[3])
            variance = tg.get_initializer(new_model, n.input[4])
            epsilon = 1e-5
            # find A and B to compute batchnorm as affine transpose Ax+B
            # TODO is a division by moving avg factor needed for variance?
            A = scale / np.sqrt(epsilon + variance)
            B = bias - (A * mean)
            nodes_to_remove += [n]
            # see if we have surrounding Unsqueeze/Squeeze nodes we can remove
            producer = tg.find_producer(new_model, bn_input)
            if producer is not None:
                if producer.op_type == "Unsqueeze":
                    bn_input = producer.input[0]
                    nodes_to_remove += [producer]
            consumer = tg.find_consumer(new_model, bn_output)
            if consumer is not None:
                if consumer.op_type == "Squeeze":
                    bn_output = consumer.output[0]
                    nodes_to_remove += [consumer]
            data_shape = tg.get_tensor_shape(new_model, bn_input)
            # create value_info and initializers for Mul and Add constants
            mul_const = oh.make_tensor_value_info(
                tg.make_new_valueinfo_name(new_model), TensorProto.FLOAT, A.shape
            )
            graph.value_info.append(mul_const)
            tg.set_initializer(new_model, mul_const.name, A)
            mul_output = oh.make_tensor_value_info(
                tg.make_new_valueinfo_name(new_model), TensorProto.FLOAT, data_shape
            )
            graph.value_info.append(mul_output)
            add_const = oh.make_tensor_value_info(
                tg.make_new_valueinfo_name(new_model), TensorProto.FLOAT, B.shape
            )
            graph.value_info.append(add_const)
            tg.set_initializer(new_model, add_const.name, B)
            # create Mul and Add nodes to replace the batchnorm
            mul_node = oh.make_node(
                "Mul", [bn_input, mul_const.name], [mul_output.name]
            )
            add_node = oh.make_node(
                "Add", [mul_output.name, add_const.name], [bn_output]
            )
            # insert where the batchnorm is to preserve topological ordering
            graph.node.insert(node_ind, mul_node)
            graph.node.insert(node_ind + 1, add_node)
    # delete marked nodes (batchnorm and (un)squeezing)
    for n in nodes_to_remove:
        graph.node.remove(n)
    new_model = si.infer_shapes(new_model)
    return new_model
