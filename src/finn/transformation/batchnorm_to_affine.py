import numpy as np
from onnx import TensorProto
from onnx import helper as oh

from finn.transformation import Transformation
from finn.transformation.infer_shapes import InferShapes


class BatchNormToAffine(Transformation):
    """Replaces any test-time BatchNorm layers with Mul-Add layers."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "BatchNormalization":
                graph_modified = True
                bn_input = n.input[0]
                bn_output = n.output[0]
                # extract batchnorm parameters as numpy arrays
                scale = model.get_initializer(n.input[1])
                bias = model.get_initializer(n.input[2])
                mean = model.get_initializer(n.input[3])
                variance = model.get_initializer(n.input[4])
                epsilon = 1e-5
                # find A and B to compute batchnorm as affine transpose Ax+B
                # TODO is a division by moving avg factor needed for variance?
                A = scale / np.sqrt(epsilon + variance)
                B = bias - (A * mean)
                # see if we have surrounding Unsqueeze/Squeeze nodes we can remove
                producer = model.find_producer(bn_input)
                if producer is not None:
                    if producer.op_type == "Unsqueeze":
                        bn_input = producer.input[0]
                consumer = model.find_consumer(bn_output)
                if consumer is not None:
                    if consumer.op_type == "Squeeze":
                        bn_output = consumer.output[0]
                data_shape = model.get_tensor_shape(bn_input)
                # create value_info and initializers for Mul and Add constants
                mul_const = oh.make_tensor_value_info(
                    model.make_new_valueinfo_name(), TensorProto.FLOAT, A.shape
                )
                graph.value_info.append(mul_const)
                model.set_initializer(mul_const.name, A)
                mul_output = oh.make_tensor_value_info(
                    model.make_new_valueinfo_name(), TensorProto.FLOAT, data_shape
                )
                graph.value_info.append(mul_output)
                add_const = oh.make_tensor_value_info(
                    model.make_new_valueinfo_name(), TensorProto.FLOAT, B.shape
                )
                graph.value_info.append(add_const)
                model.set_initializer(add_const.name, B)
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
                # remove old nodes
                graph.node.remove(n)
                if consumer is not None:
                    if consumer.op_type == "Squeeze":
                        graph.node.remove(consumer)
                if producer is not None:
                    if producer.op_type == "Unsqueeze":
                        graph.node.remove(producer)
        model = model.transform(InferShapes())
        return (model, graph_modified)
