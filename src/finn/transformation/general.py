import copy

import numpy as np
from onnx import TensorProto
from onnx import helper as oh
from onnx import numpy_helper as np_helper


def give_unique_names(model):
    """Give unique names to each node in the graph using enumeration."""
    new_model = copy.deepcopy(model)
    node_count = 0
    for n in new_model.graph.node:
        n.name = "%s_%d" % (n.op_type, node_count)
        node_count += 1
    return new_model


def set_initializer(model, tensor_name, tensor_value):
    """Set the initializer value for tensor with given name."""
    graph = model.graph
    # convert tensor_value (numpy array) into TensorProto w/ correct name
    tensor_init_proto = np_helper.from_array(tensor_value)
    tensor_init_proto.name = tensor_name
    # first, remove if an initializer already exists
    init_names = [x.name for x in graph.initializer]
    try:
        init_ind = init_names.index(tensor_name)
        init_old = graph.initializer[init_ind]
        graph.initializer.remove(init_old)
    except ValueError:
        pass
    # create and insert new initializer
    graph.initializer.append(tensor_init_proto)


def get_initializer(model, tensor_name, tensor_value):
    """Get the initializer value for tensor with given name, if any."""
    graph = model.graph
    # convert tensor_value (numpy array) into TensorProto w/ correct name
    tensor_init_proto = np_helper.from_array(tensor_value)
    tensor_init_proto.name = tensor_name
    init_names = [x.name for x in graph.initializer]
    try:
        init_ind = init_names.index(tensor_name)
        return np_helper.to_array(graph.initializer[init_ind])
    except ValueError:
        return None


def find_producer(model, tensor_name):
    """Find and return the node that produces the tensor with given name.
    Currently only works for linear graphs."""
    all_outputs = [x.output[0].name for x in model.graph.node]
    try:
        producer_ind = all_outputs.index(tensor_name)
        return model.graph.node[producer_ind]
    except ValueError:
        return None


def find_consumer(model, tensor_name):
    """Find and return the node that consumes the tensor with given name.
    Currently only works for linear graphs."""
    all_inputs = [x.input[0].name for x in model.graph.node]
    try:
        consumer_ind = all_inputs.index(tensor_name)
        return model.graph.node[consumer_ind]
    except ValueError:
        return None


def make_new_valueinfo_name(model):
    """Returns a name that can be used for a new value_info."""
    graph = model.graph
    names = [x.name for x in graph.value_info]
    names += [x.name for x in graph.input]
    names += [x.name for x in graph.output]
    candidate = str(len(names) + 1)
    while candidate in names:
        candidate = str(int(candidate) + 1)
    return candidate


def replace_batchnorm_with_affine(model):
    """Replaces any test-time BatchNorm layers with Mul-Add layers."""
    new_model = copy.deepcopy(model)
    graph = new_model.graph
    nodes_to_remove = []
    for n in graph.node:
        if n.op_type == "BatchNormalization":
            bn_input = n.input[0]
            bn_output = n.output[0]
            # extract batchnorm parameters as numpy arrays
            scale = get_initializer(n.input[1])
            bias = get_initializer(n.input[2])
            mean = get_initializer(n.input[3])
            variance = get_initializer(n.input[4])
            epsilon = 1e-5
            # find A and B to compute batchnorm as affine transpose Ax+B
            # TODO is a division by moving avg factor needed for variance?
            A = scale / np.sqrt(epsilon + variance)
            B = bias - (A * mean)
            nodes_to_remove += [n]
            # see if we have surrounding Unsqueeze/Squeeze nodes we can remove
            producer = find_producer(n)
            if producer is not None:
                if producer.op_type == "Unsqueeze":
                    bn_input = producer.input[0]
                    nodes_to_remove += [producer]
            consumer = find_consumer(n)
            if consumer is not None:
                if consumer.op_type == "Squeeze":
                    bn_output = consumer.output[0]
                    nodes_to_remove += [consumer]
            # create value_info and initializers for Mul and Add constants
            mul_const = oh.make_tensor_value_info(
                make_new_valueinfo_name(new_model), TensorProto.FLOAT, A.shape
            )
            graph.value_info.append(mul_const)
            set_initializer(new_model, mul_const.name, A)
            mul_output = oh.make_tensor_value_info(
                make_new_valueinfo_name(new_model), TensorProto.FLOAT, A.shape
            )
            graph.value_info.append(mul_output)
            add_const = oh.make_tensor_value_info(
                make_new_valueinfo_name(new_model), TensorProto.FLOAT, B.shape
            )
            graph.value_info.append(add_const)
            set_initializer(new_model, add_const.name, B)
            # create Mul and Add nodes to replace the batchnorm
            mul_node = oh.make_node(
                "Mul", [bn_input.name, mul_const.name], [mul_output.name]
            )
            add_node = oh.make_node(
                "Add", [mul_output.name, add_const.name], [bn_output.name]
            )
            graph.node.append(mul_node)
            graph.node.append(add_node)

    # delete marked nodes
    for n in nodes_to_remove:
        graph.node.remove(n)
    # TODO topologically sort nodes
    # TODO give new names, maybe run shape inference?
    return new_model
