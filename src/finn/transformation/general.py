import copy

import numpy as np
import onnx.shape_inference as si
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


def get_tensor_shape(model, tensor_name):
    """Returns the shape of tensor with given name, if it has ValueInfoProto."""
    graph = model.graph
    vi_names = [(x.name, x) for x in graph.input]
    vi_names += [(x.name, x) for x in graph.output]
    vi_names += [(x.name, x) for x in graph.value_info]
    try:
        vi_ind = [x[0] for x in vi_names].index(tensor_name)
        vi = vi_names[vi_ind][1]
        dims = [x.dim_value for x in vi.type.tensor_type.shape.dim]
        return dims
    except ValueError:
        return None


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


def get_initializer(model, tensor_name):
    """Get the initializer value for tensor with given name, if any."""
    graph = model.graph
    init_names = [x.name for x in graph.initializer]
    try:
        init_ind = init_names.index(tensor_name)
        return np_helper.to_array(graph.initializer[init_ind])
    except ValueError:
        return None


def find_producer(model, tensor_name):
    """Find and return the node that produces the tensor with given name.
    Currently only works for linear graphs."""
    all_outputs = [x.output[0] for x in model.graph.node]
    try:
        producer_ind = all_outputs.index(tensor_name)
        return model.graph.node[producer_ind]
    except ValueError:
        return None


def find_consumer(model, tensor_name):
    """Find and return the node that consumes the tensor with given name.
    Currently only works for linear graphs."""
    all_inputs = [x.input[0] for x in model.graph.node]
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
    new_model = si.infer_shapes(new_model)
    graph = new_model.graph
    nodes_to_remove = []
    node_ind = 0
    for n in graph.node:
        node_ind += 1
        if n.op_type == "BatchNormalization":
            bn_input = n.input[0]
            bn_output = n.output[0]
            # extract batchnorm parameters as numpy arrays
            scale = get_initializer(new_model, n.input[1])
            bias = get_initializer(new_model, n.input[2])
            mean = get_initializer(new_model, n.input[3])
            variance = get_initializer(new_model, n.input[4])
            epsilon = 1e-5
            # find A and B to compute batchnorm as affine transpose Ax+B
            # TODO is a division by moving avg factor needed for variance?
            A = scale / np.sqrt(epsilon + variance)
            B = bias - (A * mean)
            nodes_to_remove += [n]
            # see if we have surrounding Unsqueeze/Squeeze nodes we can remove
            producer = find_producer(new_model, bn_input)
            if producer is not None:
                if producer.op_type == "Unsqueeze":
                    bn_input = producer.input[0]
                    nodes_to_remove += [producer]
            consumer = find_consumer(new_model, bn_output)
            if consumer is not None:
                if consumer.op_type == "Squeeze":
                    bn_output = consumer.output[0]
                    nodes_to_remove += [consumer]
            data_shape = get_tensor_shape(new_model, bn_input)
            # create value_info and initializers for Mul and Add constants
            mul_const = oh.make_tensor_value_info(
                make_new_valueinfo_name(new_model), TensorProto.FLOAT, A.shape
            )
            graph.value_info.append(mul_const)
            set_initializer(new_model, mul_const.name, A)
            mul_output = oh.make_tensor_value_info(
                make_new_valueinfo_name(new_model), TensorProto.FLOAT, data_shape
            )
            graph.value_info.append(mul_output)
            add_const = oh.make_tensor_value_info(
                make_new_valueinfo_name(new_model), TensorProto.FLOAT, B.shape
            )
            graph.value_info.append(add_const)
            set_initializer(new_model, add_const.name, B)
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
