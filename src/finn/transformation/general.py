import copy

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
