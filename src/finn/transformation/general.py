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
