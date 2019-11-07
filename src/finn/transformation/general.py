import finn.core.utils as util


def give_unique_node_names(model):
    """Give unique names to each node in the graph using enumeration."""
    optype_count = {}
    for n in model.graph.node:
        if n.op_type not in optype_count.keys():
            optype_count[n.op_type] = 0
        n.name = "%s_%d" % (n.op_type, optype_count[n.op_type])
        optype_count[n.op_type] += 1
    # return model_was_changed = False as single iteration is always enough
    return (model, False)


def give_random_tensor_names(model):
    """Give random tensor names to all tensors."""
    names = model.get_all_tensor_names()
    for name in names:
        model.rename_tensor(name, util.random_string())
    # return model_was_changed = False as single iteration is always enough
    return (model, False)


def give_readable_tensor_names(model):
    """Give more human-readable names to all internal tensors. It's recommended
    to apply give_unique_node_names prior to this transform."""
    # to ensure we can use rename_tensor safely (without renaming existing
    # tensors) we start by giving random names to all tensors
    model = model.transform_single(give_random_tensor_names)
    graph = model.graph
    for n in graph.node:
        out_num = 0
        for o in n.output:
            model.rename_tensor(o, "%s_out%d" % (n.name, out_num))
            out_num += 1
        init_in_num = 0
        for i in n.input:
            if model.get_initializer(i) is not None:
                model.rename_tensor(i, "%s_param%d" % (n.name, init_in_num))
                init_in_num += 1
    # give special names to the main model input and output
    model.rename_tensor(model.graph.input[0].name, "global_in")
    model.rename_tensor(model.graph.output[0].name, "global_out")
    # return model_was_changed = False as single iteration is always enough
    return (model, False)


def convert_sub_to_add(model):
    """Convert sub nodes to add nodes of appropriate sign."""
    graph = model.graph
    for n in graph.node:
        if n.op_type == "Sub":
            A = model.get_initializer(n.input[1])
            if A is not None:
                n.op_type = "Add"
                model.set_initializer(n.input[1], -A)
    # return model_was_changed = False as single iteration is always enough
    return (model, False)
