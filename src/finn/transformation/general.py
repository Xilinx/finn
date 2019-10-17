import copy


def give_unique_names(model):
    """Give unique names to each node in the graph using enumeration."""

    new_model = copy.deepcopy(model)
    node_count = 0
    for n in new_model.graph.node:
        n.name = "%s_%d" % (n.op_type, node_count)
        node_count += 1
    return new_model
