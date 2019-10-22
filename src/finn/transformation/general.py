import copy


def give_unique_node_names(model):
    """Give unique names to each node in the graph using enumeration."""
    new_model = copy.deepcopy(model)
    node_count = 0
    for n in new_model.graph.node:
        n.name = "%s_%d" % (n.op_type, node_count)
        node_count += 1
    # return model_was_changed = False as single iteration is always enough
    return (new_model, False)
