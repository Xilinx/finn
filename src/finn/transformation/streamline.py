import finn.transformation.infer_shapes as si


def move_add_past_mul(model):
    """Move add operations past multiply operations. The aim is to have them
    next to each other such that they can be collapsed into a single add."""
    graph = model.graph
    nodes_to_remove = []
    node_ind = 0
    graph_modified = False
    for n in graph.node:
        node_ind += 1
        if n.op_type == "Add":
            consumer = model.find_consumer(n.output[0])
            if consumer.op_type == "Mul":
                # assume input 0 is from the previous layer, input 1 is the
                # trained (constant) parameter
                mul_param = model.get_initializer(consumer.input[1])
                add_param = model.get_initializer(n.input[1])
                assert mul_param is not None
                assert add_param is not None
                # TODO compute new param values
                # TODO make new nodes
                # TODO mark nodes for removal
    # delete marked nodes (batchnorm and (un)squeezing)
    for n in nodes_to_remove:
        graph.node.remove(n)
        graph_modified = True
    model = model.transform_single(si.infer_shapes)
    return (model, graph_modified)
