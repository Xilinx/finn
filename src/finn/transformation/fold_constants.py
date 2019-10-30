import finn.core.onnx_exec as oxe


def fold_constants(model):
    """Replace the output of a node with const-only inputs with a precomputed
    result."""
    graph = model.graph
    node_ind = 0
    graph_modified = False
    execution_context = model.make_empty_exec_context()
    for n in graph.node:
        node_ind += 1
        node_inp_inits = list(map(lambda x: model.get_initializer(x), n.input))
        node_inp_dyn = list(filter(lambda x: x is None, node_inp_inits))
        node_out = n.output[0]
        if len(node_inp_dyn) == 0:
            # this node has no dynamic inputs, only constant ones -- so we can
            # do constant folding.
            oxe.execute_node(n, execution_context, graph)
            # use the execution result as an initializer
            model.set_initializer(node_out, execution_context[node_out])
            # remove old node
            graph.node.remove(n)
            graph_modified = True
    # TODO remove unused tensors?
    return (model, graph_modified)
