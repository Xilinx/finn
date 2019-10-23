from onnx import helper as oh

import finn.transformation.infer_shapes as si


def collapse_repeated_op(model, op_name, make_collapsed_param_fxn):
    """Collapse repeated consecutive operations with constant parameters into
    a single operation. make_collapsed_param_fxn must take two tensors and
    return a tensor which gives the equivalent result using a single op. """
    graph = model.graph
    nodes_to_remove = []
    node_ind = 0
    graph_modified = False
    for n in graph.node:
        node_ind += 1
        if n.op_type == op_name:
            consumer = model.find_consumer(n.output[0])
            if consumer is not None and consumer.op_type == op_name:
                op0_param_name = n.input[1]
                op1_param_name = consumer.input[1]
                op0_param = model.get_initializer(op0_param_name)
                op1_param = model.get_initializer(op1_param_name)
                assert op0_param is not None
                assert op1_param is not None
                start_name = n.input[0]
                end_name = consumer.output[0]
                # compute the new parameter
                new_param = make_collapsed_param_fxn(op0_param, op1_param)
                # make and insert new node
                new_node_param_name = model.make_new_valueinfo_name()
                new_node = oh.make_node(
                    op_name, [start_name, new_node_param_name], [end_name]
                )
                graph.node.insert(node_ind, new_node)
                # replace parameter value
                model.set_initializer(new_node_param_name, new_param)
                # mark old nodes for removal
                nodes_to_remove += [n, consumer]
                graph_modified = True
    # delete marked nodes
    for n in nodes_to_remove:
        graph.node.remove(n)
        graph_modified = True
    model = model.transform_single(si.infer_shapes)
    return (model, graph_modified)


def collapse_repeated_add(model):
    return collapse_repeated_op(model, "Add", lambda x, y: y + x)


def collapse_repeated_mul(model):
    return collapse_repeated_op(model, "Mul", lambda x, y: y * x)


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
            if consumer is not None and consumer.op_type == "Mul":
                # have: (x) -> add(,B) -> (x+B) -> mul(,A) -> (xA+BA)
                # want: (x) -> mul(,A) -> (xA) -> add(,BA) -> (xA+BA)
                # assume input 0 is from the previous layer, input 1 is the
                # trained (constant) parameter
                mul_weight_name = consumer.input[1]
                add_weight_name = n.input[1]
                A = model.get_initializer(mul_weight_name)
                B = model.get_initializer(add_weight_name)
                assert A is not None
                assert B is not None
                start_name = n.input[0]
                middle_name = n.output[0]
                end_name = consumer.output[0]
                # compute new param value for add
                BA = B * A
                # make and insert new nodes
                new_mul = oh.make_node(
                    "Mul", [start_name, mul_weight_name], [middle_name]
                )
                new_add = oh.make_node(
                    "Add", [middle_name, add_weight_name], [end_name]
                )
                graph.node.insert(node_ind, new_mul)
                graph.node.insert(node_ind + 1, new_add)
                # replace add value
                model.set_initializer(add_weight_name, BA)
                # mark old nodes for removal
                nodes_to_remove += [n, consumer]
                graph_modified = True
    # delete marked nodes
    for n in nodes_to_remove:
        graph.node.remove(n)
        graph_modified = True
    model = model.transform_single(si.infer_shapes)
    return (model, graph_modified)
