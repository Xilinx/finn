import numpy as np
from onnx import helper as oh

import finn.transformation.infer_shapes as si
from finn.core.datatype import DataType


def convert_sign_to_thres(model):
    """Convert Sign node instances to MultiThreshold with threshold at 0."""
    graph = model.graph
    graph_modified = False
    node_ind = 0
    for n in graph.node:
        node_ind += 1
        if n.op_type == "Sign":
            sign_out_name = n.output[0]
            # find consumer
            consumer = model.find_consumer(sign_out_name)
            assert consumer is not None
            # change op type and create threshold
            n.op_type = "MultiThreshold"
            thres_param_name = model.make_new_valueinfo_name()
            thres_param = np.asarray([[0]], dtype=np.float32)
            n.input.append(thres_param_name)
            n.domain = "finn"
            model.set_initializer(thres_param_name, thres_param)
            # convert 0,1 -> -1,+1 with 2*x-1
            out_shape = model.get_tensor_shape(sign_out_name)
            # make a mul node
            # note how set_initializer or set_tensor_shape is called before
            # calling make_new_valueinfo_name again
            mul_param_name = model.make_new_valueinfo_name()
            model.set_initializer(mul_param_name, np.asarray([[2]], dtype=np.float32))
            mul_out_name = model.make_new_valueinfo_name()
            model.set_tensor_shape(mul_out_name, out_shape)
            mul_node = oh.make_node(
                "Mul", [sign_out_name, mul_param_name], [mul_out_name]
            )
            # make an add node
            add_param_name = model.make_new_valueinfo_name()
            model.set_initializer(add_param_name, np.asarray([[-1]], dtype=np.float32))
            add_out_name = model.make_new_valueinfo_name()
            model.set_tensor_shape(add_out_name, out_shape)
            add_node = oh.make_node(
                "Add", [mul_out_name, add_param_name], [add_out_name]
            )
            # add new nodes to graph at correct position
            graph.node.insert(node_ind, mul_node)
            graph.node.insert(node_ind + 1, add_node)
            # rewrite consumer's input
            consumer.input[0] = add_out_name
            # add quantization annotations
            model.set_tensor_datatype(sign_out_name, DataType.BINARY)
            model.set_tensor_datatype(mul_out_name, DataType.UINT2)
            model.set_tensor_datatype(add_out_name, DataType.BIPOLAR)
            graph_modified = True
    return (model, graph_modified)


def collapse_repeated_op(model, op_name, make_collapsed_param_fxn):
    """Collapse repeated consecutive operations with constant parameters into
    a single operation. make_collapsed_param_fxn must take two tensors and
    return a tensor which gives the equivalent result using a single op. """
    graph = model.graph
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
                new_node_param_name = op0_param_name
                new_node = oh.make_node(
                    op_name, [start_name, new_node_param_name], [end_name]
                )
                graph.node.insert(node_ind, new_node)
                # replace parameter value
                model.set_initializer(new_node_param_name, new_param)
                # remove old nodes
                graph.node.remove(n)
                graph.node.remove(consumer)
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
                # remove old nodes
                graph.node.remove(n)
                graph.node.remove(consumer)
                graph_modified = True
    model = model.transform_single(si.infer_shapes)
    return (model, graph_modified)


def move_scalar_mul_past_matmul(model):
    """Move scalar mul operations past matmul operations. We want to have muls
    next to each other such that they can be collapsed into a single mul."""
    graph = model.graph
    node_ind = 0
    graph_modified = False
    for n in graph.node:
        node_ind += 1
        if n.op_type == "Mul":
            consumer = model.find_consumer(n.output[0])
            if consumer is not None and consumer.op_type == "MatMul":
                mul_weight_name = n.input[1]
                matmul_weight_name = consumer.input[1]
                A = model.get_initializer(mul_weight_name)
                W = model.get_initializer(matmul_weight_name)
                assert A is not None
                assert W is not None
                start_name = n.input[0]
                middle_name = n.output[0]
                end_name = consumer.output[0]
                if all(x == 1 for x in A.shape):
                    # if the mul is scalar, we can simply swap the order of ops
                    # make and insert new nodes
                    new_matmul = oh.make_node(
                        "MatMul", [start_name, matmul_weight_name], [middle_name]
                    )
                    new_mul = oh.make_node(
                        "Mul", [middle_name, mul_weight_name], [end_name]
                    )
                    graph.node.insert(node_ind, new_matmul)
                    graph.node.insert(node_ind + 1, new_mul)
                    # remove old nodes
                    graph.node.remove(n)
                    graph.node.remove(consumer)
                    graph_modified = True
    model = model.transform_single(si.infer_shapes)
    return (model, graph_modified)


def move_scalar_add_past_matmul(model):
    """Move scalar add operations past matmul operations. We want to have adds
    next to each other such that they can be collapsed into a single add."""
    graph = model.graph
    node_ind = 0
    graph_modified = False
    for n in graph.node:
        node_ind += 1
        if n.op_type == "Add":
            consumer = model.find_consumer(n.output[0])
            if consumer is not None and consumer.op_type == "MatMul":
                add_weight_name = n.input[1]
                matmul_weight_name = consumer.input[1]
                A = model.get_initializer(add_weight_name)
                W = model.get_initializer(matmul_weight_name)
                assert A is not None
                assert W is not None
                start_name = n.input[0]
                middle_name = n.output[0]
                end_name = consumer.output[0]
                if all(x == 1 for x in A.shape):
                    # if the add is scalar, we can move it past the matmul
                    # by taking it past the matmul with a dot product
                    Anew = np.dot(A * np.ones(W.shape[0], dtype=np.float32), W)
                    # update the add weight
                    model.set_initializer(add_weight_name, Anew)
                    new_matmul = oh.make_node(
                        "MatMul", [start_name, matmul_weight_name], [middle_name]
                    )
                    new_add = oh.make_node(
                        "Add", [middle_name, add_weight_name], [end_name]
                    )
                    graph.node.insert(node_ind, new_matmul)
                    graph.node.insert(node_ind + 1, new_add)
                    # remove old nodes
                    graph.node.remove(n)
                    graph.node.remove(consumer)
                    graph_modified = True
    model = model.transform_single(si.infer_shapes)
    return (model, graph_modified)


def absorb_add_into_multi_threshold(model):
    """Absorb preceding Add ops into MultiThreshold by updating the threshold
    values."""
    graph = model.graph
    node_ind = 0
    graph_modified = False
    for n in graph.node:
        node_ind += 1
        if n.op_type == "Add":
            consumer = model.find_consumer(n.output[0])
            if consumer is not None and consumer.op_type == "MultiThreshold":
                add_weight_name = n.input[1]
                threshold_name = consumer.input[1]
                A = model.get_initializer(add_weight_name)
                T = model.get_initializer(threshold_name)
                assert A is not None
                assert T is not None
                start_name = n.input[0]
                # compute new thresholds and set initializer
                Tnew = T - A.reshape(-1, T.shape[1])
                model.set_initializer(threshold_name, Tnew)
                # wire add input directly to MultiThreshold
                consumer.input[0] = start_name
                # remove the add node
                graph.node.remove(n)
                graph_modified = True
    return (model, graph_modified)


def absorb_mul_into_multi_threshold(model):
    """Absorb preceding Mul ops into MultiThreshold by updating the threshold
    values. Only *positive* scalar/1D vectors can be absorbed."""
    graph = model.graph
    node_ind = 0
    graph_modified = False
    for n in graph.node:
        node_ind += 1
        if n.op_type == "Mul":
            mul_weight_name = n.input[1]
            A = model.get_initializer(mul_weight_name)
            assert A is not None
            is_signed = (A < 0).any()
            is_scalar = np.prod(A.shape) == 1
            is_1d = len(A.shape) == 2 and A.shape[0] == 1
            consumer = model.find_consumer(n.output[0])
            if consumer is not None and consumer.op_type == "MultiThreshold":
                if not is_signed and (is_1d or is_scalar):
                    threshold_name = consumer.input[1]
                    T = model.get_initializer(threshold_name)
                    assert T is not None
                    start_name = n.input[0]
                    # compute new thresholds and set initializer
                    Tnew = T / A.reshape(-1, T.shape[1])
                    # TODO: need to handle negative A values correctly; produce
                    # mul sign mask and merge into preceding matmul?
                    model.set_initializer(threshold_name, Tnew)
                    # wire add input directly to MultiThreshold
                    consumer.input[0] = start_name
                    # remove the mul node
                    graph.node.remove(n)
                    graph_modified = True
    return (model, graph_modified)
