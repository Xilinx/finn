from finn.transformation.base import Transformation
from finn.util.basic import get_by_name, is_finn_op
from finn.custom_op.registry import getCustomOp


def _is_fpgadataflow_node(node):
    if node is not None:
        if is_finn_op(node.domain):
            n_backend = get_by_name(node.attribute, "backend")
            if n_backend is None:
                return False
            backend_value = n_backend.s.decode("UTF-8")
            if backend_value == "fpgadataflow":
                return True
        else:
            return False
    else:
        return False


class RemoveCNVtoFCFlatten(Transformation):
    """Removes a node that implements a (1, -1) reshape if it is
    between two fpgadataflow nodes"""

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        for n in graph.node:
            if n.op_type == "Flatten":  # re-add reshape
                # shape = model.get_initializer(n.input[1])
                # if (shape == [1, -1]).all():
                producer = model.find_producer(n.input[0])
                if _is_fpgadataflow_node(producer) is True:
                    consumer = model.find_consumer(n.output[0])
                    if _is_fpgadataflow_node(consumer) is True:
                        graph_modified = True
                        consumer.input[0] = n.input[0]
                        graph.node.remove(n)
                elif producer.op_type == "Transpose":
                    transp_node = producer

                    # check if transpose converts NHWC to NCHW
                    perms = list(get_by_name(transp_node.attribute, "perm").ints)
                    if perms == [0, 3, 1, 2]:

                        producer = model.find_producer(transp_node.input[0])

                        if _is_fpgadataflow_node(producer) is True:
                            consumer = model.find_consumer(n.output[0])
                            if _is_fpgadataflow_node(consumer) is True:
                                if consumer.op_type == "StreamingFCLayer_Batch":
                                    fc_inst = getCustomOp(consumer)
                                    mw = fc_inst.get_nodeattr("MW")
                                    mh = fc_inst.get_nodeattr("MH")
                                    (b, h, w, c) = model.get_tensor_shape(
                                        transp_node.input[0]
                                    )
                                    # absorb transpose into weight matrix, allowing FC layer to operate on the NHWC input
                                    W = model.get_initializer(consumer.input[1])
                                    assert (
                                        W is not None
                                    ), "Initializer for matmul weights is not set."
                                    print("fc weights before")
                                    print(W.shape)
                                    print(W)

                                    W_new = W.reshape(c, h, w, mh)
                                    W_new = W_new.transpose((1, 2, 0, 3))
                                    W_new = W_new.reshape(mw, mh)

                                    print("fc weights after")
                                    print(W_new.shape)
                                    print(W_new)

                                    model.set_initializer(consumer.input[1], W_new)

                                    # remove transpose & flatten nodes
                                    graph_modified = True
                                    consumer.input[0] = transp_node.input[0]
                                    graph.node.remove(n)
                                    graph.node.remove(transp_node)
                                else:
                                    warnings.warn(
                                        "Could not absorb transpose into node behind flatten layer"
                                    )
                    else:
                        warnings.warn("Unsupported transpose node before flatten layer")

        return (model, graph_modified)
