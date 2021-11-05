import warnings

from finn.custom_op.registry import getCustomOp
from finn.transformation.base import Transformation
from finn.util.basic import get_by_name, is_finn_op


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
    """Removes a flatten node if it is between two fpgadataflow nodes.
    For an NHWC-Conv to FC transition, the preceding transpose is absorbed.
    The flatten operation can also be implemented by a reshape node."""

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        for n in graph.node:
            # also support implicit flatten via reshape, e.g. reshape(1,-1)
            if n.op_type == "Flatten" or n.op_type == "Reshape":
                ishape = model.get_tensor_shape(n.input[0])
                oshape = model.get_tensor_shape(n.output[0])
                if len(oshape) == 2 and ishape[0] == oshape[0]:
                    producer = model.find_producer(n.input[0])
                    if _is_fpgadataflow_node(producer) is True:
                        # standalone flatten, remove
                        consumer = model.find_consumer(n.output[0])
                        if _is_fpgadataflow_node(consumer) is True:
                            graph_modified = True
                            consumer.input[0] = n.input[0]
                            graph.node.remove(n)
                    elif producer.op_type == "Transpose":
                        # transpose + flatten, absorb into following node
                        transp_node = producer
                        # check if transpose converts NHWC to NCHW
                        perms = list(get_by_name(transp_node.attribute, "perm").ints)
                        if perms == [0, 3, 1, 2]:
                            producer = model.find_producer(transp_node.input[0])
                            if _is_fpgadataflow_node(producer) is True:
                                consumer = model.find_consumer(n.output[0])
                                if consumer.op_type == "StreamingFCLayer_Batch":
                                    fc_inst = getCustomOp(consumer)
                                    mw = fc_inst.get_nodeattr("MW")
                                    mh = fc_inst.get_nodeattr("MH")
                                    (b, h, w, c) = model.get_tensor_shape(
                                        transp_node.input[0]
                                    )
                                    # absorb transpose into weight matrix,
                                    # allowing FC layer to operate on the NHWC input
                                    W = model.get_initializer(consumer.input[1])
                                    assert (
                                        W is not None
                                    ), "Initializer for matmul weights is not set."
                                    W_new = W.reshape(c, h, w, mh)
                                    W_new = W_new.transpose((1, 2, 0, 3))
                                    W_new = W_new.reshape(mw, mh)
                                    model.set_initializer(consumer.input[1], W_new)
                                    # remove transpose & flatten nodes
                                    consumer.input[0] = transp_node.input[0]
                                    graph.node.remove(n)
                                    graph.node.remove(transp_node)
                                    graph_modified = True
                                else:
                                    warnings.warn(
                                        "Could not absorb transpose->flatten \
                                        into subsequent node"
                                    )
                        else:
                            warnings.warn(
                                "Unsupported transpose node before flatten layer"
                            )

        return (model, graph_modified)
