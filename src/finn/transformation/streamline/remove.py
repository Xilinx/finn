from finn.transformation import Transformation
from finn.transformation.infer_shapes import InferShapes


class RemoveIdentityOps(Transformation):
    """Remove identity ops like Add/Sub with zero or Mul/Div with one"""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if (
                n.op_type in ["Add", "Sub"]
                and not model.is_fork_node(n)
                and not model.is_join_node(n)
            ):
                A = model.get_initializer(n.input[1])
                # limit trafo to scalar and zeros
                if all(x == 1 for x in A.shape):
                    if A == 0:
                        producer = model.find_producer(n.input[0])
                        # remove node and wire output tensor to
                        # output of producer node
                        producer.output[0] = n.output[0]
                        graph.node.remove(n)

            elif (
                n.op_type in ["Mul", "Div"]
                and not model.is_fork_node(n)
                and not model.is_join_node(n)
            ):
                A = model.get_initializer(n.input[1])
                # limit trafo to scalar and ones
                if all(x == 1 for x in A.shape):
                    if A == 1:
                        producer = model.find_producer(n.input[0])
                        # remove node and wire output tensor to
                        # output of producer node
                        producer.output[0] = n.output[0]
                        graph.node.remove(n)
        model = model.transform(InferShapes())
        return (model, graph_modified)
