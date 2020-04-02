from finn.transformation import Transformation
from finn.transformation.infer_shapes import InferShapes


def _get_number_of_nodes(model):
    node_count = 0
    for n in model.graph.node:
        node_count += 1
    return node_count


class MoveReshape(Transformation):
    """Removes a node that implements a (1, -1) reshape and runs
    InferShapes on the model"""

    def apply(self, model):

        graph = model.graph
        graph_modified = False
        for n in graph.node:
            if n.op_type == "Reshape":
                graph_modified = True
                shape = model.get_initializer(n.input[1])
                if (shape == [1, -1]).all():
                    consumer = model.find_consumer(n.output[0])
                    if consumer is not None:
                        consumer.input[0] = n.input[0]
                    graph.node.remove(n)

        model = model.transform(InferShapes())
        return (model, graph_modified)
