from finn.transformation import Transformation
from finn.util.basic import get_by_name


def _is_fpgadataflow_node(node):
    if node is not None:
        if node.domain == "finn":
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


class MoveReshape(Transformation):
    """Removes a node that implements a (1, -1) reshape if it is
    between two fpgadataflow nodes"""

    def apply(self, model):

        graph = model.graph
        graph_modified = False
        for n in graph.node:
            if n.op_type == "Reshape":
                graph_modified = True
                shape = model.get_initializer(n.input[1])
                if (shape == [1, -1]).all():
                    producer = model.find_producer(n.input[0])
                    if _is_fpgadataflow_node(producer) is True:
                        consumer = model.find_consumer(n.output[0])
                        if _is_fpgadataflow_node(consumer) is True:
                            consumer.input[0] = n.input[0]
                            graph.node.remove(n)

        return (model, graph_modified)
