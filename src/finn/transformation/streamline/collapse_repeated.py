from onnx import helper as oh

from finn.transformation import Transformation
from finn.transformation.infer_shapes import InferShapes


class CollapseRepeatedOp(Transformation):
    """Collapse repeated consecutive operations with constant parameters into
    a single operation. make_collapsed_param_fxn must take two tensors and
    return a tensor which gives the equivalent result using a single op. """

    def __init__(self, op_name, make_collapsed_param_fxn):
        super().__init__()
        self.op_name = op_name
        self.make_collapsed_param_fxn = make_collapsed_param_fxn

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == self.op_name:
                consumer = model.find_consumer(n.output[0])
                if consumer is not None and consumer.op_type == self.op_name:
                    op0_param_name = n.input[1]
                    op1_param_name = consumer.input[1]
                    op0_param = model.get_initializer(op0_param_name)
                    op1_param = model.get_initializer(op1_param_name)
                    assert op0_param is not None
                    assert op1_param is not None
                    start_name = n.input[0]
                    end_name = consumer.output[0]
                    # compute the new parameter
                    new_param = self.make_collapsed_param_fxn(op0_param, op1_param)
                    # make and insert new node
                    new_node_param_name = op0_param_name
                    new_node = oh.make_node(
                        self.op_name, [start_name, new_node_param_name], [end_name]
                    )
                    graph.node.insert(node_ind, new_node)
                    # replace parameter value
                    model.set_initializer(new_node_param_name, new_param)
                    # remove old nodes
                    graph.node.remove(n)
                    graph.node.remove(consumer)
                    graph_modified = True
        model = model.transform(InferShapes())
        return (model, graph_modified)


class CollapseRepeatedAdd(CollapseRepeatedOp):
    """Collapse repeated adder node into a single operation."""
    def __init__(self):
        super().__init__("Add", lambda x, y: y + x)


class CollapseRepeatedMul(CollapseRepeatedOp):
    """Collapse repeated multiplier node into a single operation."""
    def __init__(self):
        super().__init__("Mul", lambda x, y: y * x)
