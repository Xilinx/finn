from abc import ABC, abstractmethod


class CustomOp(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def make_shape_compatible_op(self, node):
        pass

    @abstractmethod
    def infer_node_datatype(self, node, model):
        pass

    @abstractmethod
    def execute_node(self, node, context, graph):
        pass
