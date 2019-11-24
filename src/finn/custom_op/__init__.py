from abc import ABC, abstractmethod


class CustomOp(ABC):
    def __init__(self, onnx_node):
        super().__init__()
        self.onnx_node = onnx_node
        # TODO consider specifying a list of allowed attributes

    @abstractmethod
    def make_shape_compatible_op(self):
        pass

    @abstractmethod
    def infer_node_datatype(self, model):
        pass

    @abstractmethod
    def execute_node(self, context, graph):
        pass
