from finn.custom_op import CustomOp


class Im2Col(CustomOp):
    def get_nodeattr_types(self):
        return {
            "stride": ("i", True, 1),
            "kernel_size": ("i", True, 1),
        }

    def make_shape_compatible_op(self):
        pass

    def infer_node_datatype(self, model):
        pass

    def execute_node(self, context, graph):
        pass

    def verify_node(self):
        pass
