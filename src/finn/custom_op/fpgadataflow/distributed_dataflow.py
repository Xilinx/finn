from qonnx.custom_op.base import CustomOp

class DistributedDataflow(CustomOp):
    def get_nodeattr_types(self):
        return {
            "model": ("s", True, ""),
            "instance_name": ("s", False, ""),
            "return_full_exec_context": ("i", False, 0),
            "world_size": ("i", True, -1),
        }

    def make_shape_compatible_op(self, model):
        pass

    def infer_node_datatype(self, model):
        pass

    def verify_node(self):
        pass

    def execute_node():
        ...

