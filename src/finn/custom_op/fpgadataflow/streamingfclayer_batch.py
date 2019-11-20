from finn.core.utils import get_by_name
from finn.custom_op.fpgadataflow import HLSCustomOp


class StreamingFCLayer_Batch(HLSCustomOp):
    def make_shape_compatible_op(self, node):
        pass

    def infer_node_datatype(self, node, model):
        pass
