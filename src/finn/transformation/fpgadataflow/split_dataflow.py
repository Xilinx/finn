import numpy as np

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.create_generic_partitions import PartitionFromLambda
from qonnx.util.basic import get_by_name

from finn.util.basic import make_build_dir

from finnexperimental.analysis.partitioning import partition

class SplitDataflow(Transformation):
    def __init__(self):
        self.partition_model_dir = make_build_dir('distributed_partitions_')

    def apply(self, model):
        def assign_partition_id(node):
            if node.op_type in ["GenericPartition", "StreamingDataflowPartition"]:
                return -1

            return getCustomOp(node).get_nodeattr("device_id")

        model = model.transform(
            PartitionFromLambda(
                partitioning=assign_partition_id, partition_dir=self.partition_model_dir
            )
        )

        p_nodes = model.get_nodes_by_op_type("GenericPartition")

        for partition_ind, p_node in enumerate(p_nodes):
            # done, change node type and add info in parent graph
            p_node.op_type = "StreamingDataflowPartition"
            p_node.domain = "finn.custom_op.fpgadataflow"
            new_p_node_inst = getCustomOp(p_node)
            new_p_node_inst.set_nodeattr("device_id", partition_ind)

        model.save('model.onnx')

        return (model, False)
