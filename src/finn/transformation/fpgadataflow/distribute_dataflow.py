import numpy as np

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.create_generic_partitions import PartitionFromLambda
from qonnx.util.basic import get_by_name

from finn.util.basic import make_build_dir

from finnexperimental.analysis.partitioning import partition

from IPython.core.debugger import set_trace

class DistributeDataflow(Transformation):
    def __init__(self, target_clk_ns, target_platform, ndevices):
        self.target_clk_ns = target_clk_ns
        self.target_platform = target_platform
        self.ndevices = ndevices

        self.partition_model_dir = make_build_dir("distributed_partitions_")

    def apply(self, model):
        child_nodes = model.get_nodes_by_op_type("StreamingDataflowPartition")
        assert len(child_nodes) == 1

        child_node = child_nodes[0]
        child_node_inst = getCustomOp(child_node)

        child_model = ModelWrapper(child_node_inst.get_nodeattr("model"))

        # TODO: assert that the child model only contains dataflow nodes

        floorplans = partition(
            child_model,
            self.target_clk_ns,
            self.target_platform,
            self.ndevices,
            # TODO: Make sure we are using multiple devices
            abs_anchors=[(0, [3]), (1, [7])]
        )

        if floorplans is None:
            raise Exception("Partitioning failed")

        floorplan = floorplans[0]

        def assign_partition_id(node):
            if node.op_type in ["GenericPartition", "StreamingDataflowPartition"]:
                return -1

            return floorplan[node.name]["device_id"]

        distr_model = child_model.transform(
            PartitionFromLambda(
                partitioning=assign_partition_id, partition_dir=self.partition_model_dir
            )
        )

        p_nodes = distr_model.get_nodes_by_op_type("GenericPartition")

        for partition_ind, p_node in enumerate(p_nodes):
            # done, change node type and add info in parent graph
            p_node.op_type = "StreamingDataflowPartition"
            p_node.domain = "finn.custom_op.fpgadataflow"
            new_p_node_inst = getCustomOp(p_node)
            new_p_node_inst.set_nodeattr("partition_id", partition_ind)

        child_node.op_type = "DistributedDataflow"
        new_child_node_inst = getCustomOp(child_node)
        new_child_node_inst.set_nodeattr("world_size", len(p_nodes))

        distr_model_file = self.partition_model_dir + "/distributed_dataflow.onnx"
        distr_model.save(distr_model_file)
        new_child_node_inst.set_nodeattr("model", distr_model_file)

        return (model, False)
