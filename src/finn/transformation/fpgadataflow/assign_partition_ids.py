from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation

from finnexperimental.analysis.partitioning import partition

class AssignPartitionIDs(Transformation):
    def __init__(self, target_clk_ns, target_platform, ndevices):
        self.target_clk_ns = target_clk_ns
        self.target_platform = target_platform
        self.ndevices = ndevices

    def apply(self, model):
        floorplans = partition(
            model,
            self.target_clk_ns,
            self.target_platform,
            self.ndevices,
            # TODO: Remove this after testing
            abs_anchors=[(0, [3]), (1, [7])]
        )

        if floorplans is None:
            raise Exception("Partitioning failed")

        floorplan = floorplans[0]

        model.set_metadata_prop("world_size", str(self.ndevices))

        for node in model.graph.node:
            node_inst = getCustomOp(node)
            node_inst.set_nodeattr("device_id", floorplan[node.name]["device_id"])

        return model, False

