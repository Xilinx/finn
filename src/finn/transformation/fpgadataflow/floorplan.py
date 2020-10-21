# Copyright (c) 2020, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from finn.custom_op.registry import getCustomOp
from finn.transformation.base import Transformation
from finn.util.basic import get_by_name
from finn.analysis.floorplan_params import floorplan_params
import warnings
import json


class Floorplan(Transformation):
    """Perform Floorplanning of the dataflow design:
    
    -Separate DMAs into their own partitions IDs, 
    -If not explicitly assigned, assign DWCs to SLRs to minimize SLLs required
    -If not explicitly assigned, assign FIFOs to the SLR of the upstream node
    -TODO: split the design into sections of defined size"""

    def __init__(self, floorplan=None):
        super().__init__()
        self.user_floorplan = floorplan

    def apply(self, model):

        # read in a user-specified floorplan or generate a default one
        if self.user_floorplan is None:
            floorplan = model.analysis(floorplan_params)
            json_dir = make_build_dir(prefix="vitis_floorplan_")
            json_file = json_dir + "/floorplan.json"
            model.set_metadata_prop("floorplan_json", json_file)
            with open(json_file, "w") as f:
                json.dump(floorplan, f, indent=4)
        else:
            model.set_metadata_prop("floorplan_json", self.user_floorplan)
            model = model.transform(ApplyConfig(self.user_floorplan))

        # perform DWC and FIFO specific adjustments
        unassigned_nodes = 0
        for node in model.graph.node:
            node_inst = getCustomOp(node)
            node_slr = node_inst.get_nodeattr("slr")
            if node_slr is None or node_slr == -1:
                node_inst.set_nodeattr("slr", -1)
                node_slr = -1
                unassigned_nodes += 1
            if node.op_type == "StreamingDataWidthConverter_Batch":
                # if we have SLR assignment already. use that
                if node_slr != -1:
                    continue
                # optimize for possible SLR crossing
                in_width = node_inst.get_nodeattr("inWidth")
                out_width = node_inst.get_nodeattr("outWidth")
                # find neighbour with narrowest bus
                if in_width > out_width:
                    narrow_neighbour = model.find_consumer(node.output[0])
                else:
                    narrow_neighbour = model.find_producer(node.input[0])
                    
                node_slr = narrow_neighbour.get_nodeattr("slr")
                node_inst.set_nodeattr("slr", node_slr)
            if node.op_type == "StreamingFIFO":
                # if we have SLR assignment already. use that
                if node_slr != -1:
                    continue
                srcnode = model.find_producer(node.input[0])
                node_slr = srcnode.get_nodeattr("slr")
                node_inst.set_nodeattr("slr", node_slr)


        if unassigned_nodes > 0:
            warnings.warn(
                str(unassigned_nodes)
                + " nodes have no entry in the provided floorplan "
                + "and no default value was set"
            )

        # save the updated floorplan
        floorplan = model.analysis(floorplan_params)
        with open(model.get_metadata_prop("floorplan_json"), "w") as f:
            json.dump(floorplan, f, indent=4)


        # partition id generation
        partition_cnt = 0

        # Assign IODMAs to their own partitions
        dma_nodes = list(filter(lambda x: x.op_type == "IODMA", df_nodes))
        non_dma_nodes = list(filter(lambda x: x not in dma_nodes, df_nodes))
        dyn_tlastmarker_nodes = list(
            filter(
                lambda x: x.op_type == "TLastMarker"
                and getCustomOp(x).get_nodeattr("DynIters") == "true",
                non_dma_nodes,
            )
        )
        non_dma_nodes = list(
            filter(lambda x: x not in dyn_tlastmarker_nodes, non_dma_nodes)
        )

        for node in dma_nodes:
            node_inst = getCustomOp(node)
            node_inst.set_nodeattr("partition_id", partition_cnt)
            partition_cnt += 1

        for node in dyn_tlastmarker_nodes:
            node_inst = getCustomOp(node)
            node_inst.set_nodeattr("partition_id", partition_cnt)
            partition_cnt += 1

        for node in non_dma_nodes:
            pre_node = model.find_producer(node.input[0])
            node_inst = getCustomOp(node)
            if pre_node not in non_dma_nodes:
                # input node
                node_inst.set_nodeattr("partition_id", partition_cnt)
                partition_cnt += 1
                continue
            elif not (
                node.op_type == "StreamingFCLayer_Batch"
                and node_inst.get_nodeattr("mem_mode") is not None
                and node_inst.get_nodeattr("mem_mode") == "external"
            ):
                pre_nodes = model.find_direct_predecessors(node)
            else:
                pre_nodes = [pre_node]

            node_slr = node_inst.get_nodeattr("slr")
            for pre_node in pre_nodes:
                pre_inst = getCustomOp(pre_node)
                pre_slr = pre_inst.get_nodeattr("slr")
                if node_slr == pre_slr:
                    partition_id = pre_inst.get_nodeattr("partition_id")
                    node_inst.set_nodeattr("partition_id", partition_id)
                    break
            else:
                # no matching, new partition
                node_inst.set_nodeattr("partition_id", partition_cnt)
                partition_cnt += 1

        return (model, False)
