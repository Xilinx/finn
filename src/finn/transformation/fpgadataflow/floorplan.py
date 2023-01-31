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

import json
import warnings
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import ApplyConfig
from qonnx.util.basic import get_by_name

from finn.analysis.fpgadataflow.floorplan_params import floorplan_params
from finn.util.basic import make_build_dir


class Floorplan(Transformation):
    """Perform Floorplanning of the dataflow design:

    floorplan: path to a JSON containing a dictionary with SLR assignments
               for each node in the ONNX graph. Must be parse-able by
               the ApplyConfig transform.

    The transform applies the properties in the supplied JSON then:
    -Separates DMAs into their own partitions IDs,
    -If not explicitly assigned, assigns DWCs to SLRs to minimize SLLs required
    -If not explicitly assigned, assigns FIFOs to the SLR of the upstream node

    """

    def __init__(self, floorplan=None):
        super().__init__()
        self.user_floorplan = floorplan

    def apply(self, model):

        # read in a user-specified floorplan or generate a default one
        if self.user_floorplan is None:
            self.user_floorplan = model.analysis(floorplan_params)
            json_dir = make_build_dir(prefix="vitis_floorplan_")
            json_file = json_dir + "/floorplan.json"
            model.set_metadata_prop("floorplan_json", json_file)
            with open(json_file, "w") as f:
                json.dump(self.user_floorplan, f, indent=4)
        else:
            model.set_metadata_prop("floorplan_json", self.user_floorplan)
            model = model.transform(ApplyConfig(self.user_floorplan))

        try:
            default_slr = self.user_floorplan["Defaults"]["slr"][0]
        except Exception:
            default_slr = -1

        # perform DWC and FIFO specific adjustments
        unassigned_nodes = 0
        for node in model.graph.node:
            node_inst = getCustomOp(node)
            node_slr = node_inst.get_nodeattr("slr")
            if node_slr == -1:
                unassigned_nodes += 1
                node_inst.set_nodeattr("slr", default_slr)
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
                node_slr = getCustomOp(narrow_neighbour).get_nodeattr("slr")
                node_inst.set_nodeattr("slr", node_slr)
            if node.op_type == "StreamingFIFO":
                # if we have SLR assignment already. use that
                if node_slr != -1:
                    continue
                srcnode = model.find_producer(node.input[0])
                node_slr = getCustomOp(srcnode).get_nodeattr("slr")
                node_inst.set_nodeattr("slr", node_slr)

        if unassigned_nodes > 0:
            warnings.warn(
                str(unassigned_nodes)
                + " nodes have no entry in the provided floorplan,"
                + " SLR was set to "
                + str(default_slr)
            )

        # partition id generation
        partition_cnt = 0

        # Assign IODMAs to their own partitions
        all_nodes = list(model.graph.node)
        df_nodes = list(
            filter(lambda x: get_by_name(x.attribute, "backend") is not None, all_nodes)
        )
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
                node.op_type == "MatrixVectorActivation"
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
                    axilite_intf_name = pre_inst.get_verilog_top_module_intf_names()[
                        "axilite"
                    ]
                    if len(axilite_intf_name) != 0:
                        node_inst.set_nodeattr("partition_id", partition_cnt)
                        partition_cnt += 1
                    else:
                        partition_id = pre_inst.get_nodeattr("partition_id")
                        node_inst.set_nodeattr("partition_id", partition_id)
                break

            else:
                # no matching, new partition
                node_inst.set_nodeattr("partition_id", partition_cnt)
                partition_cnt += 1

        # save the updated floorplan
        floorplan = model.analysis(floorplan_params)
        with open(model.get_metadata_prop("floorplan_json"), "w") as f:
            json.dump(floorplan, f, indent=4)

        return (model, False)
