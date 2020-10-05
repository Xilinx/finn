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


class Floorplan(Transformation):
    """Perform Floorplanning of the dataflow design. Separate DMAs into their own
    partitions IDs, and TODO: split the design into sections of defined size"""

    def __init__(self, limits=None):
        super().__init__()
        self.resource_limits = limits

    def apply(self, model):
        target_partition_id = 0
        # we currently assume that all dataflow nodes belonging to the same partition
        # are connected to each other and there is a single input/output to/from each.
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
            node_inst.set_nodeattr("partition_id", target_partition_id)
            target_partition_id += 1

        for node in dyn_tlastmarker_nodes:
            node_inst = getCustomOp(node)
            node_inst.set_nodeattr("partition_id", target_partition_id)
            target_partition_id += 1

        for node in non_dma_nodes:
            # TODO: implement proper floorplanning; for now just a single partition
            node_inst = getCustomOp(node)
            node_inst.set_nodeattr("partition_id", target_partition_id)

        return (model, False)
