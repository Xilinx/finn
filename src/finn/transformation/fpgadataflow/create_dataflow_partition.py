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

from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.registry import getCustomOp
from finn.transformation.base import Transformation
from finn.transformation.create_generic_partitions import PartitionFromLambda
from finn.transformation.fpgadataflow.externalize_params import ExternalizeParams
from finn.util.basic import get_by_name


class CreateDataflowPartition(Transformation):
    """Split a graph into two graphs; one which contains non-FINN-dataflow nodes
    and a StreamingDataflowPartition node, and another which only contains
    FINN dataflow nodes. The StreamingDataflowPartition has a model attribute
    that indicates the filename for the second graph that only contains
    dataflow nodes. No action is taken if there are no dataflow nodes."""

    def __init__(self, partition_model_dir="dataflow_partition"):
        super().__init__()
        self.partition_model_dir = partition_model_dir

    def apply(self, model):
        def filter_fc_extw(x):
            if x.op_type == "IODMA":
                burst_mode = get_by_name(x.attribute, "burstMode")
                if burst_mode is not None:
                    burst_mode = burst_mode.s.decode("UTF-8")
                    return burst_mode == "wrap"

        extw_dma_nodes = list(filter(filter_fc_extw, model.graph.node))
        if len(extw_dma_nodes) > 0:
            model = model.transform(ExternalizeParams())

        def assign_partition_id(node):
            if node.op_type in ["GenericPartition", "StreamingDataflowPartition"]:
                return -1
            else:
                backend = get_by_name(node.attribute, "backend")
                if backend is not None and backend.s.decode("UTF-8") == "fpgadataflow":
                    assigned_partition = get_by_name(node.attribute, "partition_id")
                    if assigned_partition is not None:
                        return assigned_partition.i
                    else:
                        return 0
                else:
                    return -1

        # first, use the generic partitioning functionality to split up the graph
        parent_model = model.transform(
            PartitionFromLambda(
                partitioning=assign_partition_id, partition_dir=self.partition_model_dir
            )
        )
        # change node types to StreamingDataflowPartition
        p_nodes = parent_model.get_nodes_by_op_type("GenericPartition")
        for partition_ind, p_node in enumerate(p_nodes):
            # go into partition to extract some info
            p_node_inst = getCustomOp(p_node)
            node_model_filename = p_node_inst.get_nodeattr("model")
            p_model = ModelWrapper(node_model_filename)
            # check floorplan (SLR assignment per node)
            inst = getCustomOp(p_model.graph.node[0])
            slr = inst.get_nodeattr("slr")
            for node in p_model.graph.node:
                inst = getCustomOp(node)
                assert slr == inst.get_nodeattr(
                    "slr"
                ), """all nodes with same partition_id must have the same slr id"""
            # check that there is only one non-null mem_port per partition
            nmemports = 0
            mem_port = ""
            for node in p_model.graph.node:
                inst = getCustomOp(node)
                port = inst.get_nodeattr("mem_port")
                if port is not None and port != "":
                    nmemports += 1
                    mem_port = port
            assert nmemports <= 1, """Too many memory ports per partition"""
            # done, change node type and add info in parent graph
            p_node.op_type = "StreamingDataflowPartition"
            p_node.domain = "finn.custom_op.fpgadataflow"
            new_p_node_inst = getCustomOp(p_node)
            new_p_node_inst.set_nodeattr("partition_id", partition_ind)
            new_p_node_inst.set_nodeattr("slr", slr)
            new_p_node_inst.set_nodeattr("mem_port", mem_port)

        return (parent_model, False)
