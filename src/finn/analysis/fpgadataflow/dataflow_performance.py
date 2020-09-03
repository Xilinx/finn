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
from finn.util.fpgadataflow import is_fpgadataflow_node


def dataflow_performance(model):
    """Extract key performance indicators from given model with dataflow nodes.
    Note that the latency (critical path) analysis is very pessimistic, it
    assumes no overlap between executions and simply sums the expected cycles
    for each node along the critical path.

    Preconditions:
    - model consists of fpgadataflow nodes
    - model has cycle estimates annotated (see AnnotateCycles transformation)
    - nodes have unique names (see GiveUniqueNodeNames)

    Returns:
    - max_cycles : number of cycles for slowest node
    - max_cycles_node_name : name of slowest node
    - critical_path_cycles : pessimistic expected latency from input to output
    """
    latency_at_node_output = {}
    max_cycles = 0
    max_node_name = ""

    for node in model.graph.node:
        if is_fpgadataflow_node(node) is True:
            inst = getCustomOp(node)
            node_cycles = inst.get_nodeattr("cycles_estimate")
            if node_cycles > max_cycles:
                max_cycles = node_cycles
                max_node_name = node.name
            if node.name not in latency_at_node_output:
                # calculate based on input(s)
                predecessors = model.find_direct_predecessors(node)
                if predecessors is None:
                    # no predecessors, node is first node
                    max_pred_latency = 0
                else:
                    # find max of any of predecessors
                    pred_latencies = map(
                        lambda x: latency_at_node_output[x.name], predecessors
                    )
                    max_pred_latency = max(pred_latencies)
                latency_at_node_output[node.name] = node_cycles + max_pred_latency
    critical_path_cycles = max(latency_at_node_output.values())
    return {
        "critical_path_cycles": critical_path_cycles,
        "max_cycles": max_cycles,
        "max_cycles_node_name": max_node_name,
    }
