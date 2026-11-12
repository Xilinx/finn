# Copyright (c) 2020, Xilinx, Inc.
# Copyright (C) 2024, Advanced Micro Devices, Inc.
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

from qonnx.custom_op.registry import getCustomOp

from finn.util.fpgadataflow import is_hls_node, is_rtl_node


def dataflow_performance(model, use_characterized=False):
    """Extract key performance indicators from given model with dataflow nodes.
    Note that the latency (critical path) analysis is very pessimistic, it
    assumes no overlap between executions and simply sums the expected cycles
    for each node along the critical path.

    Preconditions:
    - model consists of HLS/RTL nodes, exception are Shuffle nodes
    they do not need to be specialized yet
    - model has cycle estimates annotated (see AnnotateCycles transformation)
    - nodes have unique names (see GiveUniqueNodeNames)

    Parameters:
    - use_characterized : take each node's cycle count from the period of its
      token access vectors (``io_chrc_period``, see DeriveTokenAccessVectors)
      instead of the cycles_estimate annotation. Nodes that carry no token
      access vectors -- fork and join nodes and FIFOs -- count as zero
      cycles, so max_cycles is the slowest *characterized* node. Note
      that this makes critical_path_cycles a partial sum; the characterized mode
      exists to report max_cycles.

    Returns:
    - max_cycles : number of cycles for slowest node
    - max_cycles_node_name : name of slowest node
    - critical_path_cycles : pessimistic expected latency from input to output
    """
    latency_at_node_output = {}
    max_cycles = 0
    max_node_name = ""

    for node in model.graph.node:
        if is_hls_node(node) or is_rtl_node(node) or node.op_type == "Shuffle":
            inst = getCustomOp(node)
            if use_characterized:
                # the period DeriveTokenAccessVectors stored with the TAVs; 0
                # for a node that has not been characterized
                node_cycles = inst.get_nodeattr("io_chrc_period")
            else:
                node_cycles = int(inst.get_nodeattr("cycles_estimate"))
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
                    pred_latencies = map(lambda x: latency_at_node_output[x.name], predecessors)
                    max_pred_latency = max(pred_latencies)
                latency_at_node_output[node.name] = node_cycles + max_pred_latency
    critical_path_cycles = max(latency_at_node_output.values())
    return {
        "critical_path_cycles": int(critical_path_cycles),
        "max_cycles": int(max_cycles),
        "max_cycles_node_name": max_node_name,
    }
