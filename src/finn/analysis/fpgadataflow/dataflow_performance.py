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

import numpy as np
from qonnx.custom_op.registry import getCustomOp

from finn.util.fpgadataflow import is_hls_node, is_rtl_node

FINNLOOP_ITERATION_OVERHEAD_CYCLES = 40


def stream_transactions(inst):
    """Return the number of output transactions for one input frame."""

    folded_shape = inst.get_folded_output_shape()
    return max(1, int(np.prod(folded_shape[:-1])))


def node_throughput_cycles(inst):
    """Return initiation cycles without latency-only buffering costs."""

    if inst.onnx_node.op_type in ["InnerShuffle_rtl", "OuterShuffle_hls", "Shuffle"]:
        return stream_transactions(inst)
    return max(1, int(inst.get_exp_cycles()))


def node_frame_boundary_cycles(inst):
    """Return recurring frame-boundary stall cycles for OuterShuffle_hls.

    A shallow FIFO cannot hide its circular-buffer phase change. InnerShuffle_rtl
    is double-buffered and does not require this correction.
    """

    if inst.onnx_node.op_type == "OuterShuffle_hls":
        return max(0, int(inst.get_exp_cycles()) - stream_transactions(inst))
    return 0


def accumulate_frame_boundary_cycles(predecessor_state, boundary_cycles):
    """Accumulate frame-boundary stalls along a stream path.

    The first stall occurs every frame. Later sequential shuffles alternate
    phases, so they contribute half their stall cycles, rounded up.
    """

    accumulated_cycles, has_boundary = predecessor_state
    boundary_cycles = int(boundary_cycles)
    if boundary_cycles <= 0:
        return predecessor_state
    if has_boundary:
        boundary_cycles = (boundary_cycles + 1) // 2
    return accumulated_cycles + boundary_cycles, True


def folding_performance(model, path=()):
    """Estimate steady-state initiation cycles, including nested FINNLoops.

    Each external frame consumes one body service interval per loop iteration,
    plus the FINNLoop controller overhead. Recurring OuterShuffle stalls are
    accumulated along each stream path using their two-phase average.
    """

    max_cycles = 0
    max_node_name = ""
    boundary_state_at_output = {}
    for node in model.graph.node:
        if node.op_type == "FINNLoop":
            inst = getCustomOp(node)
            body = inst.get_nodeattr("body")
            body_perf = folding_performance(body, path + (node.name,))
            iteration = max(1, int(inst.get_nodeattr("iteration")))
            node_cycles = iteration * (body_perf["max_cycles"] + FINNLOOP_ITERATION_OVERHEAD_CYCLES)
            # The body estimate already includes its frame-boundary stalls.
            path_boundary_state = (0, False)
        elif is_hls_node(node) or is_rtl_node(node) or node.op_type == "Shuffle":
            inst = getCustomOp(node)
            predecessor_states = [
                boundary_state_at_output.get(input_name, (0, False)) for input_name in node.input
            ]
            boundary_cycles = node_frame_boundary_cycles(inst)
            candidate_states = [
                accumulate_frame_boundary_cycles(state, boundary_cycles)
                for state in predecessor_states
            ] or [accumulate_frame_boundary_cycles((0, False), boundary_cycles)]
            path_boundary_state = max(candidate_states, key=lambda state: state[0])
            node_cycles = node_throughput_cycles(inst) + path_boundary_state[0]
        else:
            continue
        for output_name in node.output:
            boundary_state_at_output[output_name] = path_boundary_state
        if node_cycles > max_cycles:
            max_cycles = node_cycles
            max_node_name = "/".join(path + (node.name,))
    return {"max_cycles": max_cycles, "max_cycles_node_name": max_node_name}


def dataflow_performance(model):
    """Extract key performance indicators from given model with dataflow nodes.
    Note that the latency (critical path) analysis is very pessimistic, it
    assumes no overlap between executions and simply sums the expected cycles
    for each node along the critical path.

    Preconditions:
    - model consists of HLS/RTL nodes, exception are Shuffle nodes
    they do not need to be specialized yet
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
        if is_hls_node(node) or is_rtl_node(node) or node.op_type == "Shuffle":
            inst = getCustomOp(node)
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
