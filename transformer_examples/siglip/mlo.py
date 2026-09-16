# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""SigLIP vision-encoder loop detection for multi-level offload (MLO)."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from qonnx.custom_op.registry import getCustomOp
from typing import Any

from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds


def _is_layer_norm(node: Any) -> bool:
    return node.op_type == "LayerNormalization" or node.op_type.startswith("LayerNorm")


def find_vision_loop_body_ranges(model: Any, depth: int) -> list[dict[str, Any]]:
    """Find the repeated encoder blocks in a converted static SigLIP graph.

    Each vision block begins at every second LayerNorm. The LayerNorm following
    block ``depth - 1`` marks the end of the encoder stack. DuplicateStreams at
    block boundaries are kept on the producing side of an activation edge.
    """

    proto = model.model if hasattr(model, "model") else model
    nodes = list(proto.graph.node)
    layer_norm_indices = [index for index, node in enumerate(nodes) if _is_layer_norm(node)]
    if len(layer_norm_indices) < 2 * depth + 1:
        return []

    ranges = []
    for block_index in range(depth):
        start_index = layer_norm_indices[2 * block_index]
        start_node = nodes[start_index]
        if start_index > 0:
            previous = nodes[start_index - 1]
            if (
                previous.op_type.startswith("DuplicateStreams")
                and start_node.input
                and start_node.input[0] in previous.output
            ):
                start_index -= 1

        if block_index + 1 < depth:
            next_start_index = layer_norm_indices[2 * (block_index + 1)]
            end_index = next_start_index - 1
            end_node = nodes[end_index]
            next_start = nodes[next_start_index]
            if (
                end_node.op_type.startswith("DuplicateStreams")
                and next_start.input
                and next_start.input[0] in end_node.output
            ):
                end_index -= 1
        else:
            end_index = layer_norm_indices[2 * depth] - 1

        if end_index < start_index:
            raise RuntimeError(f"Invalid SigLIP loop-body range for block {block_index}")
        block_nodes = nodes[start_index : end_index + 1]
        op_types = [node.op_type for node in block_nodes]
        ranges.append(
            {
                "block": block_index,
                "start_index": start_index,
                "end_index": end_index,
                "start_node": nodes[start_index].name,
                "end_node": nodes[end_index].name,
                "node_count": len(block_nodes),
                "op_types": op_types,
                "op_counts": dict(Counter(op_types).most_common()),
            }
        )
    return ranges


def step_round_siglip_thresholds_before_mlo(model, cfg):
    """Preserve integer threshold types when repeated blocks become loop parameters."""

    return model.transform(RoundAndClipThresholds())


def _path_reaches_node(model, tensor_name: str, target_name: str) -> bool:
    """Return whether a forward tensor path reaches the named node."""

    pending = [tensor_name]
    visited = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        for consumer in model.find_consumers(current):
            if consumer.name == target_name:
                return True
            pending.extend(consumer.output)
    return False


def step_size_siglip_top_residual_fifo(model, cfg):
    """Buffer the post-loop residual while its sibling LayerNorm buffers a frame.

    The generic characterization algorithm cannot size reconvergent residuals.
    At the SigLIP MLP residual, LayerNorm consumes a complete folded frame before
    producing output, so its bypass must hold at least that complete frame.
    """

    if cfg.auto_fifo_depths:
        raise RuntimeError("SigLIP top-level residual sizing requires explicit FIFO sizing")

    matches = []
    for fork in model.get_nodes_by_op_type("DuplicateStreams_rtl"):
        if len(fork.output) != 2:
            continue
        consumers = [model.find_consumer(output) for output in fork.output]
        for norm_index, norm in enumerate(consumers):
            if norm is None or not norm.op_type.startswith("LayerNorm"):
                continue
            bypass_index = 1 - norm_index
            join = consumers[bypass_index]
            if join is None or not join.op_type.startswith("ElementwiseAdd"):
                continue
            join_inst = getCustomOp(join)
            if (
                join_inst.get_nodeattr("lhs_style") != "input"
                or join_inst.get_nodeattr("rhs_style") != "input"
            ):
                continue
            if not _path_reaches_node(model, norm.output[0], join.name):
                continue
            matches.append((fork, bypass_index, norm, join))

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one SigLIP LayerNorm residual fork, "
            f"found {[match[0].name for match in matches]}"
        )

    fork, bypass_index, norm, join = matches[0]
    folded_shape = getCustomOp(norm).get_folded_input_shape(0)
    frame_words = math.prod(folded_shape[:-1])
    # Round up so the memory-backed FIFO has a regular hardware depth.
    fifo_depth = max(2, 1 << (int(frame_words) - 1).bit_length())

    fork_inst = getCustomOp(fork)
    fork_depths = list(fork_inst.get_nodeattr("outFIFODepths"))
    fork_depths[bypass_index] = fifo_depth
    fork_inst.set_nodeattr("outFIFODepths", fork_depths)

    bypass_tensor = fork.output[bypass_index]
    join_index = list(join.input).index(bypass_tensor)
    join_inst = getCustomOp(join)
    join_depths = list(join_inst.get_nodeattr("inFIFODepths"))
    join_depths[join_index] = fifo_depth
    join_inst.set_nodeattr("inFIFODepths", join_depths)

    # The loop body has already been sized and capped before this injected step.
    # Raise only the subsequent top-level cap so it cannot truncate this FIFO.
    cfg.fifo_depth_cap = max(int(cfg.fifo_depth_cap or 0), fifo_depth)
    return model


def step_size_siglip_loop_residual_fifos(model, cfg):
    """Restore full-frame bypass FIFOs after capped loop-body RTL sizing."""

    sized = []
    for loop in model.get_nodes_by_op_type("FINNLoop"):
        loop_inst = getCustomOp(loop)
        body = loop_inst.get_nodeattr("body")
        for fifo in body.get_nodes_by_op_type("StreamingFIFO_rtl"):
            fork = body.find_producer(fifo.input[0])
            join = body.find_consumer(fifo.output[0])
            if (
                fork is None
                or fork.op_type != "DuplicateStreams_rtl"
                or len(fork.output) != 2
                or join is None
                or not join.op_type.startswith("ElementwiseAdd")
            ):
                continue
            join_inst = getCustomOp(join)
            if (
                join_inst.get_nodeattr("lhs_style") != "input"
                or join_inst.get_nodeattr("rhs_style") != "input"
            ):
                continue
            bypass_index = list(fork.output).index(fifo.input[0])
            norm_path = fork.output[1 - bypass_index]
            matching_norms = [
                norm
                for norm in body.get_nodes_by_op_type("LayerNorm_rtl")
                if _path_reaches_node(body, norm_path, norm.name)
                and _path_reaches_node(body, norm.output[0], join.name)
            ]
            if len(matching_norms) != 1:
                continue
            fifo_inst = getCustomOp(fifo)
            frame_words = math.prod(fifo_inst.get_folded_input_shape(0)[:-1])
            fifo_inst.set_nodeattr("depth", frame_words)

            fork_inst = getCustomOp(fork)
            fork_depths = list(fork_inst.get_nodeattr("outFIFODepths"))
            fork_depths[bypass_index] = frame_words
            fork_inst.set_nodeattr("outFIFODepths", fork_depths)

            join_index = list(join.input).index(fifo.output[0])
            join_depths = list(join_inst.get_nodeattr("inFIFODepths"))
            join_depths[join_index] = frame_words
            join_inst.set_nodeattr("inFIFODepths", join_depths)
            sized.append((fifo.name, frame_words))
        loop_inst.set_nodeattr("body", body.graph)

    if len(sized) != 2:
        raise RuntimeError(f"Expected two sized SigLIP loop residual FIFOs, found {sized}")
    return model


def make_mlo_boundary_step(depth: int):
    """Create a builder injection which marks the first repeated vision block."""

    def step_mark_siglip_mlo_boundary(model, cfg):
        ranges = find_vision_loop_body_ranges(model, depth)
        if len(ranges) != depth:
            raise RuntimeError(f"Expected {depth} SigLIP vision blocks, found {len(ranges)}")
        first_signature = ranges[0]["op_types"]
        mismatched = [item["block"] for item in ranges if item["op_types"] != first_signature]
        if mismatched:
            raise RuntimeError(f"SigLIP loop-body topology differs in blocks {mismatched}")

        nodes = model.graph.node
        cfg.loop_body_range = (
            nodes[ranges[0]["start_index"]],
            nodes[ranges[0]["end_index"]],
        )
        cfg.loop_body_hierarchy = [["", "layers.0"]]
        output_path = Path(cfg.output_dir) / "siglip_mlo_ranges.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(ranges, output_file, indent=2)
        return model

    step_mark_siglip_mlo_boundary.__name__ = "step_mark_siglip_mlo_boundary"
    return step_mark_siglip_mlo_boundary
