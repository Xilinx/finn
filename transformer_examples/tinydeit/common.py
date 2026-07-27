"""Shared utilities for the TinyDeiT FINN MLO flow.

TinyDeiT checkpoints can export GELU either as a repeated 51-node floating-point
polynomial subgraph or as an Erf-based ONNX decomposition. The flow collapses
both forms into PWPolyF before regular FINN conversion so the merged RTL operator
can be used.
"""

from __future__ import annotations

import argparse
import json
import math
import onnx
from collections import Counter
from onnx import helper
from pathlib import Path
from qonnx.core.datatype import DataType
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import RemoveUnusedTensors
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = REPO_ROOT / "onnx-checkpoints" / "deit_tiny_quant.onnx"
DEFAULT_BUILD_DIR = REPO_ROOT / "transformer_examples" / "tinydeit" / "build"
DEFAULT_BUILD_CSV = DEFAULT_BUILD_DIR / "builds.csv"
DEFAULT_BOARD = "VCK190"
DEFAULT_TARGET_FPS = 1000
DEFAULT_CLOCK_NS = 4.0
TRANSFORMER_DEPTH = 12
ATTENTION_MULTITHRESHOLDS_PER_BLOCK = 5

EXPORTED_PWPOLYF_SEQUENCE = [
    "Reshape",
    "Cast",
    "Less",
    "Cast",
    "BitShift",
    "BitShift",
    "BitShift",
    "BitwiseOr",
    "Cast",
    "BitwiseOr",
    "Cast",
    "BitwiseOr",
    "Cast",
    "Cast",
    "Cast",
    "Cast",
    "Where",
    "Where",
    "Where",
    "BitwiseAnd",
    "BitwiseAnd",
    "BitwiseAnd",
    "Sub",
    "Less",
    "Equal",
    "GreaterOrEqual",
    "Equal",
    "Cast",
    "Max",
    "And",
    "And",
    "Mul",
    "Add",
    "Add",
    "Add",
    "Add",
    "Where",
    "Where",
    "Clip",
    "Unsqueeze",
    "GatherND",
    "Gather",
    "Gather",
    "Gather",
    "Mul",
    "Add",
    "Mul",
    "Add",
    "Where",
    "Where",
    "Reshape",
]

LAYER_NORM_OP_TYPES = {
    "LayerNormalization",
    "LayerNorm",
    "LayerNorm_hls",
    "LayerNorm_rtl",
}

SOFTMAX_OP_TYPES = {"Softmax", "HWSoftmax", "HWSoftmax_hls", "HWSoftmax_rtl"}

RTL_PREFERRED_OP_TYPES = [
    "MVAU",
    "HWSoftmax",
    "LayerNorm",
    "PWPolyF",
    "ElementwiseAdd",
    "ElementwiseSub",
    "ElementwiseMul",
    "Where",
    "Requant",
]


def repo_path(path: str | Path) -> Path:
    """Resolve a user path relative to the repository root."""

    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def model_op_counts(model: onnx.ModelProto) -> dict[str, int]:
    return dict(Counter(node.op_type for node in model.graph.node).most_common())


def is_layer_norm(node: onnx.NodeProto) -> bool:
    return node.op_type in LAYER_NORM_OP_TYPES or node.op_type.startswith("LayerNorm")


def is_softmax(node: onnx.NodeProto) -> bool:
    return node.op_type in SOFTMAX_OP_TYPES


def summarize_model(model: onnx.ModelProto) -> dict[str, Any]:
    """Return a JSON-serializable structural summary."""

    return {
        "ir_version": model.ir_version,
        "opsets": {op.domain or "ai.onnx": op.version for op in model.opset_import},
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "inputs": [value_info_summary(x) for x in model.graph.input],
        "outputs": [value_info_summary(x) for x in model.graph.output],
        "op_counts": model_op_counts(model),
        "blocks": find_transformer_blocks(model),
    }


def value_info_summary(value_info: onnx.ValueInfoProto) -> dict[str, Any]:
    tensor_type = value_info.type.tensor_type
    dims = []
    for dim in tensor_type.shape.dim:
        if dim.dim_param:
            dims.append(dim.dim_param)
        elif dim.dim_value:
            dims.append(dim.dim_value)
        else:
            dims.append(None)
    return {"name": value_info.name, "shape": dims}


def find_transformer_blocks(
    model: onnx.ModelProto, depth: int = TRANSFORMER_DEPTH
) -> list[dict[str, Any]]:
    """Detect repeated TinyDeiT transformer blocks from LayerNorm pairs.

    Each block starts at the first LayerNorm in a pair and ends immediately before
    the next block's first LayerNorm.  The final block ends before the post-stack
    LayerNorm.
    """

    nodes = list(model.graph.node)
    ln_indices = [idx for idx, node in enumerate(nodes) if is_layer_norm(node)]
    if len(ln_indices) < (2 * depth + 1):
        return []

    blocks = []
    for block_idx in range(depth):
        start_idx = ln_indices[2 * block_idx]
        end_idx = ln_indices[2 * (block_idx + 1)] - 1
        block_nodes = nodes[start_idx : end_idx + 1]
        blocks.append(
            {
                "block": block_idx,
                "start_index": start_idx,
                "end_index": end_idx,
                "start_node": nodes[start_idx].name,
                "end_node": nodes[end_idx].name,
                "node_count": len(block_nodes),
                "op_counts": dict(Counter(node.op_type for node in block_nodes).most_common()),
                "softmax_nodes": [node.name for node in block_nodes if is_softmax(node)],
            }
        )
    return blocks


def find_mlo_loop_body_ranges(
    model: onnx.ModelProto, depth: int = TRANSFORMER_DEPTH
) -> list[dict[str, Any]]:
    """Detect TinyDeiT transformer loop-body ranges for FINN MLO rolling.

    The structural block summary starts at the first LayerNorm and, for all but
    the last block, ends at the DuplicateStreams node that feeds the next block.
    Loop rolling needs a self-contained repeated body instead: include the
    DuplicateStreams node that fans out the block input, and stop at the final
    residual add.  With these boundaries all 12 TinyDeiT blocks have the same
    topology, including the final block before the post-stack LayerNorm.
    """

    nodes = list(model.graph.node)
    blocks = find_transformer_blocks(model, depth)
    if len(blocks) != depth:
        return []

    ranges = []
    for block in blocks:
        start_idx = block["start_index"]
        start_node = nodes[start_idx]
        if start_idx > 0:
            prev_node = nodes[start_idx - 1]
            if (
                prev_node.op_type.startswith("DuplicateStreams")
                and start_node.input
                and start_node.input[0] in prev_node.output
            ):
                start_idx -= 1

        end_idx = block["end_index"]
        end_node = nodes[end_idx]
        if end_node.op_type.startswith("DuplicateStreams"):
            end_idx -= 1

        if end_idx < start_idx:
            raise RuntimeError(f"Invalid loop-body range for block {block['block']}")

        loop_nodes = nodes[start_idx : end_idx + 1]
        loop_block = dict(block)
        loop_block.update(
            {
                "loop_start_index": start_idx,
                "loop_end_index": end_idx,
                "loop_start_node": nodes[start_idx].name,
                "loop_end_node": nodes[end_idx].name,
                "loop_node_count": len(loop_nodes),
                "loop_op_types": [node.op_type for node in loop_nodes],
            }
        )
        ranges.append(loop_block)
    return ranges


def first_loop_body_node_range(model: Any, depth: int = TRANSFORMER_DEPTH) -> tuple[Any, Any]:
    proto = model.model if hasattr(model, "model") else model
    ranges = find_mlo_loop_body_ranges(proto, depth)
    if not ranges:
        raise RuntimeError("Could not detect TinyDeiT MLO loop-body ranges")
    loop_range = ranges[0]
    return (
        model.graph.node[loop_range["loop_start_index"]],
        model.graph.node[loop_range["loop_end_index"]],
    )


def exported_pwpolyf_match_indices(model: onnx.ModelProto) -> list[tuple[int, int]]:
    """Return topological ranges matching the exported PWPolyF decomposition."""

    nodes = list(model.graph.node)
    seq_len = len(EXPORTED_PWPOLYF_SEQUENCE)
    matches = []
    idx = 0
    while idx <= len(nodes) - seq_len:
        window = [node.op_type for node in nodes[idx : idx + seq_len]]
        if window == EXPORTED_PWPOLYF_SEQUENCE:
            matches.append((idx, idx + seq_len - 1))
            idx += seq_len
        else:
            idx += 1
    return matches


def _infer_pwpolyf_k_and_degree(model: Any, nodes: Iterable[Any]) -> tuple[int, int]:
    gathernd = next(node for node in nodes if node.op_type == "GatherND")
    coeffs = model.get_initializer(gathernd.input[0])
    if coeffs is None or len(coeffs.shape) != 2:
        raise RuntimeError("Matched PWPolyF export does not expose a coefficient table")
    num_segments = int(coeffs.shape[0])
    degree = int(coeffs.shape[1]) - 1
    subdivisions = (num_segments - 1) // 10
    if 1 + 10 * subdivisions != num_segments:
        raise RuntimeError(f"Unexpected PWPolyF coefficient shape {coeffs.shape}")
    k_float = math.log2(subdivisions)
    k = int(k_float)
    if (1 << k) != subdivisions:
        raise RuntimeError(f"Unexpected PWPolyF subdivision count {subdivisions}")
    return k, degree


def collapse_exported_pwpolyf(
    model: Any, expected_count: int | None = TRANSFORMER_DEPTH
) -> tuple[Any, int]:
    """Replace exported TinyDeiT GELU/PWPolyF subgraphs with PWPolyF nodes."""

    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    matches = exported_pwpolyf_match_indices(model.model)
    erf_count = len(model.get_nodes_by_op_type("Erf"))
    if matches and erf_count:
        raise RuntimeError("TinyDeiT export mixes polynomial PWPolyF and Erf GELU decompositions")
    if expected_count is not None and matches and len(matches) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} exported PWPolyF decompositions, found {len(matches)}"
        )
    if not matches:
        if erf_count:
            from finn.transformation.fpgadataflow.convert_to_hw_layers import (  # noqa: PLC0415
                InferPWPolyFLayer,
            )

            model = model.transform(InferPWPolyFLayer())
            remaining_erf_count = len(model.get_nodes_by_op_type("Erf"))
            converted_count = erf_count - remaining_erf_count
            if remaining_erf_count:
                raise RuntimeError(
                    f"Found {erf_count} Erf nodes but converted only {converted_count} "
                    "complete GELU decompositions"
                )
            if expected_count is not None and converted_count != expected_count:
                raise RuntimeError(
                    f"Expected {expected_count} Erf GELU decompositions, "
                    f"converted {converted_count}"
                )
            model = model.transform(RemoveUnusedTensors())
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
            return model, converted_count
        if expected_count is not None:
            raise RuntimeError(
                f"Expected {expected_count} exported PWPolyF decompositions, found 0"
            )
        return model, 0

    nodes = list(model.graph.node)
    replacement_by_start = {}
    skip_indices = set()
    for match_idx, (start_idx, end_idx) in enumerate(matches):
        match_nodes = nodes[start_idx : end_idx + 1]
        start_node = match_nodes[0]
        end_node = match_nodes[-1]
        input_name = start_node.input[0]
        output_name = end_node.output[0]
        input_shape = model.get_tensor_shape(input_name)
        if input_shape is None or len(input_shape) == 0:
            raise RuntimeError(f"Could not infer shape for PWPolyF input {input_name}")
        k, degree = _infer_pwpolyf_k_and_degree(model, match_nodes)
        node = helper.make_node(
            "PWPolyF",
            [input_name],
            [output_name],
            domain="finn.custom_op.fpgadataflow",
            backend="fpgadataflow",
            func="gelu",
            K=k,
            degree=degree,
            NumChannels=int(input_shape[-1]),
            PE=1,
            inputDataType="FLOAT32",
            outputDataType="FLOAT32",
            numInputVectors=[int(x) for x in input_shape[:-1]],
            name=f"PWPolyF_tinydeit_{match_idx}",
        )
        if hasattr(start_node, "metadata_props"):
            node.metadata_props.extend(start_node.metadata_props)
        replacement_by_start[start_idx] = node
        skip_indices.update(range(start_idx, end_idx + 1))
        model.set_tensor_datatype(input_name, DataType["FLOAT32"])
        model.set_tensor_datatype(output_name, DataType["FLOAT32"])

    new_nodes = []
    for idx, node in enumerate(nodes):
        if idx in replacement_by_start:
            new_nodes.append(replacement_by_start[idx])
        if idx not in skip_indices:
            new_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    return model, len(matches)


def write_rtl_specialization_config(path: str | Path) -> dict[str, Any]:
    """Write a specialization config that explicitly prefers RTL where supported."""

    config = {"Defaults": {"preferred_impl_style": ["rtl", RTL_PREFERRED_OP_TYPES]}}
    write_json(path, config)
    return config


def ensure_conv_kernel_shape_attrs(model: Any) -> tuple[Any, int]:
    """Populate missing Conv kernel_shape attrs from folded weight initializers."""

    changed = 0
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        if any(attr.name == "kernel_shape" for attr in node.attribute):
            continue
        weights = model.get_initializer(node.input[1])
        if weights is None or len(weights.shape) < 4:
            raise RuntimeError(f"Could not infer kernel_shape for Conv node {node.name}")
        node.attribute.append(
            helper.make_attribute("kernel_shape", [int(x) for x in weights.shape[-2:]])
        )
        changed += 1
    return model, changed


def mark_attention_multithreshold_layouts_unknown(
    model: Any, depth: int = TRANSFORMER_DEPTH
) -> tuple[Any, int]:
    """Prevent image-layout inference on TinyDeiT attention activations.

    The five rank-4 MultiThreshold tensors in each attention block use
    ``[batch, head, token, feature]`` axes, not image-style NCHW axes. Leaving
    the MultiThreshold default ``data_layout=NCHW`` in place causes
    ``InferDataLayouts`` to label the head dimension as channels. Hardware
    conversion then inserts redundant NCHW/NHWC shuffles and folds the
    three-head dimension instead of the 197-token dimension.

    Mark only the structurally identified attention MultiThreshold nodes as
    layout-agnostic. The input-image MultiThreshold remains NCHW.
    """

    model = model.transform(InferShapes())
    blocks = find_transformer_blocks(model.model, depth)
    if len(blocks) != depth:
        raise RuntimeError(f"Expected {depth} TinyDeiT blocks, found {len(blocks)}")

    normalized = []
    nodes = list(model.graph.node)
    for block in blocks:
        block_nodes = nodes[block["start_index"] : block["end_index"] + 1]
        block_multithresholds = []
        for node in block_nodes:
            if node.op_type != "MultiThreshold":
                continue
            input_shape = model.get_tensor_shape(node.input[0])
            if input_shape is not None and len(input_shape) == 4:
                block_multithresholds.append(node)
        if len(block_multithresholds) != ATTENTION_MULTITHRESHOLDS_PER_BLOCK:
            raise RuntimeError(
                f"Expected {ATTENTION_MULTITHRESHOLDS_PER_BLOCK} rank-4 attention "
                f"MultiThreshold nodes in block {block['block']}, found "
                f"{len(block_multithresholds)}"
            )
        for node in block_multithresholds:
            getCustomOp(node).set_nodeattr("data_layout", "UNKNOWN")
            normalized.append(node.name)

    return model, len(normalized)


def move_forked_scalar_mul_past_matmul(model: Any) -> tuple[Any, int]:
    """Move scalar dequantization past forked MatMul consumers.

    QVS exports layer-normalization activations as Thresholding -> scalar Mul
    before the three Q/K/V projections. The scalar Mul fans out to all three
    MatMuls, so the generic single-consumer streamline transform cannot move it.
    Moving the scalar to each MatMul output preserves the math and exposes
    integer MatMul inputs for FINN MVAU inference.
    """

    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    nodes_to_remove = set()
    new_nodes_after = {}
    moved = 0
    for node in list(model.graph.node):
        if node.op_type != "Mul" or len(node.input) != 2 or len(node.output) != 1:
            continue
        lhs_init = model.get_initializer(node.input[0])
        rhs_init = model.get_initializer(node.input[1])
        if lhs_init is not None and lhs_init.size == 1 and rhs_init is None:
            scalar_input = node.input[0]
            data_input = node.input[1]
        elif rhs_init is not None and rhs_init.size == 1 and lhs_init is None:
            scalar_input = node.input[1]
            data_input = node.input[0]
        else:
            continue

        consumers = model.find_consumers(node.output[0])
        if len(consumers) < 2:
            continue
        if any(
            consumer.op_type != "MatMul" or node.output[0] not in consumer.input
            for consumer in consumers
        ):
            continue

        for consumer in consumers:
            output_name = consumer.output[0]
            moved_matmul_output = f"{consumer.name}_prescale_out"
            for input_idx, input_name in enumerate(consumer.input):
                if input_name == node.output[0]:
                    consumer.input[input_idx] = data_input
            consumer.output[0] = moved_matmul_output
            output_shape = model.get_tensor_shape(output_name)
            if output_shape is not None:
                model.set_tensor_shape(moved_matmul_output, output_shape)
            new_mul = helper.make_node(
                "Mul",
                [moved_matmul_output, scalar_input],
                [output_name],
                name=f"{node.name}_after_{consumer.name}",
            )
            if hasattr(node, "metadata_props"):
                new_mul.metadata_props.extend(node.metadata_props)
            new_nodes_after.setdefault(consumer.name, []).append(new_mul)
        nodes_to_remove.add(node.name)
        moved += 1

    if moved == 0:
        return model, 0

    new_nodes = []
    for node in model.graph.node:
        if node.name not in nodes_to_remove:
            new_nodes.append(node)
            new_nodes.extend(new_nodes_after.get(node.name, []))
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    return model, moved


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        default=str(DEFAULT_CHECKPOINT.relative_to(REPO_ROOT)),
        help="Input TinyDeiT ONNX checkpoint, relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--output-dir",
        default=str((DEFAULT_BUILD_DIR / "flow").relative_to(REPO_ROOT)),
        help="Output directory, relative to repo root unless absolute.",
    )


def resolve_common_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    return repo_path(args.input), repo_path(args.output_dir)
