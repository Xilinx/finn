#!/usr/bin/env python3
"""Prepare TinyDeiT for FINN dataflow and optional MLO loop rolling."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes

import finn.builder.build_dataflow_steps as steps
from finn.builder.build_dataflow_config import DataflowBuildConfig, DataflowOutputType
from finn.transformation.fpgadataflow.loop_rolling import LoopExtraction, LoopRolling
from finn.transformation.fpgadataflow.set_loop_boundary import SetLoopBoundary
from finn.transformation.general import ApplyConfig
from finn.transformation.streamline.extract_norm_scale_bias import ExtractNormScaleBias
from transformer_examples.tinydeit.common import (
    DEFAULT_BOARD,
    DEFAULT_CLOCK_NS,
    DEFAULT_TARGET_FPS,
    TRANSFORMER_DEPTH,
    add_common_args,
    collapse_exported_pwpolyf,
    ensure_conv_kernel_shape_attrs,
    find_mlo_loop_body_ranges,
    find_transformer_blocks,
    first_loop_body_node_range,
    model_op_counts,
    move_forked_scalar_mul_past_matmul,
    resolve_common_paths,
    summarize_model,
    write_json,
    write_rtl_specialization_config,
)


def make_cfg(args: argparse.Namespace, output_dir: Path) -> DataflowBuildConfig:
    return DataflowBuildConfig(
        output_dir=str(output_dir),
        synth_clk_period_ns=args.clock_ns,
        board=args.board,
        target_fps=args.target_fps,
        standalone_thresholds=True,
        infer_shuffle_skip_first=False,
        save_intermediate_models=True,
        generate_outputs=[DataflowOutputType.ESTIMATE_REPORTS],
        mlo=args.mlo,
        no_stdout_redirect=True,
    )


def save_checkpoint(model: ModelWrapper, output_dir: Path, name: str, save: bool) -> None:
    if save:
        model.save(str(output_dir / f"{name}.onnx"))
    print(f"{name}: {len(model.graph.node)} nodes")
    print(Counter(node.op_type for node in model.graph.node).most_common(20))


def roll_mlo(model: ModelWrapper, output_dir: Path, depth: int) -> ModelWrapper:
    blocks = find_transformer_blocks(model.model, depth)
    if len(blocks) != depth:
        raise RuntimeError(f"Expected {depth} transformer blocks, found {len(blocks)}")
    loop_ranges = find_mlo_loop_body_ranges(model.model, depth)
    if len(loop_ranges) != depth:
        raise RuntimeError(f"Expected {depth} loop-body ranges, found {len(loop_ranges)}")
    loop_op_types = loop_ranges[0]["loop_op_types"]
    mismatched = [item["block"] for item in loop_ranges if item["loop_op_types"] != loop_op_types]
    if mismatched:
        raise RuntimeError(f"Loop-body topology mismatch in blocks {mismatched}")
    start_node, end_node = first_loop_body_node_range(model, depth)
    node_metadata = {
        "pkg.torch.onnx.name_scopes": "['', 'layers.0']",
        "pkg.torch.onnx.class_hierarchy": "['TinyDeiT', 'Block']",
    }
    model = model.transform(SetLoopBoundary(node_metadata, (start_node, end_node)))
    loop_extraction = LoopExtraction(hierarchy_list=[["", "layers.0"]])
    model = model.transform(loop_extraction)
    fn_count = len(model.get_nodes_by_op_type("fn_loop-body"))
    if fn_count != depth:
        raise RuntimeError(f"Loop extraction found {fn_count} function calls, expected {depth}")
    model = model.transform(LoopRolling(loop_extraction.loop_body_template))
    model = model.transform(InferShapes(), apply_to_subgraphs=True)
    model = model.transform(InferDataTypes(), apply_to_subgraphs=True)
    model = model.transform(GiveUniqueNodeNames(), apply_to_subgraphs=True)
    model = model.transform(GiveReadableTensorNames())
    loop_template = Path("loop-body-template.onnx")
    if loop_template.is_file():
        loop_template.replace(output_dir / "loop-body-template.onnx")
    return model


def prepare(args: argparse.Namespace) -> Path:
    input_path, output_dir = resolve_common_paths(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_cfg(args, output_dir)
    model = ModelWrapper(str(input_path))
    write_json(output_dir / "00_input_summary.json", summarize_model(model.model))
    save_checkpoint(model, output_dir, "00_input", args.save_intermediate)

    for name, step in [
        ("01_qonnx_to_finn", steps.step_qonnx_to_finn),
        ("02_tidy_up", steps.step_tidy_up),
    ]:
        model = step(model, cfg)
        save_checkpoint(model, output_dir, name, args.save_intermediate)

    if args.collapse_pwpolyf:
        model, count = collapse_exported_pwpolyf(model, expected_count=TRANSFORMER_DEPTH)
        print(f"Collapsed exported GELU/PWPolyF decompositions: {count}")
        save_checkpoint(model, output_dir, "03_collapse_pwpolyf", args.save_intermediate)

    model, conv_attrs = ensure_conv_kernel_shape_attrs(model)
    if conv_attrs:
        print(f"Filled missing Conv kernel_shape attributes: {conv_attrs}")

    if not args.skip_streamline:
        model = steps.step_streamline(model, cfg)
        save_checkpoint(model, output_dir, "04_streamline", args.save_intermediate)

    model = model.transform(ExtractNormScaleBias())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    save_checkpoint(model, output_dir, "04b_extract_norm_scale_bias", args.save_intermediate)

    model, moved_muls = move_forked_scalar_mul_past_matmul(model)
    if moved_muls:
        print(f"Moved forked scalar Mul nodes past MatMul consumers: {moved_muls}")
        save_checkpoint(model, output_dir, "04c_move_forked_scalar_muls", args.save_intermediate)

    model = steps.step_convert_to_hw(model, cfg)
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(InferDataTypes())
    save_checkpoint(model, output_dir, "05_convert_to_hw", args.save_intermediate)

    model = steps.step_create_dataflow_partition(model, cfg)
    save_checkpoint(model, output_dir, "06_dataflow_partition", args.save_intermediate)

    specialize_config = output_dir / "specialize_layers_config.json"
    write_rtl_specialization_config(specialize_config)
    model = model.transform(ApplyConfig(str(specialize_config)))
    model = steps.step_specialize_layers(model, cfg)
    save_checkpoint(model, output_dir, "07_specialize_layers", args.save_intermediate)

    if args.mlo:
        model = roll_mlo(model, output_dir, args.depth)
        save_checkpoint(model, output_dir, "08_mlo_rolled", args.save_intermediate)

    final_path = output_dir / ("tinydeit_mlo.onnx" if args.mlo else "tinydeit_dataflow.onnx")
    model.save(str(final_path))
    write_json(output_dir / "final_summary.json", summarize_model(model.model))
    write_json(output_dir / "final_op_counts.json", model_op_counts(model.model))
    print(f"Wrote final model: {final_path}")
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--board", default=DEFAULT_BOARD)
    parser.add_argument("--clock-ns", type=float, default=DEFAULT_CLOCK_NS)
    parser.add_argument("--target-fps", type=int, default=DEFAULT_TARGET_FPS)
    parser.add_argument("--depth", type=int, default=TRANSFORMER_DEPTH)
    parser.add_argument("--skip-streamline", action="store_true")
    parser.add_argument("--no-collapse-pwpolyf", dest="collapse_pwpolyf", action="store_false")
    parser.add_argument("--no-mlo", dest="mlo", action="store_false")
    parser.add_argument("--save-intermediate", action="store_true")
    parser.set_defaults(collapse_pwpolyf=True, mlo=True)
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
