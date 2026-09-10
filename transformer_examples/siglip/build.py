# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Compile a QV-LSQ SigLIP image tower with the phase-based FINN builder."""

from __future__ import annotations

import argparse
import json
import onnx
from collections import Counter
from pathlib import Path
from typing import Any

import finn.builder.build_dataflow as build
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    DataflowOutputType,
    VerificationStepType,
    default_build_dataflow_steps,
    estimate_only_dataflow_steps,
)
from transformer_examples.siglip.config import (
    DEFAULT_PROFILE,
    SiglipProfile,
    load_profile,
)
from transformer_examples.siglip.mlo import (
    make_mlo_boundary_step,
    step_round_siglip_thresholds_before_mlo,
    step_size_siglip_top_residual_fifo,
)
from transformer_examples.siglip.phases import (
    make_siglip_folding_step,
    phase_optimize_siglip,
    phase_prepare_siglip,
)


def _shape(value_info) -> list[int | None]:
    return [dimension.dim_value or None for dimension in value_info.type.tensor_type.shape.dim]


def validate_model(model_path: Path, profile: SiglipProfile) -> dict[str, Any]:
    """Fail early when an input is not the static QONNX graph this flow expects."""

    model = onnx.load(str(model_path), load_external_data=False)
    if len(model.graph.input) != 1:
        raise ValueError(f"Expected one image input, found {len(model.graph.input)}")
    input_shape = _shape(model.graph.input[0])
    expected_shape = [1, 3, profile.model["image_size"], profile.model["image_size"]]
    if input_shape != expected_shape:
        raise ValueError(f"Expected image input shape {expected_shape}, found {input_shape}")
    if any(dimension is None for dimension in input_shape):
        raise ValueError("Dynamic input dimensions are not supported")

    counts = Counter(f"{node.domain or 'ai.onnx'}::{node.op_type}" for node in model.graph.node)
    requirements = {
        "qonnx.custom_op.general::Quant": 1,
        "ai.onnx::LayerNormalization": 2 * profile.model["vision_depth"],
        "ai.onnx::Softmax": profile.model["vision_depth"],
    }
    missing = {
        op_type: minimum for op_type, minimum in requirements.items() if counts[op_type] < minimum
    }
    if missing:
        raise ValueError(f"Input is not a complete quantized SigLIP vision tower: {missing}")
    output_names = [value_info.name for value_info in model.graph.output]
    expected_output = profile.model["output_name"]
    if expected_output not in output_names:
        raise ValueError(f"Expected model output {expected_output!r}, found {output_names}")
    return {
        "input": str(model_path.resolve()),
        "input_size_bytes": model_path.stat().st_size,
        "input_name": model.graph.input[0].name,
        "input_shape": input_shape,
        "outputs": output_names,
        "node_count": len(model.graph.node),
        "op_counts": dict(counts.most_common()),
    }


def validate_quantization_report(report_path: Path, profile: SiglipProfile) -> dict[str, Any]:
    """Validate optional QV export provenance without inferring accuracy."""

    with report_path.open(encoding="utf-8") as report_file:
        report = json.load(report_file)
    expected = {
        "model_id": profile.model["model_id"],
        "quantizer": profile.quantization["scheme"],
        "weight_bits": profile.quantization["weight_bits"],
        "act_bits": profile.quantization["activation_bits"],
        "edge_bits": profile.quantization["edge_bits"],
        "exported_vision_depth": profile.model["vision_depth"],
        "exported_image_size": profile.model["image_size"],
        "patch_size": profile.model["patch_size"],
    }
    mismatches = {
        key: {"expected": value, "actual": report.get(key)}
        for key, value in expected.items()
        if report.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Quantization report does not match the profile: {mismatches}")
    return {key: report[key] for key in expected}


def _verification_steps(level: str, mode: str):
    if level == "none":
        return None
    if mode == "estimate" and level in ("cppsim", "rtlsim"):
        raise ValueError(f"{level} verification requires stitched_ip or ooc_synth mode")
    steps = [
        VerificationStepType.QONNX_TO_FINN_PYTHON,
        VerificationStepType.TIDY_UP_PYTHON,
        VerificationStepType.STREAMLINED_PYTHON,
    ]
    if level in ("cppsim", "rtlsim"):
        steps.append(VerificationStepType.FOLDED_HLS_CPPSIM)
    if level == "rtlsim":
        steps.extend(
            [
                VerificationStepType.NODE_BY_NODE_RTLSIM,
                VerificationStepType.STITCHED_IP_RTLSIM,
            ]
        )
    return steps


def build_siglip(
    model_path: Path,
    output_dir: Path,
    profile: SiglipProfile,
    mode: str,
    verify_level: str,
    input_npy: Path | None,
    expected_output_npy: Path | None,
    measure_rtlsim_performance: bool,
) -> int:
    """Run the requested phase-based FINN build."""

    if (input_npy is None) != (expected_output_npy is None):
        raise ValueError("--input-npy and --expected-output-npy must be supplied together")
    if verify_level != "none" and input_npy is None:
        raise ValueError("Verification requires --input-npy and --expected-output-npy")

    outputs = [DataflowOutputType.ESTIMATE_REPORTS]
    steps = list(estimate_only_dataflow_steps)
    if mode != "estimate":
        outputs.append(DataflowOutputType.STITCHED_IP)
        steps = list(default_build_dataflow_steps)
    steps[0] = phase_prepare_siglip
    if mode == "ooc_synth":
        outputs.append(DataflowOutputType.OOC_SYNTH)
    if measure_rtlsim_performance:
        if mode == "estimate":
            raise ValueError("RTL performance measurement requires stitched_ip or ooc_synth mode")
        outputs.append(DataflowOutputType.RTLSIM_PERFORMANCE)
    steps[1] = phase_optimize_siglip

    build_config = profile.build
    folding_config = profile.resolve_file(build_config.get("folding_config"))
    specialization_config = profile.resolve_file(build_config.get("specialization_config"))
    if folding_config is None:
        raise ValueError("The SigLIP VCK190 profile requires a reviewed folding config")
    if not folding_config.is_file():
        raise ValueError(f"Folding config does not exist: {folding_config}")
    pre_decomposition_folding = make_siglip_folding_step(folding_config, False)
    final_folding = make_siglip_folding_step(folding_config, True)
    cfg = DataflowBuildConfig(
        output_dir=str(output_dir),
        steps=steps,
        generate_outputs=outputs,
        board=build_config["board"],
        synth_clk_period_ns=float(build_config["clock_ns"]),
        # Keep the automatic result as a reviewable baseline; the injected
        # profile steps below apply the measured manual schedule afterwards.
        target_fps=float(build_config["target_fps"]),
        mvau_wwidth_max=int(build_config["mvau_wwidth_max"]),
        max_multithreshold_bit_width=int(profile.quantization["edge_bits"]),
        folding_config_file=None,
        specialize_layers_config_file=(
            str(specialization_config) if specialization_config else None
        ),
        standalone_thresholds=True,
        infer_shuffle_skip_first=False,
        folding_two_pass_relaxation=False,
        # Loop bodies still use large-FIFO RTL sizing. The MLO top level uses
        # explicit depths because characterization cannot size reconvergent
        # residual paths.
        auto_fifo_depths=False,
        fifo_depth_cap=int(build_config["fifo_depth_cap"]),
        save_intermediate_models=True,
        mlo=True,
        # Non-None placeholders satisfy pre-build checks. The injected step
        # replaces these with the detected converted-graph block range.
        loop_body_range=[],
        loop_body_hierarchy=[["", "layers.0"]],
        inject_steps_before={
            "step_loop_rolling": [
                step_round_siglip_thresholds_before_mlo,
                make_mlo_boundary_step(profile.model["vision_depth"]),
            ],
            "step_set_fifo_depths": [step_size_siglip_top_residual_fifo],
        },
        inject_steps_after={
            "step_target_fps_parallelization": [pre_decomposition_folding],
            "step_transpose_decomposition": [final_folding],
        },
        stitched_ip_gen_dcp=mode == "ooc_synth",
        verify_steps=_verification_steps(verify_level, mode),
        verify_input_npy=str(input_npy) if input_npy else None,
        verify_expected_output_npy=(str(expected_output_npy) if expected_output_npy else None),
        verification_atol=float(build_config["verification_atol"]),
        # Verify the finite FIFO implementation that will be synthesized.
        verify_rtlsim_behavioral=False,
        rtlsim_batch_size=2,
        enable_build_pdb_debug=False,
    )
    return build.build_dataflow_cfg(str(model_path), cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="QV-LSQ QONNX model")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--quantization-report", type=Path)
    parser.add_argument(
        "--mode", choices=("estimate", "stitched_ip", "ooc_synth"), default="estimate"
    )
    parser.add_argument("--verify", choices=("none", "python", "cppsim", "rtlsim"), default="none")
    parser.add_argument("--input-npy", type=Path)
    parser.add_argument("--expected-output-npy", type=Path)
    parser.add_argument("--measure-rtlsim-performance", action="store_true")
    parser.add_argument(
        "--validate-only", action="store_true", help="Validate input/profile provenance and exit"
    )
    args = parser.parse_args()

    if not args.model.is_file():
        parser.error(f"Model does not exist: {args.model}")
    profile = load_profile(args.profile)
    provenance = {
        "profile": profile.name,
        "model": validate_model(args.model, profile),
        "quantization_report": None,
    }
    if args.quantization_report:
        provenance["quantization_report"] = validate_quantization_report(
            args.quantization_report, profile
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "siglip_input_provenance.json").open(
        "w", encoding="utf-8"
    ) as output_file:
        json.dump(provenance, output_file, indent=2)
    if args.validate_only:
        return

    result = build_siglip(
        model_path=args.model,
        output_dir=args.output_dir,
        profile=profile,
        mode=args.mode,
        verify_level=args.verify,
        input_npy=args.input_npy,
        expected_output_npy=args.expected_output_npy,
        measure_rtlsim_performance=args.measure_rtlsim_performance,
    )
    if result != 0:
        raise SystemExit(result)


if __name__ == "__main__":
    main()
