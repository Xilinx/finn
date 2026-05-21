#!/usr/bin/env python3
"""Run the TinyDeiT FINN MLO build for V80."""

from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import gen_finn_dt_tensor

import finn.builder.build_dataflow as build
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    DataflowOutputType,
    VerificationStepType,
)
from finn.core.onnx_exec import execute_onnx
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.util.basic import getHWCustomOp
from tinydeit.common import (
    DEFAULT_BOARD,
    DEFAULT_BUILD_DIR,
    DEFAULT_CHECKPOINT,
    DEFAULT_CLOCK_NS,
    DEFAULT_TARGET_FPS,
    repo_path,
)
from tinydeit.prepare_model import prepare


def _rounded_threshold_datatype(dtype: DataType) -> DataType:
    max_val = dtype.max() + 1
    if not dtype.signed():
        return DataType.get_smallest_possible(max_val)
    return DataType.get_smallest_possible(-(max_val) - 1)


def step_round_mlo_threshold_params(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Round FINNLoop threshold parameter tensors after bit-width minimization.

    RoundAndClipThresholds handles regular Thresholding initializers, but MLO
    rolling turns per-layer thresholds into indexed FINNLoop inputs.  The RTL
    threshold generator still expects those parameter tensors and the loop-body
    weightDataType attributes to reflect integer thresholds.
    """

    for loop_node in model.get_nodes_by_op_type("FINNLoop"):
        loop_inst = getHWCustomOp(loop_node)
        loop_body = loop_inst.get_nodeattr("body")
        body_input_names = [value_info.name for value_info in loop_body.graph.input]

        for node in loop_body.graph.node:
            if not node.op_type.startswith("Thresholding"):
                continue
            inst = getHWCustomOp(node)
            dtype = inst.get_input_datatype(0)
            if not dtype.is_integer():
                continue
            try:
                body_input_index = body_input_names.index(node.input[1])
            except ValueError:
                continue

            top_param_name = loop_node.input[body_input_index]
            thresholds = model.get_initializer(top_param_name)
            if thresholds is None:
                raise RuntimeError(f"Missing FINNLoop threshold parameter {top_param_name}")

            new_thresholds = np.clip(np.ceil(thresholds), dtype.min(), dtype.max() + 1)
            new_thresholds = new_thresholds.astype(np.float32)
            tdt = _rounded_threshold_datatype(dtype)
            model.set_initializer(top_param_name, new_thresholds)
            model.set_tensor_datatype(top_param_name, tdt)
            loop_body.set_tensor_datatype(node.input[1], tdt)
            inst.set_nodeattr("weightDataType", tdt.name)

        loop_inst.set_nodeattr("body", loop_body.graph)
    return model


BUILD_STEPS_ESTIMATE = [
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    step_round_mlo_threshold_params,
    "step_transpose_decomposition",
    "step_generate_estimate_reports",
]

BUILD_STEPS_RTL = BUILD_STEPS_ESTIMATE + [
    "step_hw_codegen",
    "step_hw_ipgen",
    "step_set_fifo_depths",
    "step_create_stitched_ip",
]

BUILD_STEPS_FULL_RTLSIM = BUILD_STEPS_RTL + ["step_measure_rtlsim_performance"]


def build_config(args: argparse.Namespace, output_dir: Path) -> DataflowBuildConfig:
    if args.mode == "estimate":
        steps = BUILD_STEPS_ESTIMATE
        outputs = [DataflowOutputType.ESTIMATE_REPORTS]
        verify_steps = []
    elif args.mode in ["rtl", "dcp"]:
        steps = BUILD_STEPS_RTL
        outputs = [DataflowOutputType.ESTIMATE_REPORTS, DataflowOutputType.STITCHED_IP]
        verify_steps = []
        if args.stitched_rtlsim:
            verify_steps.append(VerificationStepType.STITCHED_IP_RTLSIM)
    elif args.mode == "full-rtlsim":
        steps = BUILD_STEPS_FULL_RTLSIM
        outputs = [
            DataflowOutputType.ESTIMATE_REPORTS,
            DataflowOutputType.STITCHED_IP,
            DataflowOutputType.RTLSIM_PERFORMANCE,
        ]
        verify_steps = [VerificationStepType.STITCHED_IP_RTLSIM]
    else:
        steps = BUILD_STEPS_RTL + [
            "step_out_of_context_synthesis",
            "step_synthesize_bitfile",
            "step_make_driver",
            "step_deployment_package",
        ]
        outputs = [
            DataflowOutputType.ESTIMATE_REPORTS,
            DataflowOutputType.STITCHED_IP,
            DataflowOutputType.OOC_SYNTH,
            DataflowOutputType.BITFILE,
            DataflowOutputType.CPP_DRIVER,
            DataflowOutputType.DEPLOYMENT_PACKAGE,
        ]
        verify_steps = [VerificationStepType.FOLDED_HLS_CPPSIM]

    return DataflowBuildConfig(
        output_dir=str(output_dir),
        synth_clk_period_ns=args.clock_ns,
        board=args.board,
        target_fps=args.target_fps,
        standalone_thresholds=True,
        infer_shuffle_skip_first=False,
        save_intermediate_models=True,
        verify_steps=verify_steps,
        verify_input_npy=str(output_dir / "input.npy"),
        verify_expected_output_npy=str(output_dir / "expected_output.npy"),
        verify_save_full_context=True,
        verification_atol=args.atol,
        generate_outputs=outputs,
        steps=steps,
        mlo=True,
        auto_fifo_depths=False,
        rtlsim_batch_size=args.rtlsim_batch_size,
        stitched_ip_gen_dcp=args.mode == "dcp" or args.stitched_ip_dcp,
        no_stdout_redirect=True,
        enable_build_pdb_debug=False,
    )


def prepare_cppsim(model: ModelWrapper, num_workers: int | None) -> ModelWrapper:
    model = model.transform(SetExecMode("cppsim"), apply_to_subgraphs=True)
    model = model.transform(PrepareCppSim(num_workers), apply_to_subgraphs=True)
    model = model.transform(CompileCppSim(num_workers), apply_to_subgraphs=True)
    model = model.transform(SetExecMode("cppsim"), apply_to_subgraphs=True)
    return model


def generate_reference_io(
    reference_model_path: Path,
    output_dir: Path,
    seed: int,
    shape_model_path: Path | None = None,
    cppsim_prepare: bool = True,
    cppsim_workers: int | None = None,
) -> None:
    shape_model = ModelWrapper(str(shape_model_path or reference_model_path))
    input_name = shape_model.get_first_global_in()
    input_shape = shape_model.get_tensor_shape(input_name)
    if input_shape is None:
        raise RuntimeError(f"Could not infer input shape for {input_name}")
    np.random.seed(seed)
    input_tensor = gen_finn_dt_tensor(DataType["FLOAT32"], input_shape)
    np.save(output_dir / "input.npy", input_tensor)

    reference_model = ModelWrapper(str(reference_model_path))
    if cppsim_prepare:
        reference_model = prepare_cppsim(reference_model, cppsim_workers)
    reference_input_name = reference_model.get_first_global_in()
    reference_output_name = reference_model.get_first_global_out()
    output_dict = execute_onnx(
        reference_model,
        {reference_input_name: input_tensor},
        return_full_exec_context=True,
    )
    np.save(output_dir / "expected_output.npy", output_dict[reference_output_name])
    np.savez(output_dir / "expected_context.npz", **output_dict)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_CHECKPOINT.relative_to(repo_path("."))))
    parser.add_argument(
        "--output-dir",
        default=str((DEFAULT_BUILD_DIR / "v80_mlo").relative_to(repo_path("."))),
    )
    parser.add_argument(
        "--mode", choices=["estimate", "rtl", "dcp", "full-rtlsim", "bitfile"], default="rtl"
    )
    parser.add_argument("--board", default=DEFAULT_BOARD)
    parser.add_argument("--clock-ns", type=float, default=DEFAULT_CLOCK_NS)
    parser.add_argument("--target-fps", type=int, default=DEFAULT_TARGET_FPS)
    parser.add_argument("--atol", type=float, default=1e-1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rtlsim-batch-size", type=int, default=1)
    parser.add_argument("--node-by-node", action="store_true")
    parser.add_argument("--stitched-rtlsim", action="store_true")
    parser.add_argument("--stitched-ip-dcp", action="store_true")
    parser.add_argument("--prepared-model", default=None)
    parser.add_argument("--reference-model", default=None)
    parser.add_argument(
        "--no-reference-cppsim-prepare", dest="reference_cppsim_prepare", action="store_false"
    )
    parser.add_argument("--reference-cppsim-workers", type=int, default=None)
    parser.add_argument("--skip-reference-io", action="store_true")
    parser.set_defaults(reference_cppsim_prepare=True)
    args = parser.parse_args()

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.prepared_model is None:
        prep_args = argparse.Namespace(
            input=args.input,
            output_dir=str(output_dir / "prepare"),
            board=args.board,
            clock_ns=args.clock_ns,
            target_fps=args.target_fps,
            depth=12,
            skip_streamline=False,
            collapse_pwpolyf=True,
            mlo=True,
            save_intermediate=True,
        )
        model_path = prepare(prep_args)
    else:
        model_path = repo_path(args.prepared_model)

    cfg = build_config(args, output_dir)
    if not args.skip_reference_io and cfg.verify_steps:
        reference_model_path = (
            repo_path(args.reference_model) if args.reference_model else model_path
        )
        generate_reference_io(
            reference_model_path,
            output_dir,
            args.seed,
            model_path,
            args.reference_cppsim_prepare,
            args.reference_cppsim_workers,
        )
    build.build_dataflow_cfg(str(model_path), cfg)
    print(f"Build output: {output_dir}")


if __name__ == "__main__":
    main()
