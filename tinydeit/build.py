#!/usr/bin/env python3
"""Run the TinyDeiT FINN MLO build for V80."""

from __future__ import annotations

import argparse
import csv
import json
import numpy as np
import socket
import subprocess
from datetime import datetime, timezone
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
from finn.transformation.fpgadataflow.set_folding import SetFolding
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.util.basic import getHWCustomOp
from finn.util.config import extract_model_config_to_json
from qonnx.transformation.general import GiveUniqueNodeNames
from tinydeit.common import (
    DEFAULT_BOARD,
    DEFAULT_BUILD_CSV,
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


DEFAULT_FOLDING_TARGET_CYCLES = 15000

FOLDING_HW_ATTRS = [
    "PE",
    "SIMD",
    "parallel_window",
    "ram_style",
    "resType",
    "mem_mode",
    "runtime_writeable_weights",
    "depth_trigger_uram",
    "depth_trigger_bram",
]


def _canonicalize_loop_body_names(model: ModelWrapper) -> ModelWrapper:
    model = model.transform(GiveUniqueNodeNames())
    for node in model.get_nodes_by_op_type("FINNLoop"):
        node_inst = getHWCustomOp(node)
        loop_body = node_inst.get_nodeattr("body")
        loop_body = loop_body.transform(GiveUniqueNodeNames(prefix=node.name + "_"))
        node_inst.set_nodeattr("body", loop_body.graph)
    return model


def step_tinydeit_post_transpose_parallelization(
    model: ModelWrapper, cfg: DataflowBuildConfig
) -> ModelWrapper:
    """Re-run folding after Shuffle decomposition creates final shuffle nodes.

    FINN's standard target-fps folding step runs before TinyDeiT's transpose
    decomposition.  The actual stitched design contains InnerShuffle/OuterShuffle
    nodes created later, so this final pass makes their SIMD and the surrounding
    RTL operators match the same aggressive target.
    """

    if not getattr(cfg, "tinydeit_post_transpose_folding", True):
        return model

    target_cycles_per_frame = cfg._resolve_cycles_per_frame()
    if target_cycles_per_frame is None:
        print("No target_fps provided, skipping step_tinydeit_post_transpose_parallelization.")
        return model

    model = model.transform(
        SetFolding(
            target_cycles_per_frame,
            mvau_wwidth_max=cfg.mvau_wwidth_max,
            two_pass_relaxation=cfg.folding_two_pass_relaxation,
        ),
        apply_to_subgraphs=True,
    )
    model = _canonicalize_loop_body_names(model)
    extract_model_config_to_json(
        model, cfg.output_dir + "/auto_folding_config.json", FOLDING_HW_ATTRS
    )
    return model


BUILD_STEPS_ESTIMATE = [
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    step_round_mlo_threshold_params,
    "step_transpose_decomposition",
    step_tinydeit_post_transpose_parallelization,
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
        if args.mode == "dcp":
            outputs.append(DataflowOutputType.OOC_SYNTH)
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

    cfg = DataflowBuildConfig(
        output_dir=str(output_dir),
        synth_clk_period_ns=args.clock_ns,
        board=args.board,
        target_fps=args.target_fps,
        mvau_wwidth_max=args.mvau_wwidth_max,
        folding_two_pass_relaxation=args.folding_two_pass_relaxation,
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
        # MLO rolling is performed explicitly in prepare_model.py before this
        # build starts. Keep cfg.mlo disabled here so build_dataflow does not
        # require loop-body metadata for another rolling pass.
        mlo=False,
        auto_fifo_depths=False,
        fifosim_n_inferences=args.fifosim_n_inferences,
        rtlsim_batch_size=args.rtlsim_batch_size,
        stitched_ip_gen_dcp=args.mode == "dcp" or args.stitched_ip_dcp,
        no_stdout_redirect=True,
        enable_build_pdb_debug=False,
    )
    cfg.tinydeit_post_transpose_folding = args.post_transpose_folding
    return cfg


BUILD_CSV_FIELDS = [
    "timestamp_utc",
    "hostname",
    "git_commit",
    "mode",
    "board",
    "clock_ns",
    "target_fps",
    "return_code",
    "timing_status",
    "wns_ns",
    "fmax_mhz",
    "estimated_throughput_fps",
    "resources",
    "folding_pe_simd",
    "build_step_times",
    "output_dir",
    "model_path",
    "report_dir",
    "stitched_ip_dir",
    "dcp_paths",
]


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open() as f:
        return json.load(f)


def _json_cell(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_path("."),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _folding_summary(config: dict) -> dict:
    attrs = ["PE", "SIMD", "TH", "MW", "MH", "mem_mode", "resType", "ram_style", "gemm_type"]
    summary = {}
    for node_name, node_cfg in sorted(config.items()):
        if node_name == "Defaults" or not isinstance(node_cfg, dict):
            continue
        node_summary = {attr: node_cfg[attr] for attr in attrs if attr in node_cfg}
        if any(attr in node_summary for attr in ["PE", "SIMD", "TH"]):
            summary[node_name] = node_summary
    return summary


def _parse_int_cell(cell: str) -> int:
    return int(cell.replace(",", "").strip())


def _partition_resource_summary(report_path: Path) -> dict:
    if not report_path.is_file():
        return {}

    key_map = {
        "Total LUTs": "stitched_LUT",
        "Logic LUTs": "stitched_Logic_LUT",
        "LUTRAMs": "stitched_LUTRAM",
        "SRLs": "stitched_SRL",
        "FFs": "stitched_FF",
        "RAMB36": "stitched_BRAM_36K",
        "RAMB18": "stitched_BRAM_18K",
        "URAM": "stitched_URAM",
        "DSP Blocks": "stitched_DSP",
    }
    headers = None
    with report_path.open() as f:
        for line in f:
            if not line.startswith("|"):
                continue
            cells = [cell.strip() for cell in line.split("|")[1:-1]]
            if len(cells) < 2:
                continue
            if cells[0] == "Instance" and cells[1] == "Module":
                headers = cells
                continue
            if cells[0] == "finn_design_wrapper" and headers is not None:
                return {
                    key_map[header]: _parse_int_cell(value)
                    for header, value in zip(headers, cells)
                    if header in key_map
                }
    return {}


def _resource_summary(ooc: dict, post_synth: dict, output_dir: Path) -> dict:
    resource_keys = [
        "LUT",
        "FF",
        "DSP",
        "BRAM",
        "BRAM_18K",
        "BRAM_36K",
        "URAM",
        "SRL",
        "total_power_W",
    ]
    source = ooc or post_synth or {}
    summary = {key: source[key] for key in resource_keys if key in source}
    partition_report = output_dir / "stitched_ip" / "finn_design_partition_util.rpt"
    summary.update(_partition_resource_summary(partition_report))
    return summary


def _timing_status(ooc: dict) -> str:
    if not ooc:
        return "not_run"
    wns = ooc.get("WNS")
    if wns is None:
        return "unknown"
    return "met" if float(wns) >= 0 else "failed"


def record_build_result(
    args: argparse.Namespace,
    output_dir: Path,
    model_path: Path,
    return_code: int,
) -> Path:
    output_dir = output_dir.resolve()
    report_dir = output_dir / "report"
    final_config = _load_json(output_dir / "final_hw_config.json")
    if not final_config:
        final_config = _load_json(output_dir / "auto_folding_config.json")
    ooc = _load_json(report_dir / "ooc_synth_and_timing.json")
    post_synth = _load_json(report_dir / "post_synth_resources.json")
    step_times = _load_json(output_dir / "time_per_step.json")
    dcp_paths = sorted(str(path.resolve()) for path in output_dir.glob("stitched_ip/**/*.dcp"))

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "git_commit": _git_commit(),
        "mode": args.mode,
        "board": args.board,
        "clock_ns": args.clock_ns,
        "target_fps": args.target_fps,
        "return_code": return_code,
        "timing_status": _timing_status(ooc),
        "wns_ns": ooc.get("WNS", ""),
        "fmax_mhz": ooc.get("fmax_mhz", ""),
        "estimated_throughput_fps": ooc.get("estimated_throughput_fps", ""),
        "resources": _json_cell(_resource_summary(ooc, post_synth, output_dir)),
        "folding_pe_simd": _json_cell(_folding_summary(final_config)),
        "build_step_times": _json_cell(step_times),
        "output_dir": str(output_dir),
        "model_path": str(model_path.resolve()),
        "report_dir": str(report_dir),
        "stitched_ip_dir": str((output_dir / "stitched_ip").resolve()),
        "dcp_paths": _json_cell(dcp_paths),
    }

    csv_path = repo_path(args.build_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.is_file()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BUILD_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return csv_path


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
    parser.add_argument(
        "--folding-target-cycles",
        type=int,
        default=DEFAULT_FOLDING_TARGET_CYCLES,
        help=(
            "Aggressive folding target in cycles/frame. The build converts this "
            "to an equivalent target FPS; set to 0 to use --target-fps directly."
        ),
    )
    parser.add_argument("--mvau-wwidth-max", type=int, default=10000)
    parser.add_argument(
        "--folding-two-pass-relaxation",
        dest="folding_two_pass_relaxation",
        action="store_true",
    )
    parser.add_argument(
        "--no-folding-two-pass-relaxation",
        dest="folding_two_pass_relaxation",
        action="store_false",
    )
    parser.add_argument(
        "--no-post-transpose-folding",
        dest="post_transpose_folding",
        action="store_false",
    )
    parser.add_argument(
        "--build-csv", default=str(DEFAULT_BUILD_CSV.relative_to(repo_path(".")))
    )
    parser.add_argument("--atol", type=float, default=1e-1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rtlsim-batch-size", type=int, default=1)
    parser.add_argument(
        "--fifosim-n-inferences",
        type=int,
        default=1,
        help=(
            "Number of inferences used by loop-body FIFO sizing simulation. "
            "TinyDeiT's loop body can stall in the stale tail of the second "
            "dummy inference, so the default keeps sizing to one inference."
        ),
    )
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
    parser.set_defaults(folding_two_pass_relaxation=False, post_transpose_folding=True)
    args = parser.parse_args()

    if args.folding_target_cycles and args.folding_target_cycles > 0:
        target_cycles_per_sec = 10**9 / args.clock_ns
        target_fps_from_cycles = int(round(target_cycles_per_sec / args.folding_target_cycles))
        args.target_fps = max(args.target_fps, target_fps_from_cycles)

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
    ret = -1
    try:
        ret = build.build_dataflow_cfg(str(model_path), cfg)
    except Exception:
        csv_path = record_build_result(args, output_dir, model_path, ret)
        print(f"Build CSV: {csv_path}")
        print(f"Build output: {output_dir}")
        raise
    else:
        csv_path = record_build_result(args, output_dir, model_path, ret)
        print(f"Build CSV: {csv_path}")
        print(f"Build output: {output_dir}")
        if ret != 0:
            raise SystemExit(ret)


if __name__ == "__main__":
    main()
