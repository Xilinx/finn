# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

# custom steps for mobilenetv1
import os
from benchmark_helpers import (
    CORE_BUILD_OUTPUT_FILES,
    ESTIMATE_OUTPUTS,
    ESTIMATE_PHASE,
    FEATURE_OUTPUTS,
    benchmark_config_paths,
    benchmark_root,
    bitfile_output_files,
    cached_build,
    check_build_outputs,
    feature_metrics,
    feature_steps,
    get_verify_steps,
    make_benchmark_cfg,
    resume_checkpoint,
    timed_steps,
    verification_io_dir,
)
from custom_steps_mobilenet import step_mobilenet_streamline

import finn.builder.build_dataflow as build
from finn.util.basic import make_build_dir

build_fd = benchmark_root()

# model
model_name = "mobilenetv1-w4a4"
model_file = os.path.join(build_fd, "models", "%s_pre_post_tidy_opset-11.onnx" % model_name)


# verification parameters
verify_input_npy = os.path.join(verification_io_dir(), model_name + "_input.npy")
verify_expected_output_npy = os.path.join(verification_io_dir(), model_name + "_output.npy")


def select_verif_steps(platform):
    steps = [
        "streamlined_python",
        "folded_hls_cppsim",
        "node_by_node_rtlsim",
        "stitched_ip_rtlsim",
    ]
    # Skip stitched_ip_rtlsim for ZCU104 due to URAM initialization issues
    if platform == "ZCU104":
        steps.remove("stitched_ip_rtlsim")
    return steps


# Build steps: the MobileNet-specific streamline (handles depthwise convs via
# MoveMulPastDWConv, QuantAvgPool datalayout, flatten reordering, then conv
# lowering) replaces the standard streamline phase. The rest uses the phase-based
# default flow.
build_steps = [
    "phase_prepare_model",
    step_mobilenet_streamline,
    "phase_convert_to_hardware",
    "phase_optimize_hardware",
    "phase_build_hardware",
    "phase_generate_outputs",
]


# select target clock frequency
def select_clk_period(platform):
    if platform in ["ZCU104"]:
        return 5.4
    elif platform in ["U55C"]:
        return 3.0


def configure_build(board, output_dir, **overrides):
    folding, specialize = benchmark_config_paths(
        "mobilenet_v1",
        f"mobilenet_folding_config_{board}",
        f"mobilenet_specialize_layers_{board}",
    )
    cfg = dict(
        steps=build_steps,
        folding_config_file=folding,
        specialize_layers_config_file=specialize,
        synth_clk_period_ns=select_clk_period(board),
        auto_fifo_depths=True,
        standalone_thresholds=True,
        verify_steps=get_verify_steps(select_verif_steps(board)),
        verify_input_npy=verify_input_npy,
        verify_expected_output_npy=verify_expected_output_npy,
    )
    cfg.update(overrides)
    return make_benchmark_cfg(board, output_dir, **cfg)


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.parametrize(
    "board",
    [
        "ZCU104",
        "U55C",
    ],
)
def test_mobilenetv1(board, bench_recorder):
    # Create output directory only when test actually runs
    output_dir = make_build_dir(f"build_mobilenet_v1_{board}_")

    # Run build flow
    cfg = configure_build(board, output_dir)
    build.build_dataflow_cfg(model_file, cfg)
    bench_recorder(model_name, board, output_dir)

    # Check that all expected output products (and the per-step verification
    # markers) are present, reporting every missing artifact at once instead of
    # aborting on the first one. The bitfile artifacts depend on the shell flow
    # (ZCU104 -> .bit/.hwh via Vivado/Zynq, U55C -> .xclbin via Vitis/Alveo).
    build_output_files = CORE_BUILD_OUTPUT_FILES + bitfile_output_files(board)
    check_build_outputs(output_dir, build_output_files)


# --- feature tests: FIFO sizing, folding, DWC padding -----------------------
# Each test resumes the convert-phase checkpoint of a cached estimate-only build
# (committed folding) with the feature's settings, builds the stitched IP and
# checks it in rtlsim: output equal to the golden output and throughput within
# 2 % of the build's own analytic estimate.

# ZCU104 is left out: its stitched-IP rtlsim gives X outputs (URAM init)
FEATURE_BUILDS = ["U55C"]


def reference_estimate(board):
    """Estimate-only build with the committed folding: its convert-phase
    checkpoint and estimated throughput are the feature tests' reference."""

    def build_estimate(output_dir):
        cfg = configure_build(
            board,
            output_dir,
            generate_outputs=ESTIMATE_OUTPUTS,
            stop_step=ESTIMATE_PHASE,
            verify_steps=[],
        )
        assert build.build_dataflow_cfg(model_file, cfg) == 0

    return cached_build(
        f"est_mobilenet_v1_{board}_", build_estimate, "report/estimate_network_performance.json"
    )


def run_feature_build(board, name, estimate_only=False, **overrides):
    """Resume the reference checkpoint with ``overrides`` and, unless
    ``estimate_only``, check the stitched IP in rtlsim. Returns the metrics."""
    ref_dir = reference_estimate(board)
    output_dir = make_build_dir(f"{name}_mobilenet_v1_{board}_")
    cfg = configure_build(
        board,
        output_dir,
        steps=feature_steps(build_steps),
        generate_outputs=ESTIMATE_OUTPUTS if estimate_only else FEATURE_OUTPUTS,
        stop_step=ESTIMATE_PHASE if estimate_only else None,
        verify_steps=[] if estimate_only else ["stitched_ip_rtlsim"],
        stitched_ip_gen_dcp=False,
        save_intermediate_models=True,
        **overrides,
    )
    # make_benchmark_cfg drops None values: apply them afterwards (e.g. no folding JSON)
    for key, value in overrides.items():
        if value is None:
            setattr(cfg, key, None)
    with timed_steps() as step_times:
        ret = build.build_dataflow_cfg(resume_checkpoint(ref_dir, output_dir), cfg)
    metrics = feature_metrics(output_dir, ref_dir, test=name, step_times_s=step_times)
    assert ret == 0, f"build failed, see {output_dir}/build_dataflow.log"
    if not estimate_only:
        assert metrics["verified"], "stitched-IP rtlsim output differs from the golden output"
        assert (metrics["rtlsim_vs_est"] or 0) >= 0.98, "rtlsim %s fps, estimate %s fps" % (
            metrics["rtlsim_fps"],
            metrics["est_fps"],
        )
    return metrics


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.fifo_sizing
@pytest.mark.xdist_group("mobilenet_v1")
@pytest.mark.parametrize("strategy", ["heuristic_analytical", "heuristic_rtlsim"])
@pytest.mark.parametrize("board", FEATURE_BUILDS)
def test_mobilenetv1_fifo_sizing(board, strategy):
    run_feature_build(board, f"fifo-{strategy}", auto_fifo_depths=True, auto_fifo_strategy=strategy)
