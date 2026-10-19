# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

# custom steps for resnet50v1.5
import os
from benchmark_helpers import (
    ESTIMATE_OUTPUTS,
    ESTIMATE_PHASE,
    FEATURE_OUTPUTS,
    benchmark_config_paths,
    benchmark_root,
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
from custom_steps_resnet50 import step_resnet50_streamline, step_resnet50_tidy

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import make_build_dir

build_flow_folder = benchmark_root()

# model
model_name = "resnet50_w1a2_exported"
model_file = os.path.join(build_flow_folder, "models", model_name + ".onnx")

# verification parameters
verify_input_npy = os.path.join(verification_io_dir(), model_name + "_input.npy")
verify_expected_output_npy = os.path.join(verification_io_dir(), model_name + "_output.npy")

verif_steps = [
    "initial_python",
    "streamlined_python",
    "folded_hls_cppsim",
    "node_by_node_rtlsim",
    "stitched_ip_rtlsim",
]

# build output products
# TEMP(stitched-ip-only): the V80/SLASH synth needs the slashkit .deb + Vivado
# 2025.1, so for now we stop at stitched IP (+ rtlsim performance). The output
# products gate which steps actually execute, so leaving BITFILE/PYNQ_DRIVER/
# DEPLOYMENT_PACKAGE out skips synth/driver/deployment. Re-add them once the
# SLASH toolchain is in place.
build_outputs = [
    build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
    build_cfg.DataflowOutputType.STITCHED_IP,
    build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
]

# ResNet-50 uses custom tidy/streamline steps; the rest is phase-based. The
# custom streamline step lowers convolutions and fully detangles the residual
# fork/join transposes.
resnet50_build_steps = [
    step_resnet50_tidy,
    step_resnet50_streamline,
    "phase_convert_to_hardware",
    "phase_optimize_hardware",
    "phase_build_hardware",
    "phase_generate_outputs",
]


def configure_build(board, output_dir, **overrides):
    folding, specialize = benchmark_config_paths(
        "resnet50",
        f"resnet50_folding_config_{board}",
        f"resnet50_specialize_layers_{board}",
    )
    cfg = dict(
        steps=resnet50_build_steps,
        standalone_thresholds=True,
        # TEMP(stitched-ip-only): stop at stitched IP (+ rtlsim performance) until
        # the V80/SLASH synth toolchain is available.
        generate_outputs=build_outputs,
        folding_config_file=folding,
        specialize_layers_config_file=specialize,
        auto_fifo_depths=True,
        synth_clk_period_ns=4.0,
        # TEMP(stitched-ip-only): V80 (Versal) uses the SLASH shell, which requires
        # Vivado 2025.1. The builder resolves V80 -> SLASH_ALVEO only at synth time;
        # since this run stops at stitched IP (no BITFILE), synth never runs and the
        # shell is never consulted, so it is left unset here.
        verify_steps=get_verify_steps(verif_steps),
        verify_input_npy=verify_input_npy,
        verify_expected_output_npy=verify_expected_output_npy,
    )
    cfg.update(overrides)
    return make_benchmark_cfg(board, output_dir, **cfg)


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.parametrize("board", ["V80"])
def test_resnet50(board, bench_recorder):
    output_dir = make_build_dir("build_resnet50_")

    # Run build flow
    cfg = configure_build(board, output_dir)
    build.build_dataflow_cfg(model_file, cfg)
    bench_recorder(model_name, board, output_dir)

    # Check that all expected output products (and the per-step verification
    # markers) are present, reporting every missing artifact at once instead of
    # aborting on the first one.
    # TEMP(stitched-ip-only): bitfile/driver/deployment artifacts are omitted
    # while the V80/SLASH synth toolchain is unavailable; re-add
    # bitfile_output_files(SLASH_ALVEO) and driver/driver.py once enabled.
    build_output_files = [
        "time_per_step.json",
        "final_hw_config.json",
        "template_specialize_layers_config.json",
        "stitched_ip/ip/component.xml",
        "report/estimate_layer_cycles.json",
        "report/estimate_layer_resources.json",
        "report/estimate_network_performance.json",
        "report/rtlsim_performance.json",
    ]
    check_build_outputs(output_dir, build_output_files)


# --- feature tests: FIFO sizing, folding, DWC padding -----------------------
# Each test resumes the convert-phase checkpoint of a cached estimate-only build
# (committed folding) with the feature's settings, builds the stitched IP and
# checks it in rtlsim: output equal to the golden output and throughput within
# 2 % of the build's own analytic estimate.

FEATURE_BUILDS = ["V80"]


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
        f"est_resnet50_{board}_", build_estimate, "report/estimate_network_performance.json"
    )


def run_feature_build(board, name, estimate_only=False, **overrides):
    """Resume the reference checkpoint with ``overrides`` and, unless
    ``estimate_only``, check the stitched IP in rtlsim. Returns the metrics."""
    ref_dir = reference_estimate(board)
    output_dir = make_build_dir(f"{name}_resnet50_{board}_")
    cfg = configure_build(
        board,
        output_dir,
        steps=feature_steps(resnet50_build_steps),
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
@pytest.mark.xdist_group("resnet50")
@pytest.mark.parametrize("strategy", ["heuristic_analytical", "heuristic_rtlsim"])
@pytest.mark.parametrize("board", FEATURE_BUILDS)
def test_resnet50_fifo_sizing(board, strategy):
    run_feature_build(board, f"fifo-{strategy}", auto_fifo_depths=True, auto_fifo_strategy=strategy)
