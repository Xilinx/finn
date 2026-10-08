# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

# custom steps for mobilenetv1
import os
from benchmark_helpers import (
    CORE_BUILD_OUTPUT_FILES,
    benchmark_config_paths,
    benchmark_root,
    bitfile_output_files,
    check_build_outputs,
    get_verify_steps,
    make_benchmark_cfg,
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
