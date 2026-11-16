# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

# custom steps for the COCO YOLOv8
import os
from benchmark_helpers import (
    CORE_BUILD_OUTPUT_FILES,
    benchmark_root,
    bitfile_output_files,
    check_build_outputs,
    get_verify_steps,
    make_benchmark_cfg,
    verification_io_dir,
)
from custom_steps_yolov8 import step_yolov8_streamline

import finn.builder.build_dataflow as build
from finn.util.basic import make_build_dir


# model
def get_model_file(model):
    return os.path.join(benchmark_root(), "models", model + "_comact_tidy.onnx")


# verification parameters
# The verification I/O is a fixed-seed random uint8 192x320 image and the
# corresponding output of the original (pre-streamline) model.
def get_verify_io(model):
    return (
        os.path.join(verification_io_dir(), model + "_input.npy"),
        os.path.join(verification_io_dir(), model + "_output.npy"),
    )


# initial_python is not listed: the models are already FINN-ONNX (MultiThreshold
# activations, UINT8 input annotation), so step_qonnx_to_finn/step_tidy_up are not
# part of the flow.
verif_steps = [
    "streamlined_python",
    "folded_hls_cppsim",
    "node_by_node_rtlsim",
    "stitched_ip_rtlsim",
]


# Per-precision settings on top of the shared config. The 8-bit activations are
# implemented as Requant instead of 255-threshold Thresholding layers, which would
# otherwise dominate the LUT budget.
MODEL_OVERRIDES = {
    "quantyolov8_4w4a": {},
    "quantyolov8_8w8a": {"requant_bitwidth_threshold": 8},
}

# Build steps: the models are already FINN-ONNX, so the flow starts directly with
# the YOLOv8-specific streamline (C2f split/concat + SPPF handling + conv
# lowering), which replaces the standard prepare/streamline phases. The rest uses
# the phase-based default flow.
yolov8_build_steps = [
    step_yolov8_streamline,
    "phase_convert_to_hardware",
    "phase_optimize_hardware",
    "phase_build_hardware",
    "phase_generate_outputs",
]


def configure_build(board, model, output_dir, **overrides):
    # Folding was derived from target_fps=90 auto-folding. No specialize config:
    # the default layer specialization is used.
    folding = os.path.join(
        benchmark_root(), "yolov8", "folding_config", f"{model}_folding_config_{board}.json"
    )
    verify_input_npy, verify_expected_output_npy = get_verify_io(model)
    cfg = dict(
        steps=yolov8_build_steps,
        folding_config_file=folding,
        synth_clk_period_ns=10.0,
        auto_fifo_depths=True,
        standalone_thresholds=True,
        verify_steps=get_verify_steps(verif_steps),
        verify_input_npy=verify_input_npy,
        verify_expected_output_npy=verify_expected_output_npy,
    )
    cfg.update(MODEL_OVERRIDES[model])
    cfg.update(overrides)
    return make_benchmark_cfg(board, output_dir, **cfg)


YOLOV8_BUILDS = [
    ("U55C", "quantyolov8_4w4a"),
    ("U55C", "quantyolov8_8w8a"),
]


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.parametrize("board,model", YOLOV8_BUILDS)
def test_yolov8(board, model, bench_recorder):
    output_dir = make_build_dir(f"build_yolov8_{model}_{board}_")

    # Run build flow
    cfg = configure_build(board, model, output_dir)
    build.build_dataflow_cfg(get_model_file(model), cfg)
    bench_recorder(model, board, output_dir)

    # Check that all expected output products are present, reporting every
    # missing artifact at once instead of aborting on the first one.
    build_output_files = CORE_BUILD_OUTPUT_FILES + bitfile_output_files(board)
    check_build_outputs(output_dir, build_output_files)
