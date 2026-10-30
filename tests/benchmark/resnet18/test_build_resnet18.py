# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

# custom steps for the CIFAR-10 ResNet-18
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
from custom_steps_resnet18 import step_resnet18_attach_preproc, step_resnet18_streamline

import finn.builder.build_dataflow as build
from finn.util.basic import make_build_dir


# model
def get_model_file(model):
    return os.path.join(benchmark_root(), "models", model + "-qonnx.onnx")


# verification parameters
# The verification I/O is a (raw uint8 CIFAR-10 image, top-1 class label) pair.
# Both ResNet-18 precisions classify the BNN-PYNQ cnv sample to the same class as
# cnv does, so the cnv CIFAR-10 I/O is reused directly -- no ResNet-18-specific
# golden file.
verify_input_npy = os.path.join(verification_io_dir(), "cnv_cifar10_input.npy")
verify_expected_output_npy = os.path.join(verification_io_dir(), "cnv_cifar10_output.npy")


# finn_onnx_python is not listed: the ToTensor preprocessing is merged only after
# step_qonnx_to_finn, so that step's graph does not consume raw uint8 images yet.
def select_verif_steps(model):
    steps = [
        "initial_python",
        "streamlined_python",
        "folded_hls_cppsim",
        "node_by_node_rtlsim",
        "stitched_ip_rtlsim",
    ]
    # Skip stitched_ip_rtlsim for 8w8a due to URAM initialization issues (its
    # layer4 weights live in runtime-writeable URAM)
    if model == "resnet18_8w8a":
        steps.remove("stitched_ip_rtlsim")
    return steps


# Per-precision settings on top of the shared config. The 8-bit activations are
# implemented as Requant instead of 255-threshold Thresholding layers, which would
# otherwise dominate the LUT budget.
MODEL_OVERRIDES = {
    "resnet18_4w4a": {},
    "resnet18_8w8a": {"requant_bitwidth_threshold": 8},
}

# Build steps: the preprocessing/TopK is attached right after the QONNX->FINN
# conversion, and the ResNet-18-specific streamline (residual fork/join handling +
# conv lowering) replaces the standard streamline phase. The rest uses the
# phase-based default flow.
resnet18_build_steps = [
    "step_qonnx_to_finn",
    step_resnet18_attach_preproc,
    "step_tidy_up",
    step_resnet18_streamline,
    "phase_convert_to_hardware",
    "phase_optimize_hardware",
    "phase_build_hardware",
    "phase_generate_outputs",
]


def configure_build(board, model, output_dir, **overrides):
    # Folding was derived from target_fps=5000 auto-folding. On U55C the 8w8a
    # weights (~89 Mbit) exceed the BRAM capacity, so its large layer4 weight
    # memories are placed in URAM, which requires runtime-writeable weights on
    # UltraScale+; 4w4a fits entirely in BRAM.
    # No specialize config: the default layer specialization is used.
    folding = os.path.join(
        benchmark_root(), "resnet18", "folding_config", f"{model}_folding_config_{board}.json"
    )
    cfg = dict(
        steps=resnet18_build_steps,
        folding_config_file=folding,
        synth_clk_period_ns=4.0,
        auto_fifo_depths=True,
        standalone_thresholds=True,
        verify_steps=get_verify_steps(select_verif_steps(model)),
        verify_input_npy=verify_input_npy,
        verify_expected_output_npy=verify_expected_output_npy,
    )
    cfg.update(MODEL_OVERRIDES[model])
    cfg.update(overrides)
    return make_benchmark_cfg(board, output_dir, **cfg)


RESNET18_BUILDS = [
    ("U55C", "resnet18_4w4a"),
    ("U55C", "resnet18_8w8a"),
]


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.parametrize("board,model", RESNET18_BUILDS)
def test_resnet18(board, model, bench_recorder):
    output_dir = make_build_dir(f"build_resnet18_{model}_{board}_")

    # Run build flow
    cfg = configure_build(board, model, output_dir)
    build.build_dataflow_cfg(get_model_file(model), cfg)
    bench_recorder(model, board, output_dir)

    # Check that all expected output products are present, reporting every
    # missing artifact at once instead of aborting on the first one.
    build_output_files = CORE_BUILD_OUTPUT_FILES + bitfile_output_files(board)
    check_build_outputs(output_dir, build_output_files)
