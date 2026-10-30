# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

# custom steps for the BSD300 ESPCN super-resolution network
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
from custom_steps_espcn import (
    step_espcn_attach_preproc,
    step_espcn_lower_deconv,
    step_espcn_streamline,
    step_espcn_subpixel_to_deconv,
)

import finn.builder.build_dataflow as build
from finn.util.basic import make_build_dir

# model
model_name = "espcn-bsd300"
model_file = os.path.join(benchmark_root(), "models", model_name + ".onnx")

# verification parameters
# The verification I/O is a raw uint8 128x128 BSD300 crop and the corresponding
# 2x upscaled 256x256 output of the PyTorch model.
verify_input_npy = os.path.join(verification_io_dir(), model_name + "_input.npy")
verify_expected_output_npy = os.path.join(verification_io_dir(), model_name + "_output.npy")

verif_steps = [
    "finn_onnx_python",
    "initial_python",
    "streamlined_python",
    "folded_hls_cppsim",
    "node_by_node_rtlsim",
    "stitched_ip_rtlsim",
]

# Build steps: before the QONNX->FINN conversion, the sub-pixel convolution is
# rewritten as a ConvTranspose and the ToTensor preprocessing is attached. The
# ESPCN-specific streamline replaces the standard streamline phase, and the
# ConvTranspose is lowered (FMPadding_Pixel + Im2Col + MatMul) before the
# phase-based default flow converts the rest to HW layers.
espcn_build_steps = [
    step_espcn_subpixel_to_deconv,
    step_espcn_attach_preproc,
    "phase_prepare_model",
    step_espcn_streamline,
    step_espcn_lower_deconv,
    "phase_convert_to_hardware",
    "phase_optimize_hardware",
    "phase_build_hardware",
    "phase_generate_outputs",
]


def configure_build(board, output_dir, **overrides):
    # Folding is hand-tuned (~381 fps estimated): every layer is balanced at
    # ~524k cycles/frame. The MVAU weight memories are only 8-32 words deep at
    # this parallelism, so they are placed in LUTRAM instead of BRAM. (Default
    # auto-folding stalls at ~10.6 fps: mvau_wwidth_max caps the deconv MVAU.)
    # No specialize config: the default layer specialization is used.
    folding = os.path.join(
        benchmark_root(), "espcn", "folding_config", f"espcn_folding_config_{board}.json"
    )
    cfg = dict(
        steps=espcn_build_steps,
        folding_config_file=folding,
        synth_clk_period_ns=5.0,
        auto_fifo_depths=True,
        standalone_thresholds=True,
        verify_steps=get_verify_steps(verif_steps),
        verify_input_npy=verify_input_npy,
        verify_expected_output_npy=verify_expected_output_npy,
    )
    cfg.update(overrides)
    return make_benchmark_cfg(board, output_dir, **cfg)


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.parametrize("board", ["KV260_SOM"])
def test_espcn(board, bench_recorder):
    output_dir = make_build_dir(f"build_espcn_{board}_")

    # Run build flow
    cfg = configure_build(board, output_dir)
    build.build_dataflow_cfg(model_file, cfg)
    bench_recorder(model_name, board, output_dir)

    # Check that all expected output products are present, reporting every
    # missing artifact at once instead of aborting on the first one.
    build_output_files = CORE_BUILD_OUTPUT_FILES + bitfile_output_files(board)
    check_build_outputs(output_dir, build_output_files)
