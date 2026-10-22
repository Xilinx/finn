# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

# custom steps for vgg10-radioml
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
from custom_steps_vgg10 import step_pre_streamline

import finn.builder.build_dataflow as build
from finn.util.basic import make_build_dir

build_flow_folder = benchmark_root()

# model
model_name = "radioml_w4a4_small_tidy"
model_file = os.path.join(build_flow_folder, "models", "%s.onnx" % model_name)


# verification parameters
verify_input_npy = os.path.join(verification_io_dir(), model_name + "_input.npy")
verify_expected_output_npy = os.path.join(verification_io_dir(), model_name + "_output.npy")

verif_steps = [
    "streamlined_python",
    "folded_hls_cppsim",
    "node_by_node_rtlsim",
    "stitched_ip_rtlsim",
]

# vgg10-radioml uses one custom step, step_pre_streamline (3D->4D + fold scalar
# mul/add into TopK), run between tidy and streamline. Everything else is
# phase-based: the standard convert-to-hw now infers the label-select and
# elementwise layers this model used to convert by hand.
build_steps = [
    "step_tidy_up",
    step_pre_streamline,
    "phase_optimize_model",
    "phase_convert_to_hardware",
    "phase_optimize_hardware",
    "phase_build_hardware",
    "phase_generate_outputs",
]


def configure_build(board, output_dir, **overrides):
    folding, specialize = benchmark_config_paths(
        "vgg10-radioml",
        "vgg10radioml_folding_config",
        "vgg10radioml_specialize_layers",
    )
    cfg = dict(
        folding_config_file=folding,
        specialize_layers_config_file=specialize,
        synth_clk_period_ns=4.0,
        steps=build_steps,
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
@pytest.mark.parametrize("board", ["ZCU104"])
def test_vgg10radioml(board, bench_recorder):
    output_dir = make_build_dir("build_vgg10-radioml_")

    # Run build flow
    cfg = configure_build(board, output_dir)
    build.build_dataflow_cfg(model_file, cfg)
    bench_recorder(model_name, board, output_dir)

    # Check that all expected output products (and the per-step verification
    # markers) are present, reporting every missing artifact at once instead of
    # aborting on the first one.
    build_output_files = CORE_BUILD_OUTPUT_FILES + bitfile_output_files(board)
    check_build_outputs(output_dir, build_output_files)
