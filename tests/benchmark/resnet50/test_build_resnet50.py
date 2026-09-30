############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

import pytest

# custom steps for resnet50v1.5
from benchmark_helpers import (
    bitfile_output_files,
    check_build_outputs,
    get_verify_steps,
)
from custom_steps_resnet50 import (
    step_resnet50_slr_floorplan,
    step_resnet50_streamline,
    step_resnet50_tidy,
)

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import make_build_dir, vitis_default_platform

build_flow_folder = "tests/benchmark/"

# model
model_name = "resnet50_w1a2_exported"
model_file = build_flow_folder + "models/" + model_name + ".onnx"

# verification parameters
verify_input_npy = build_flow_folder + "verification_io/" + model_name + "_input.npy"
verify_expected_output_npy = build_flow_folder + "verification_io/" + model_name + "_output.npy"

verif_steps = [
    "initial_python",
    "streamlined_python",
    "folded_hls_cppsim",
    "node_by_node_rtlsim",
    "stitched_ip_rtlsim",
]

# build output products
build_outputs = [
    build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
    build_cfg.DataflowOutputType.STITCHED_IP,
    build_cfg.DataflowOutputType.PYNQ_DRIVER,
    build_cfg.DataflowOutputType.BITFILE,
    build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
    build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
]

# ResNet-50 uses custom tidy/streamline steps; the rest is phase-based. The
# custom streamline step lowers convolutions and fully detangles the residual
# fork/join transposes.
# The SLR floorplan is injected before bitfile synthesis.
resnet50_build_steps = [
    step_resnet50_tidy,
    step_resnet50_streamline,
    "phase_convert_to_hardware",
    "phase_optimize_hardware",
    "phase_build_hardware",
    "phase_generate_outputs",
]


def configure_build(board, output_dir):
    cfg = build_cfg.DataflowBuildConfig(
        # non-interactive run: surface the real error instead of dropping into pdb
        enable_build_pdb_debug=False,
        steps=resnet50_build_steps,
        inject_steps_before={"step_synthesize_bitfile": [step_resnet50_slr_floorplan]},
        standalone_thresholds=True,
        generate_outputs=build_outputs,
        output_dir=output_dir,
        folding_config_file=(
            f"{build_flow_folder}resnet50/folding_config/" f"resnet50_folding_config_{board}.json"
        ),
        auto_fifo_depths=True,
        synth_clk_period_ns=4.0,
        board=board,
        shell_flow_type=build_cfg.ShellFlowType.VITIS_ALVEO,
        vitis_platform=vitis_default_platform[board],
        vitis_opt_strategy=build_cfg.VitisOptStrategyCfg.PERFORMANCE_BEST,
        specialize_layers_config_file=(
            f"{build_flow_folder}resnet50/specialize_layers_config/"
            f"resnet50_specialize_layers_{board}.json"
        ),
        verify_steps=get_verify_steps(verif_steps),
        verify_input_npy=verify_input_npy,
        verify_expected_output_npy=verify_expected_output_npy,
    )
    return cfg


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.parametrize("board", ["U250"])
def test_resnet50(board):
    output_dir = make_build_dir("build_resnet50_")

    # Run build flow
    cfg = configure_build(board, output_dir)
    build.build_dataflow_cfg(model_file, cfg)

    # Check that all expected output products (and the per-step verification
    # markers) are present, reporting every missing artifact at once instead of
    # aborting on the first one. ResNet-50 targets U250 via the Vitis/Alveo flow,
    # which emits a .xclbin (no .bit/.hwh/timing report).
    build_output_files = [
        "time_per_step.json",
        "final_hw_config.json",
        "template_specialize_layers_config.json",
        "stitched_ip/ip/component.xml",
        "driver/driver.py",
        "report/estimate_layer_cycles.json",
        "report/estimate_layer_resources.json",
        "report/estimate_network_performance.json",
        "report/rtlsim_performance.json",
    ] + bitfile_output_files(build_cfg.ShellFlowType.VITIS_ALVEO)
    check_build_outputs(output_dir, build_output_files)
