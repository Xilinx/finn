# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

# custom steps for resnet50v1.5
import os
from benchmark_helpers import (
    benchmark_root,
    check_build_outputs,
    get_verify_steps,
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


def configure_build(board, output_dir):
    cfg = build_cfg.DataflowBuildConfig(
        # non-interactive run: surface the real error instead of dropping into pdb
        enable_build_pdb_debug=False,
        steps=resnet50_build_steps,
        standalone_thresholds=True,
        generate_outputs=build_outputs,
        output_dir=output_dir,
        folding_config_file=os.path.join(
            build_flow_folder,
            "resnet50",
            "folding_config",
            f"resnet50_folding_config_{board}.json",
        ),
        auto_fifo_depths=True,
        synth_clk_period_ns=4.0,
        board=board,
        # TEMP(stitched-ip-only): V80 (Versal) uses the SLASH shell for bitfile
        # generation, but SLASH requires Vivado 2025.1. Since this run stops at
        # stitched IP (no BITFILE), shell_flow_type is only consulted during synth,
        # so it is left unset to avoid the Vivado-version check. Restore
        # shell_flow_type=build_cfg.ShellFlowType.SLASH_ALVEO when synthesizing.
        specialize_layers_config_file=os.path.join(
            build_flow_folder,
            "resnet50",
            "specialize_layers_config",
            f"resnet50_specialize_layers_{board}.json",
        ),
        verify_steps=get_verify_steps(verif_steps),
        verify_input_npy=verify_input_npy,
        verify_expected_output_npy=verify_expected_output_npy,
    )
    return cfg


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
