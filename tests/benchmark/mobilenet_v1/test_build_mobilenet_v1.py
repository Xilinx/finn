# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import os

# custom steps for mobilenetv1
from custom_steps_mobilenet import (
    step_mobilenet_slr_floorplan,
    step_mobilenet_streamline,
)

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import make_build_dir

build_fd = "tests/benchmark/"

# model
model_name = "mobilenetv1-w4a4"
model_file = build_fd + "models/%s_pre_post_tidy_opset-11.onnx" % model_name


# verification parameters
verify_input_npy = build_fd + "verification_io/" + model_name + "_input.npy"
verify_expected_output_npy = build_fd + "verification_io/" + model_name + "_output.npy"


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


# build output products
build_outputs = [
    build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
    build_cfg.DataflowOutputType.STITCHED_IP,
    build_cfg.DataflowOutputType.PYNQ_DRIVER,
    build_cfg.DataflowOutputType.BITFILE,
    build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
    build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
]


# Build steps: the MobileNet-specific streamline (handles depthwise convs via
# MoveMulPastDWConv, QuantAvgPool datalayout, flatten reordering, then conv
# lowering) replaces the standard streamline phase. The rest uses the phase-based
# default flow. The U55C SLR floorplan is injected before bitfile synthesis
# (see configure_build).
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


def platform_to_shell(platform):
    if platform in ["U55C"]:
        return build_cfg.ShellFlowType.VITIS_ALVEO
    elif platform in ["ZCU104"]:
        return build_cfg.ShellFlowType.VIVADO_ZYNQ
    else:
        raise Exception("Unknown platform, can't determine ShellFlowType")


def configure_build(board, output_dir):
    f_file = f"{build_fd}mobilenet_v1/folding_config/mobilenet_folding_config_{board}"
    sl_file = f"{build_fd}mobilenet_v1/specialize_layers_config/mobilenet_specialize_layers_{board}"
    # U55C (Alveo) applies SLR floorplanning before bitfile synthesis
    inject_before = {}
    if board == "U55C":
        inject_before = {"step_synthesize_bitfile": [step_mobilenet_slr_floorplan]}
    cfg = build_cfg.DataflowBuildConfig(
        generate_outputs=build_outputs,
        output_dir=output_dir,
        steps=build_steps,
        inject_steps_before=inject_before,
        folding_config_file=f_file + ".json",
        synth_clk_period_ns=select_clk_period(board),
        board=board,
        shell_flow_type=platform_to_shell(board),
        auto_fifo_depths=False,
        specialize_layers_config_file=sl_file + ".json",
        standalone_thresholds=True,
        verify_steps=select_verif_steps(board),
        verify_input_npy=verify_input_npy,
        verify_expected_output_npy=verify_expected_output_npy,
    )
    return cfg


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
def test_mobilenetv1(board):
    # Create output directory only when test actually runs
    output_dir = make_build_dir("build_mobilenet_v1_")

    # Run build flow
    cfg = configure_build(board, output_dir)
    build.build_dataflow_cfg(model_file, cfg)

    # Check if the ezxpected output products are there
    assert os.path.isfile(output_dir + "/time_per_step.json")
    assert os.path.isfile(output_dir + "/final_hw_config.json")
    assert os.path.isfile(output_dir + "/template_specialize_layers_config.json")
    assert os.path.isfile(output_dir + "/stitched_ip/ip/component.xml")
    assert os.path.isfile(output_dir + "/driver/driver.py")
    assert os.path.isfile(output_dir + "/report/estimate_layer_cycles.json")
    assert os.path.isfile(output_dir + "/report/estimate_layer_resources.json")
    assert os.path.isfile(output_dir + "/report/estimate_network_performance.json")
    assert os.path.isfile(output_dir + "/report/rtlsim_performance.json")
    assert os.path.isfile(output_dir + "/bitfile/finn-accel.bit")
    assert os.path.isfile(output_dir + "/bitfile/finn-accel.hwh")
    assert os.path.isfile(output_dir + "/report/post_synth_resources.xml")
    assert os.path.isfile(output_dir + "/report/post_route_timing.rpt")
    assert os.path.isfile(output_dir + "/report/post_synth_resources.json")
    # Verification outputs
    verify_out_dir = output_dir + "/verification_output"
    assert os.path.isfile(verify_out_dir + "/verify_streamlined_python_0_SUCCESS.npy")
    assert os.path.isfile(verify_out_dir + "/verify_folded_hls_cppsim_0_SUCCESS.npy")
    assert os.path.isfile(verify_out_dir + "/verify_node_by_node_rtlsim_0_SUCCESS.npy")
    # ZCU104 skips stitched_ip_rtlsim due to URAM initialization issues
    if board != "ZCU104":
        assert os.path.isfile(verify_out_dir + "/verify_stitched_ip_rtlsim_0_SUCCESS.npy")
