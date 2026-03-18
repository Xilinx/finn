############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

import pytest

import os
import re

# custom steps for mobilenetv1
from custom_steps import (
    step_mobilenet_convert_to_hw_layers,
    step_mobilenet_convert_to_hw_layers_separate_th,
    step_mobilenet_lower_convs,
    step_mobilenet_slr_floorplan,
    step_mobilenet_streamline,
)

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import make_build_dir

build_fd = "tests/benchmark/"
output_dir = make_build_dir("build_mobilenet-v1_")

# model
model_name = "mobilenetv1-w4a4"
model_file = build_fd + "models/%s_pre_post_tidy_opset-11.onnx" % model_name


# verification parameters
verify_input_npy = build_fd + "verification_io/" + model_name + "_input.npy"
verify_expected_output_npy = build_fd + "verification_io/" + model_name + "_output.npy"

verif_steps = [
    "finn_onnx_python",
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


# select build steps (ZCU104/102 folding config is based on separate thresholding nodes)
def select_build_steps(platform):
    if platform in ["ZCU102", "ZCU104"]:
        return [
            step_mobilenet_streamline,
            step_mobilenet_lower_convs,
            step_mobilenet_convert_to_hw_layers_separate_th,
            "step_create_dataflow_partition",
            "step_specialize_layers",
            "step_apply_folding_config",
            "step_minimize_bit_width",
            "step_generate_estimate_reports",
            "step_hw_codegen",
            "step_hw_ipgen",
            "step_set_fifo_depths",
            "step_create_stitched_ip",
            "step_synthesize_bitfile",
            "step_make_driver",
            "step_deployment_package",
        ]
    elif platform in ["U250"]:
        return [
            step_mobilenet_streamline,
            step_mobilenet_lower_convs,
            step_mobilenet_convert_to_hw_layers,
            "step_create_dataflow_partition",
            "step_specialize_layers",
            "step_apply_folding_config",
            "step_minimize_bit_width",
            "step_generate_estimate_reports",
            "step_hw_codegen",
            "step_hw_ipgen",
            "step_set_fifo_depths",
            "step_create_stitched_ip",
            step_mobilenet_slr_floorplan,
            "step_synthesize_bitfile",
            "step_make_driver",
            "step_deployment_package",
        ]


# select target clock frequency
def select_clk_period(platform):
    if platform in ["ZCU102", "ZCU104"]:
        return 5.4
    elif platform in ["U250"]:
        return 3.0


def platform_to_shell(platform):
    if platform in ["U250"]:
        return build_cfg.ShellFlowType.VITIS_ALVEO
    elif platform in ["ZCU102", "ZCU104"]:
        return build_cfg.ShellFlowType.VIVADO_ZYNQ
    else:
        raise Exception("Unknown platform, can't determine ShellFlowType")


def configure_build(board):
    f_file = f"{build_fd}mobilenet-v1/folding_config/mobilenet_folding_config_{board}"
    sl_file = f"{build_fd}mobilenet-v1/specialize_layers_config/mobilenet_specialize_layers_{board}"
    cfg = build_cfg.DataflowBuildConfig(
        generate_outputs=build_outputs,
        output_dir=output_dir,
        steps=select_build_steps(board),
        folding_config_file=f_file + ".json",
        synth_clk_period_ns=select_clk_period(board),
        board=board,
        shell_flow_type=platform_to_shell(board),
        auto_fifo_depths=False,
        specialize_layers_config_file=sl_file + ".json",
        verify_steps=verif_steps,
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
        "ZCU102",
        pytest.param("ZCU104", marks=pytest.mark.xfail(reason="not tested")),
        pytest.param("U250", marks=pytest.mark.xfail(reason="not tested")),
    ],
)
def test_mobilenetv1(board):
    # Check vivado version
    vivado_path = os.environ.get("XILINX_VIVADO")
    match = re.search(r"\b(20\d{2})\.(1|2)\b", vivado_path)
    year, minor = int(match.group(1)), int(match.group(2))
    if board == "AUP-ZU3_8GB" and (year, minor) != (2024, 1):
        pytest.skip("""Vivado version 2024.1 needed for the AUP-ZU3.""")
    elif board != "AUP-ZU3_8GB" and (year, minor) != (2022, 2):
        pytest.skip("""Vivado version 2022.2 needed.""")

    # Run build flow
    cfg = configure_build(board)
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
    assert os.path.isfile(verify_out_dir + "/verify_initial_python_0_SUCCESS.npy")
    assert os.path.isfile(verify_out_dir + "/verify_streamlined_python_0_SUCCESS.npy")
    assert os.path.isfile(verify_out_dir + "/verify_folded_hls_cppsim_0_SUCCESS.npy")
    assert os.path.isfile(verify_out_dir + "/verify_node_by_node_rtlsim_0_SUCCESS.npy")
    assert os.path.isfile(verify_out_dir + "/verify_stitched_ip_rtlsim_0_SUCCESS.npy")
