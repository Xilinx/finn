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
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg

from finn.util.basic import make_build_dir, alveo_default_platform


build_flow_folder = "tests/benchmark/"
output_dir = make_build_dir("build_bnn-pynq_")

# model
def get_model_file(model):
    return build_flow_folder + "models/" + model + ".onnx"


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

# verification parameters
def get_verify_input_npy(model):
    if "tfc-w" in model:
        verify_input_npy = build_flow_folder + "verification_io/tfc_mnist_input.npy"
    else:
        verify_input_npy = build_flow_folder + "verification_io/cnv_cifar10_input.npy"
    return verify_input_npy

def get_verify_output_npy(model):
    if "tfc-w" in model:
        verify_expected_output_npy = build_flow_folder + "verification_io/tfc_mnist_output.npy"
    else:
        verify_expected_output_npy = build_flow_folder + "verification_io/cnv_cifar10_output.npy"
    return verify_expected_output_npy

def platform_to_shell(platform):
    if platform in ["U250"]:
        return build_cfg.ShellFlowType.VITIS_ALVEO
    elif platform in ["AUP-ZU3_8GB", "Pynq-Z1", "Ultra96", "ZCU104"]:
        return build_cfg.ShellFlowType.VIVADO_ZYNQ
    else:
        raise Exception("Unknown platform, can't determine ShellFlowType")


def configure_build(board, model):
    shell_flow_type = platform_to_shell(board)
    if shell_flow_type == build_cfg.ShellFlowType.VITIS_ALVEO:
        vitis_platform = alveo_default_platform[board]
    else:
        vitis_platform = None
    if board in ["AUP-ZU3_8GB"]:
        cfg = build_cfg.DataflowBuildConfig(
            generate_outputs=build_outputs,
            output_dir=output_dir,
            folding_config_file = f"{build_flow_folder}bnn-pynq/folding_config/{model}_folding_config_{board}.json",
            synth_clk_period_ns=10.0,
            board=board,
            shell_flow_type=platform_to_shell(board),
            vitis_platform=vitis_platform,
            stitched_ip_gen_dcp=True,
            specialize_layers_config_file=f"{build_flow_folder}bnn-pynq/specialize_layers_config/{model}_specialize_layers_{board}.json",
            verify_steps=verif_steps,
            verify_input_npy=get_verify_input_npy(model),
            verify_expected_output_npy=get_verify_output_npy(model),
        )
    else:
        cfg = build_cfg.DataflowBuildConfig(
            generate_outputs=build_outputs,
            output_dir=output_dir,
            folding_config_file = f"{build_flow_folder}bnn-pynq/folding_config/{model}_folding_config.json",
            synth_clk_period_ns=10.0,
            board=board,
            shell_flow_type=platform_to_shell(board),
            vitis_platform=vitis_platform,
            stitched_ip_gen_dcp=True,
            specialize_layers_config_file=f"{build_flow_folder}bnn-pynq/specialize_layers_config/{model}_specialize_layers.json",
            verify_steps=verif_steps,
            verify_input_npy=get_verify_input_npy(model),
            verify_expected_output_npy=get_verify_output_npy(model),
        )
    return cfg



@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.parametrize("board", ["AUP-ZU3_8GB", "Pynq-Z1", pytest.param("Ultra96", marks=pytest.mark.xfail(reason="not tested")), pytest.param("ZCU104", marks=pytest.mark.xfail(reason="not tested")), pytest.param("U250", marks=pytest.mark.xfail(reason="not tested"))])
@pytest.mark.parametrize("model", ["tfc-w1a1", "tfc-w1a2", "tfc-w2a2", "cnv-w1a1", "cnv-w1a2", "cnv-w2a2"])
def test_bnnpynq(board, model):
    # Check vivado version
    vivado_path = os.environ.get("XILINX_VIVADO")
    match = re.search(r"\b(20\d{2})\.(1|2)\b", vivado_path)
    year, minor = int(match.group(1)), int(match.group(2))
    if board == "AUP-ZU3_8GB" and (year, minor) != (2024, 1):
        pytest.skip(
            """Vivado version 2024.1 needed for the AUP-ZU3."""
        )
    elif board != "AUP-ZU3_8GB" and (year, minor) != (2022, 2):
        pytest.skip(
            """Vivado version 2022.2 needed."""
        )

    # Run build flow
    cfg = configure_build(board, model)
    model_file = get_model_file(model)
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
