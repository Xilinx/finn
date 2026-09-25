# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import os

# custom steps for vgg10-radioml
from custom_steps_vgg10 import step_pre_streamline

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import make_build_dir

build_flow_folder = "tests/benchmark/"

# model
model_name = "radioml_w4a4_small_tidy"
model_file = build_flow_folder + "models/%s.onnx" % model_name


# verification parameters
verify_input_npy = build_flow_folder + "verification_io/" + model_name + "_input.npy"
verify_expected_output_npy = build_flow_folder + "verification_io/" + model_name + "_output.npy"

verif_steps = [
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


def configure_build(board, output_dir):
    cfg = build_cfg.DataflowBuildConfig(
        generate_outputs=build_outputs,
        output_dir=output_dir,
        steps=build_steps,
        folding_config_file=(
            f"{build_flow_folder}vgg10-radioml/" f"folding_config/vgg10radioml_folding_config.json"
        ),
        synth_clk_period_ns=4.0,
        board=board,
        shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
        standalone_thresholds=True,
        specialize_layers_config_file=(
            f"{build_flow_folder}vgg10-radioml/"
            f"specialize_layers_config/vgg10radioml_specialize_layers.json"
        ),
        verify_steps=verif_steps,
        verify_input_npy=verify_input_npy,
        verify_expected_output_npy=verify_expected_output_npy,
    )
    return cfg


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.parametrize("board", ["AUP-ZU3_8GB", "ZCU104"])
def test_vgg10radioml(board):
    output_dir = make_build_dir("build_vgg10-radioml_")

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
    assert os.path.isfile(verify_out_dir + "/verify_initial_python_0_SUCCESS.npy")
    assert os.path.isfile(verify_out_dir + "/verify_streamlined_python_0_SUCCESS.npy")
    assert os.path.isfile(verify_out_dir + "/verify_folded_hls_cppsim_0_SUCCESS.npy")
    assert os.path.isfile(verify_out_dir + "/verify_node_by_node_rtlsim_0_SUCCESS.npy")
    assert os.path.isfile(verify_out_dir + "/verify_stitched_ip_rtlsim_0_SUCCESS.npy")
