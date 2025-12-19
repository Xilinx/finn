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
from qonnx.core.modelwrapper import ModelWrapper
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.util.basic import make_build_dir
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
import finn.transformation.streamline.absorb as absorb
from qonnx.transformation.change_3d_tensors_to_4d import Change3DTo4DTensors
from qonnx.transformation.general import GiveUniqueNodeNames

# custom steps for vgg10-radioml
def step_pre_streamline(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(Change3DTo4DTensors())
    model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
    return model


def step_convert_final_layers(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(to_hw.InferChannelwiseLinearLayer())
    model = model.transform(to_hw.InferLabelSelectLayer())
    model = model.transform(GiveUniqueNodeNames())
    return model


build_flow_folder = "tests/benchmark/"
output_dir = make_build_dir("build_vgg10-radioml_")

# model
model_name = "radioml_w4a4_small_tidy"
model_file = build_flow_folder + "models/%s.onnx" % model_name


# verification parameters
verify_input_npy = build_flow_folder + "verification_io/" + model_name + "_input.npy"
verify_expected_output_npy = build_flow_folder + "verification_io/" + model_name + "_output.npy"

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

build_steps = [
        "step_tidy_up",
        step_pre_streamline,
        "step_streamline",
        "step_convert_to_hw",
        step_convert_final_layers,
        "step_create_dataflow_partition",
        "step_specialize_layers",
        "step_target_fps_parallelization",
        "step_apply_folding_config",
        "step_minimize_bit_width",
        "step_generate_estimate_reports",
        "step_hw_codegen",
        "step_hw_ipgen",
        "step_set_fifo_depths",
        "step_create_stitched_ip",
        "step_measure_rtlsim_performance",
        "step_out_of_context_synthesis",
        "step_synthesize_bitfile",
        "step_deployment_package",
    ]


def configure_build(board):
    cfg = build_cfg.DataflowBuildConfig(
        generate_outputs=build_outputs,
        output_dir=output_dir,
        steps=build_steps,
        folding_config_file = f"{build_flow_folder}vgg10-radioml/folding_config/vgg10radioml_folding_config.json",
        synth_clk_period_ns=4.0,
        board=board,
        shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
        split_large_fifos=True,
        standalone_thresholds=True,
        specialize_layers_config_file=f"{build_flow_folder}vgg10-radioml/specialize_layers_config/vgg10radioml_specialize_layers.json",
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
