############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

import pytest

import os
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.insert_topk import InsertTopK

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.util.basic import make_build_dir

build_fd = "tests/benchmark/"


# Custom step to insert a TopK node so the accelerator returns the predicted
# Top-1 class.
def step_postprocess(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(InsertTopK(k=1))
    return model


# model
model_name = "MLP_W3A3_python_speech_features_pre-processing_QONNX_opset-11"
model_file = build_fd + "models/" + model_name + ".onnx"

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


# Configure build
def configure_build(board, output_dir):
    f_file = f"{build_fd}kws/folding_config/kws_folding_config_{board}"
    sl_file = f"{build_fd}kws/specialize_layers_config/kws_specialize_layers"
    cfg = build_cfg.DataflowBuildConfig(
        generate_outputs=build_outputs,
        output_dir=output_dir,
        inject_steps_before={"step_qonnx_to_finn": [step_postprocess]},
        folding_config_file=f_file + ".json",
        synth_clk_period_ns=10.0,
        board=board,
        shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
        stitched_ip_gen_dcp=True,
        specialize_layers_config_file=sl_file + ".json",
        verify_steps=verif_steps,
        verify_input_npy=verify_input_npy,
        verify_expected_output_npy=verify_expected_output_npy,
    )
    return cfg


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.parametrize("board", ["AUP-ZU3_8GB"])
def test_kws(board):
    output_dir = make_build_dir("build_kws_")

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
