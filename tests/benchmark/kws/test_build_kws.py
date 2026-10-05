############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

import pytest

from benchmark_helpers import (
    bitfile_output_files,
    check_build_outputs,
    get_verify_steps,
)
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
        # non-interactive run: surface the real error instead of dropping into pdb
        enable_build_pdb_debug=False,
        generate_outputs=build_outputs,
        output_dir=output_dir,
        inject_steps_before={"step_qonnx_to_finn": [step_postprocess]},
        folding_config_file=f_file + ".json",
        synth_clk_period_ns=10.0,
        board=board,
        shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
        stitched_ip_gen_dcp=True,
        specialize_layers_config_file=sl_file + ".json",
        verify_steps=get_verify_steps(verif_steps),
        verify_input_npy=verify_input_npy,
        verify_expected_output_npy=verify_expected_output_npy,
    )
    return cfg


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.parametrize("board", ["AUP-ZU3_8GB"])
def test_kws(board, bench_recorder):
    output_dir = make_build_dir("build_kws_")

    # Run build flow
    cfg = configure_build(board, output_dir)
    build.build_dataflow_cfg(model_file, cfg)
    bench_recorder(model_name, board, output_dir)

    # Check that all expected output products are present, reporting every
    # missing artifact at once instead of aborting on the first one. This model
    # builds on AUP-ZU3_8GB via the Vivado/Zynq flow (.bit/.hwh).
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
    ] + bitfile_output_files(build_cfg.ShellFlowType.VIVADO_ZYNQ)
    check_build_outputs(output_dir, build_output_files)
