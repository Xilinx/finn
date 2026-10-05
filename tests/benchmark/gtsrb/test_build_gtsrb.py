# Copyright (C) 2024, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import pytest

import numpy as np
from benchmark_helpers import (
    bitfile_output_files,
    check_build_outputs,
    get_verify_steps,
)
from onnx import helper as oh
from qonnx.core.datatype import DataType
from qonnx.transformation.insert_topk import InsertTopK

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import make_build_dir

build_fd = "tests/benchmark/"


def custom_step_add_preproc(model, cfg):
    # GTSRB data with raw uint8 pixels is divided by 255 prior to training
    # reflect this in the inference graph so we can perform inference directly
    # on raw uint8 data
    in_name = model.graph.input[0].name
    new_in_name = model.make_new_valueinfo_name()
    new_param_name = model.make_new_valueinfo_name()
    div_param = np.asarray(255.0, dtype=np.float32)
    new_div = oh.make_node(
        "Div",
        [in_name, new_param_name],
        [new_in_name],
        name="PreprocDiv",
    )
    model.set_initializer(new_param_name, div_param)
    model.graph.node.insert(0, new_div)
    model.graph.node[1].input[0] = new_in_name
    # set input dtype to uint8
    model.set_tensor_datatype(in_name, DataType["UINT8"])
    return model


# Insert TopK node to get predicted Top-1 class
def custom_step_add_postproc(model, cfg):
    model = model.transform(InsertTopK(k=1))
    return model


# model
model_name = "cnv_1w1a_gtsrb"
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


def configure_build(board, output_dir):
    f_file = f"{build_fd}gtsrb/folding_config/gtsrb_folding_config_{board}"
    sl_file = f"{build_fd}gtsrb/specialize_layers_config/gtsrb_specialize_layers"
    cfg = build_cfg.DataflowBuildConfig(
        # non-interactive run: surface the real error instead of dropping into pdb
        enable_build_pdb_debug=False,
        output_dir=output_dir,
        synth_clk_period_ns=10.0,
        board=board,
        inject_steps_before={
            "step_qonnx_to_finn": [custom_step_add_preproc, custom_step_add_postproc]
        },
        verify_steps=get_verify_steps(verif_steps),
        verify_input_npy=verify_input_npy,
        verify_expected_output_npy=verify_expected_output_npy,
        folding_config_file=f_file + ".json",
        shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
        generate_outputs=build_outputs,
        specialize_layers_config_file=sl_file + ".json",
    )
    return cfg


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.parametrize("board", ["AUP-ZU3_8GB"])
def test_gtsrb(board, bench_recorder):
    output_dir = make_build_dir("build_gtsrb_")

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
