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
import os
from onnx import helper as oh
from qonnx.core.datatype import DataType
from qonnx.transformation.insert_topk import InsertTopK

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_config import default_build_dataflow_steps
from finn.util.basic import make_build_dir

build_flow_folder = "tests/benchmark/"
output_dir = make_build_dir("build_gtsrb_")


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


custom_build_steps = (
    [custom_step_add_preproc] + [custom_step_add_postproc] + default_build_dataflow_steps
)

# model
model_name = "cnv_1w1a_gtsrb"
model_file = build_flow_folder + "models/" + model_name + ".onnx"

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


def configure_build(board):
    cfg = build_cfg.DataflowBuildConfig(
        output_dir=output_dir,
        synth_clk_period_ns=10.0,
        board=board,
        steps=custom_build_steps,
        verify_steps=verif_steps,
        verify_input_npy=verify_input_npy,
        verify_expected_output_npy=verify_expected_output_npy,
        folding_config_file=f"{build_flow_folder}gtsrb/gtsrb_folding_config_{board}.json",
        shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
        generate_outputs=build_outputs,
        specialize_layers_config_file=f"{build_flow_folder}gtsrb/gtsrb_specialize_layers.json",
    )
    return cfg


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.parametrize("board", ["Pynq-Z1", "AUP-ZU3_8GB"])
def test_gtsrb(board):
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
