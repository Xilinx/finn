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
from qonnx.util.test import (
    download_model,
    get_golden_in_and_output,
    get_model_input_metadata,
)

import finn.builder.build_dataflow as build
import finn.builder.frontend_steps as fe_steps
from finn.builder.build_dataflow_steps import step_tidy_up
from finn.util.basic import make_build_dir

frontend_test_networks = [
    "FINN-TFC_W2A2",
    "FINN-CNV_W2A2",
    "MobileNetv1-w4a4",
    "rn18_w4a4_a2q_plus_12b",
]

frontend_steps = [
    step_tidy_up,
    fe_steps.step_aggregate_scale_bias,
    fe_steps.step_convert_to_thresholds_new,
    fe_steps.step_convert_to_thresholds_old,
    fe_steps.step_lower_convs_to_matmul,
    fe_steps.step_convert_to_channels_last,
    fe_steps.step_convert_to_hw,
]

frontend_step_names_verify = [
    "step_aggregate_scale_bias",
    "step_convert_to_thresholds_new",
    "step_convert_to_thresholds_old",
    "step_lower_convs_to_matmul",
    "step_convert_to_channels_last",
    "step_convert_to_hw",
]


@pytest.mark.parametrize("model_name", frontend_test_networks)
def test_frontend_flow(model_name):
    # download the model to be tested
    filename = download_model(model_name, do_cleanup=True, add_preproc=True)
    assert os.path.isfile(filename), f"Download for model {model_name} failed"
    x, golden_y = get_golden_in_and_output(model_name, preprocesing=True)
    debug = True
    if debug:
        output_dir = os.environ["FINN_BUILD_DIR"] + "/test_frontend_flow_%s" % model_name
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = make_build_dir("test_frontend_flow_%s_" % model_name)
    x_filename = output_dir + "/x.npy"
    golden_y_filename = output_dir + "/golden_y.npy"
    np.save(x_filename, x)
    np.save(golden_y_filename, golden_y)
    current_irange = get_model_input_metadata(model_name, include_preprocessing=True)["range"]
    cfg = build.DataflowBuildConfig(
        output_dir=output_dir,
        verbose=True,
        verify_input_npy=x_filename,
        verify_expected_output_npy=golden_y_filename,
        standalone_thresholds=True,
        steps=frontend_steps,
        input_range_info=[current_irange],
        verify_steps=frontend_step_names_verify,
        synth_clk_period_ns=5,
        generate_outputs=[],
    )
    build.build_dataflow_cfg(filename, cfg)
    # check that intermediate model files are created
    # and that verification is successful
    for step_name in frontend_step_names_verify:
        step_checkpoint = output_dir + "/intermediate_models/" + step_name + ".onnx"
        assert os.path.isfile(step_checkpoint), step_checkpoint + " not found"
        step_ok_fname = output_dir + "/verification_output/verify_" + step_name + "_0_SUCCESS.npy"
        assert os.path.isfile(step_ok_fname), step_ok_fname + " not found"
    if not debug:
        os.unlink(output_dir)
