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
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.streamline import Streamline
from qonnx.util.cleanup import cleanup_model
from qonnx.util.range_analysis import RangeInfo
from qonnx.util.test import download_model, get_random_input

import finn.transformation.streamline.absorb as absorb
from finn.core.onnx_exec import execute_onnx
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.qonnx.quant_act_to_multithreshold import (
    default_filter_function_generator,
)

frontend_test_networks = [
    "FINN-TFC_W2A2",
    "FINN-CNV_W2A2",
]

uint8_to_unitfloat = {
    "range": (np.asarray(0.0, dtype=np.float32), np.asarray(1.0, dtype=np.float32)),
    "int_range": (np.asarray(0.0, dtype=np.float32), np.asarray(255.0, dtype=np.float32)),
    "scale": np.asarray(1.0 / 255.0, dtype=np.float32),
    "bias": np.asarray(0.0, dtype=np.float32),
    "is_initializer": False,
}

model_details_scaledint = {
    "FINN-TFC_W2A2": {
        "scaledint_input_range": RangeInfo(shape=(1, 1, 28, 28), **uint8_to_unitfloat)
    },
    "FINN-CNV_W2A2": {
        "scaledint_input_range": RangeInfo(shape=(1, 3, 32, 32), **uint8_to_unitfloat)
    },
}


@pytest.mark.parametrize("model_name", frontend_test_networks)
def test_frontend_flow(model_name):
    # Step 0: download the model to be tested
    filename = download_model(model_name)
    assert os.path.isfile(filename), f"Download for model {model_name} failed"
    model = ModelWrapper(filename)
    # Step 1: tidy-up
    model = cleanup_model(model, extract_conv_bias=True)
    assert model.check_all_tensor_shapes_specified(), "Tidy-up failed (shape inference)"
    # TODO other checks for tidy-up here
    # generate golden in/out pair
    x = get_random_input(model_name)
    input_dict = {model.graph.input[0].name: x}
    golden_output_dict = execute_onnx(model, input_dict)
    golden_y = golden_output_dict[model.graph.output[0].name]
    # Step-2 : streamlining
    # TODO get input range directly from test_model_details once this is added
    current_irange = model_details_scaledint[model_name]["scaledint_input_range"]
    model = model.transform(Streamline(irange=current_irange))
    # convert Quant node to thresholds where it makes sense
    # TODO to be replaced by the new threshold conversion methodology when it's ready
    model = model.transform(
        ConvertQONNXtoFINN(
            filter_function=default_filter_function_generator(max_multithreshold_bit_width=8)
        )
    )
    model = model.transform(absorb.AbsorbAddIntoMultiThreshold())
    model = model.transform(absorb.FactorOutMulSignMagnitude())
    model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
    model = model.transform(absorb.AbsorbSignBiasIntoMultiThreshold())
    model = model.transform(absorb.Absorb1BitMulIntoMatMul())
    model = model.transform(absorb.Absorb1BitMulIntoConv())
    streamlined_input_dict = {model.graph.input[0].name: x}
    streamlined_output_dict = execute_onnx(model, streamlined_input_dict)
    streamlined_y = streamlined_output_dict[model.graph.output[0].name]
    assert np.isclose(golden_y, streamlined_y).all(), "Streamlined output mismatches golden output"

    # TODO add checks for streamlining
