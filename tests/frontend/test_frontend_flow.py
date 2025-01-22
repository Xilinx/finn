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
from qonnx.transformation.channels_last import ConvertToChannelsLastAndClean
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.streamline import Streamline
from qonnx.util.cleanup import cleanup_model
from qonnx.util.test import download_model, get_model_input_metadata, get_random_input

import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
import finn.transformation.streamline.absorb as absorb
from finn.core.onnx_exec import execute_onnx
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.qonnx.quant_act_to_multithreshold import (
    default_filter_function_generator,
)
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds

frontend_test_networks = [
    "FINN-TFC_W2A2",
    "FINN-CNV_W2A2",
]


@pytest.mark.parametrize("model_name", frontend_test_networks)
def test_frontend_flow(model_name):
    # download the model to be tested
    filename = download_model(model_name, do_cleanup=True, add_preproc=True)
    assert os.path.isfile(filename), f"Download for model {model_name} failed"
    model = ModelWrapper(filename)
    # Step 1: tidy-up
    model = cleanup_model(model, extract_conv_bias=True)
    model.save(f"frontend-step1-tidyup-{model_name}.onnx")
    assert model.check_all_tensor_shapes_specified(), "Tidy-up failed (shape inference)"
    # TODO other checks for tidy-up here
    # generate golden in/out pair
    x = get_random_input(model_name, preprocesing=True)
    input_dict = {model.graph.input[0].name: x}
    golden_output_dict = execute_onnx(model, input_dict)
    golden_y = golden_output_dict[model.graph.output[0].name]
    # Step 2 : range-analysis streamlining
    current_irange = get_model_input_metadata(model_name, include_preprocessing=True)["range"]
    model = model.transform(Streamline(irange=current_irange))
    model = model.transform(InferDataTypes())
    # TODO add checks for streamlining
    streamlined_output_dict = execute_onnx(model, input_dict)
    streamlined_y = streamlined_output_dict[model.graph.output[0].name]
    assert np.isclose(golden_y, streamlined_y, atol=1e-3).all(), "Streamlined model output mismatch"
    model.save(f"frontend-step2-streamline-{model_name}.onnx")
    need_lowering = len(model.get_nodes_by_op_type("Conv")) > 0
    if need_lowering:
        # Step 3: convolution lowering
        model = model.transform(LowerConvsToMatMul())
        lowered_output_dict = execute_onnx(model, input_dict)
        lowered_y = lowered_output_dict[model.graph.output[0].name]
        assert np.isclose(
            golden_y, lowered_y, atol=1e-3
        ).all(), "Conv-lowered model output mismatch"
        model.save(f"frontend-step3-lowerconvs-{model_name}.onnx")
        # TODO checks for convolution lowering
        # Step 4: data layout conversion
        model = model.transform(ConvertToChannelsLastAndClean())
        chanslast_output_dict = execute_onnx(model, input_dict)
        chanslast_y = chanslast_output_dict[model.graph.output[0].name]
        assert np.isclose(
            golden_y, chanslast_y, atol=1e-3
        ).all(), "Channels-last model output mismatch"
        model.save(f"frontend-step4-chanslast-{model_name}.onnx")
    # Step 5: convert Quant nodes to thresholds where it makes sense
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
    model = model.transform(RoundAndClipThresholds())
    thres_output_dict = execute_onnx(model, input_dict)
    thres_y = thres_output_dict[model.graph.output[0].name]
    assert np.isclose(golden_y, thres_y, atol=1e-3).all(), "Thresholds model output mismatch"

    model.save(f"frontend-step5-thresholds-{model_name}.onnx")
    # TODO add checks for threshold conversion
    # Step 6: convert to hardware nodes
    model = model.transform(InferDataTypes())
    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    model = model.transform(to_hw.InferThresholdingLayer())
    model = model.transform(to_hw.InferElementwiseBinaryOperation())
    model = model.transform(to_hw.InferReLUAsElementwiseMax())
    model = model.transform(to_hw.InferQuantAsFloat2Int())
    model = model.transform(to_hw.InferConvInpGen())
    model = model.transform(to_hw.InferStreamingMaxPool())
    model = model.transform(RemoveCNVtoFCFlatten())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model.save(f"frontend-step6-hwlayers-{model_name}.onnx")
