# Copyright (c) 2020, Xilinx
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

import os
from pkgutil import get_data

import brevitas.onnx as bo
import numpy as np
import onnx
import onnx.numpy_helper as nph
import torch

import finn.core.onnx_exec as oxe
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
import finn.transformation.streamline.absorb as absorb
from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.registry import getCustomOp
from finn.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.util.test import get_test_model_trained


export_onnx_path = "test_output_tfc.onnx"
export_onnx_path_cnv = "test_output_cnv.onnx"


def test_convert_to_hls_layers_tfc_w1a1():
    tfc = get_test_model_trained("TFC", 1, 1)
    bo.export_finn_onnx(tfc, (1, 1, 28, 28), export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(Streamline())
    model = model.transform(ConvertBipolarMatMulToXnorPopcount())
    model = model.transform(absorb.AbsorbAddIntoMultiThreshold())
    model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
    model = model.transform(RoundAndClipThresholds())
    model = model.transform(to_hls.InferBinaryStreamingFCLayer())
    fc0 = model.graph.node[2]
    assert fc0.op_type == "StreamingFCLayer_Batch"
    assert model.get_tensor_shape(fc0.input[0]) == [1, 784]
    assert model.get_tensor_shape(fc0.input[1]) == [784, 64]
    assert model.get_tensor_shape(fc0.input[2]) == [64, 1]
    fc1 = model.graph.node[3]
    assert fc1.op_type == "StreamingFCLayer_Batch"
    assert model.get_tensor_shape(fc1.input[0]) == [1, 64]
    assert model.get_tensor_shape(fc1.input[1]) == [64, 64]
    assert model.get_tensor_shape(fc1.input[2]) == [64, 1]
    fc2 = model.graph.node[4]
    assert fc2.op_type == "StreamingFCLayer_Batch"
    assert model.get_tensor_shape(fc2.input[0]) == [1, 64]
    assert model.get_tensor_shape(fc2.input[1]) == [64, 64]
    assert model.get_tensor_shape(fc2.input[2]) == [64, 1]
    fc3 = model.graph.node[5]
    assert fc3.op_type == "StreamingFCLayer_Batch"
    assert model.get_tensor_shape(fc3.input[0]) == [1, 64]
    assert model.get_tensor_shape(fc3.input[1]) == [64, 10]
    os.remove(export_onnx_path)

    fc0w = getCustomOp(fc0)
    fc0w.set_nodeattr("SIMD", 784)
    fc0w.set_nodeattr("PE", 16)

    fc1w = getCustomOp(fc1)
    fc1w.set_nodeattr("SIMD", 16)
    fc1w.set_nodeattr("PE", 16)

    fc2w = getCustomOp(fc2)
    fc2w.set_nodeattr("SIMD", 16)
    fc2w.set_nodeattr("PE", 16)

    fc3w = getCustomOp(fc3)
    fc3w.set_nodeattr("SIMD", 16)
    fc3w.set_nodeattr("PE", 10)

    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())
    model = model.transform(SetExecMode("npysim"))

    raw_i = get_data("finn", "data/onnx/mnist-conv/test_data_set_0/input_0.pb")
    input_tensor = onnx.load_tensor_from_string(raw_i)
    # run using FINN-based execution
    input_dict = {"global_in": nph.to_array(input_tensor)}
    output_dict = oxe.execute_onnx(model, input_dict)
    produced = output_dict[list(output_dict.keys())[0]]
    # run using PyTorch/Brevitas
    input_tensor = torch.from_numpy(nph.to_array(input_tensor)).float()
    assert input_tensor.shape == (1, 1, 28, 28)
    # do forward pass in PyTorch/Brevitas
    expected = tfc.forward(input_tensor).detach().numpy()
    assert np.isclose(produced, expected, atol=1e-3).all()


def test_convert_to_hls_layers_tfc_w1a2():
    tfc = get_test_model_trained("TFC", 1, 2)
    bo.export_finn_onnx(tfc, (1, 1, 28, 28), export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(Streamline())
    from finn.transformation.fpgadataflow.convert_to_hls_layers import (
        InferQuantizedStreamingFCLayer,
    )

    model = model.transform(InferQuantizedStreamingFCLayer())

    fc0 = model.graph.node[2]
    assert fc0.op_type == "StreamingFCLayer_Batch"
    assert model.get_tensor_shape(fc0.input[0]) == [1, 784]
    assert model.get_tensor_shape(fc0.input[1]) == [784, 64]
    assert model.get_tensor_shape(fc0.input[2]) == [64, 2]
    fc1 = model.graph.node[3]
    assert fc1.op_type == "StreamingFCLayer_Batch"
    assert model.get_tensor_shape(fc1.input[0]) == [1, 64]
    assert model.get_tensor_shape(fc1.input[1]) == [64, 64]
    assert model.get_tensor_shape(fc1.input[2]) == [64, 2]
    fc2 = model.graph.node[4]
    assert fc2.op_type == "StreamingFCLayer_Batch"
    assert model.get_tensor_shape(fc2.input[0]) == [1, 64]
    assert model.get_tensor_shape(fc2.input[1]) == [64, 64]
    assert model.get_tensor_shape(fc2.input[2]) == [64, 2]
    fc3 = model.graph.node[5]
    assert fc3.op_type == "StreamingFCLayer_Batch"
    assert model.get_tensor_shape(fc3.input[0]) == [1, 64]
    assert model.get_tensor_shape(fc3.input[1]) == [64, 10]
    fc0w = getCustomOp(fc0)
    fc0w.set_nodeattr("SIMD", 784)
    fc0w.set_nodeattr("PE", 16)
    fc1w = getCustomOp(fc1)
    fc1w.set_nodeattr("SIMD", 16)
    fc1w.set_nodeattr("PE", 16)
    fc2w = getCustomOp(fc2)
    fc2w.set_nodeattr("SIMD", 16)
    fc2w.set_nodeattr("PE", 16)
    fc3w = getCustomOp(fc3)
    fc3w.set_nodeattr("SIMD", 16)
    fc3w.set_nodeattr("PE", 10)
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())
    model = model.transform(SetExecMode("npysim"))
    raw_i = get_data("finn", "data/onnx/mnist-conv/test_data_set_0/input_0.pb")
    input_tensor = onnx.load_tensor_from_string(raw_i)
    # run using FINN-based execution
    input_dict = {"global_in": nph.to_array(input_tensor)}
    output_dict = oxe.execute_onnx(model, input_dict, True)
    produced = output_dict[model.graph.output[0].name]
    model = ModelWrapper(export_onnx_path)
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(Streamline())
    golden_output_dict = oxe.execute_onnx(model, input_dict, True)
    expected = golden_output_dict[model.graph.output[0].name]
    assert np.isclose(produced, expected, atol=1e-3).all()
    os.remove(export_onnx_path)
