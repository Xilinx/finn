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

from pkgutil import get_data

import numpy as np
from onnx import TensorProto, helper

from finn.core.modelwrapper import ModelWrapper
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from finn.transformation.merge_onnx_models import MergeONNXModels
import finn.core.onnx_exec as oxe


def test_merge_onnx_models():
    # load first model
    raw_m = get_data("finn", "data/onnx/mnist-conv/model.onnx")
    model1 = ModelWrapper(raw_m)
    model1 = model1.transform(InferShapes())
    model1 = model1.transform(GiveUniqueNodeNames())
    model1 = model1.transform(GiveReadableTensorNames())

    # set up second model that should be inserted before the first model
    shape = [1, 1, 28, 28]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)
    a0 = helper.make_tensor_value_info("a0", TensorProto.FLOAT, [])
    a1 = helper.make_tensor_value_info("a1", TensorProto.FLOAT, [])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, shape)

    mul_node = helper.make_node("Mul", ["inp", "a0"], ["mul_out"])
    div_node = helper.make_node("Div", ["mul_out", "a1"], ["outp"])

    graph = helper.make_graph(
        nodes=[mul_node, div_node],
        name="model2-graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[a0, a1],
    )

    model2 = helper.make_model(graph, producer_name="model2")
    model2 = ModelWrapper(model2)
    # initialize model2
    a0_value = np.random.uniform(low=0.1, high=0.99, size=(1)).astype(np.float32)
    model2.set_initializer("a0", a0_value)
    a1_value = 1.0 / a0_value
    model2.set_initializer("a1", a1_value)
    model2 = model2.transform(InferShapes())
    model2 = model2.transform(GiveUniqueNodeNames())
    model2 = model2.transform(GiveReadableTensorNames())

    # simulate the models before the merging and pass the output of model2 to model1
    inp_values = np.random.uniform(low=-1, high=1, size=tuple(shape)).astype(np.float32)
    idict = {model2.graph.input[0].name: inp_values}
    odict = oxe.execute_onnx(model2, idict)
    temp = odict[model2.graph.output[0].name]

    idict = {model1.graph.input[0].name: temp}
    odict = oxe.execute_onnx(model1, idict)
    outp = odict[model1.graph.output[0].name]
    # merge models
    model_transformed = model1.transform(MergeONNXModels(model2))

    idict = {model_transformed.graph.input[0].name: inp_values}
    odict = oxe.execute_onnx(model_transformed, idict)
    outp_transformed = odict[model_transformed.graph.output[0].name]

    assert (outp == outp_transformed).all()
    assert len(model_transformed.graph.node) == len(model1.graph.node) + len(
        model2.graph.node
    )
