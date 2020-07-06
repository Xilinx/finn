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
import onnx
import onnx.numpy_helper as np_helper
from onnx import TensorProto, helper

from finn.core.modelwrapper import ModelWrapper
from finn.core.datatype import DataType
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_data_layouts import InferDataLayouts
from finn.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from finn.transformation.merge_onnx_models import MergeONNXModels
import finn.core.onnx_exec as oxe


def test_merge_onnx_models():
    # load pre model
    raw_m = get_data("finn", "data/onnx/mnist-conv/model.onnx")
    model1 = ModelWrapper(raw_m)
    # the input for model1 comes from a uint8 vector so we set the finn datatype
    # of the input tensor to DataType.UINT8 to verify that the datatypes are correctly
    # preserved in the transformed model
    model1.set_tensor_datatype(model1.graph.input[0].name, DataType.UINT8)
    model1 = model1.transform(InferShapes())
    model1 = model1.transform(GiveUniqueNodeNames())
    model1 = model1.transform(GiveReadableTensorNames())

    # set up post model
    shape = [1, 10]
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
    a0_value = np.random.uniform(low=0, high=1, size=(1)).astype(np.float32)
    model2.set_initializer("a0", a0_value)
    a1_value = np.random.uniform(low=0.1, high=1, size=(1)).astype(np.float32)
    model2.set_initializer("a1", a1_value)
    # set a dummy sparsity annotation to check if it gets correctly transferred
    # to the merged model
    sparsity = {"dw": {"kernel_shape": 0}}
    model2.set_tensor_sparsity("a1", sparsity)
    model2 = model2.transform(InferShapes())
    model2 = model2.transform(InferDataTypes())
    model2 = model2.transform(InferDataLayouts())
    model2 = model2.transform(GiveUniqueNodeNames())
    model2 = model2.transform(GiveReadableTensorNames())

    # simulate the models before the merging and pass the output of model1 to model2
    # load one of the test vectors
    raw_i = get_data("finn", "data/onnx/mnist-conv/test_data_set_0/input_0.pb")
    inp_values = onnx.load_tensor_from_string(raw_i)
    inp_values = np_helper.to_array(inp_values)
    idict = {model1.graph.input[0].name: inp_values}
    odict = oxe.execute_onnx(model1, idict)
    temp = odict[model1.graph.output[0].name]

    idict = {model2.graph.input[0].name: temp}
    odict = oxe.execute_onnx(model2, idict)
    outp = odict[model2.graph.output[0].name]
    # merge models
    model_transformed = model2.transform(MergeONNXModels(model1))

    idict = {model_transformed.graph.input[0].name: inp_values}
    odict = oxe.execute_onnx(model_transformed, idict)
    outp_transformed = odict[model_transformed.graph.output[0].name]

    model_transformed.save("test.onnx")
    assert (outp == outp_transformed).all()
    assert len(model_transformed.graph.node) == len(model1.graph.node) + len(
        model2.graph.node
    )
    # to test if the value is preserved we set the sparsity annotation of input[1]
    # of the division block to a dummy value, we can now look for the division block
    # and check if the sparsity annotation is still the same
    for n in model_transformed.graph.node:
        if n.op_type == "Div":
            tensor_name = n.input[1]
            set_sparsity = model_transformed.get_tensor_sparsity(tensor_name)
            assert sparsity == set_sparsity

    # check if finn datatype of graph.input[0] is still set to UINT8
    assert model_transformed.get_tensor_datatype("global_in") == DataType.UINT8
