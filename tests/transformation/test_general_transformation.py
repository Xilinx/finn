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

from finn.core.modelwrapper import ModelWrapper
from finn.transformation.general import GiveUniqueNodeNames

import numpy as np
import onnx
import finn.core.onnx_exec as oxe
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.general import GiveUniqueParameterTensors


def test_give_unique_node_names():
    raw_m = get_data("finn", "data/onnx/mnist-conv/model.onnx")
    model = ModelWrapper(raw_m)
    model = model.transform(GiveUniqueNodeNames())
    assert model.graph.node[0].name == "Reshape_0"
    assert model.graph.node[1].name == "Conv_0"
    assert model.graph.node[11].name == "Add_2"


def test_give_unique_parameter_tensors():

    # Create model
    input_shape = [4, 4]
    in1 = onnx.helper.make_tensor_value_info("in1", onnx.TensorProto.FLOAT, input_shape)
    out1 = onnx.helper.make_tensor_value_info(
        "out1", onnx.TensorProto.FLOAT, input_shape
    )

    graph_nodes = []
    graph_nodes += [
        onnx.helper.make_node("Add", inputs=["in1", "param1"], outputs=["t1"])
    ]

    graph_nodes += [
        onnx.helper.make_node("Sum", inputs=["t1", "param1", "param1"], outputs=["t2"])
    ]

    graph_nodes += [
        onnx.helper.make_node("Sum", inputs=["t2", "param2", "param1"], outputs=["t3"])
    ]

    graph_nodes += [
        onnx.helper.make_node("Add", inputs=["t3", "param1"], outputs=["out1"])
    ]

    onnx_graph = onnx.helper.make_graph(
        nodes=graph_nodes, name="simple_graph", inputs=[in1], outputs=[out1],
    )

    onnx_model = onnx.helper.make_model(onnx_graph, producer_name="simple-model")
    model = ModelWrapper(onnx_model)

    # Set param values
    np.random.seed(0)
    param1 = np.random.rand(*input_shape).astype(np.float32)
    param2 = np.random.rand(*input_shape).astype(np.float32)
    model.set_initializer("param1", param1)
    model.set_initializer("param2", param2)
    model = model.transform(InferShapes())

    # Apply transformation
    new_model = model.transform(GiveUniqueParameterTensors())
    new_model = new_model.transform(InferShapes())

    # Test
    # Breaks the model?
    input_tensor = np.random.rand(*input_shape).astype(np.float32)
    input_dict = {"in1": input_tensor}

    # run original
    expected_context = oxe.execute_onnx(model, input_dict)
    expected_output = expected_context[model.graph.output[0].name]

    # run modified
    produced_context = oxe.execute_onnx(new_model, input_dict)
    produced_output = produced_context[new_model.graph.output[0].name]

    assert np.isclose(
        expected_output, produced_output, atol=1e-8
    ).all(), " GiveUniqueParameterTensors() transform breaks the model"

    # Does the job?
    param_set = set()
    param_cnt = 0
    for n in new_model.graph.node:
        for i in range(1, len(n.input)):
            param_set |= {n.input[i]}
            param_cnt += 1

    assert len(param_set) == param_cnt, " There are still parameters reused"
