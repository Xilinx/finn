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
import onnx
from collections import Counter
import brevitas.onnx as bo
import numpy as np
import finn.core.data_layout as DataLayout

from finn.core.modelwrapper import ModelWrapper
from finn.util.test import get_test_model_trained

export_onnx_path = "test_output_lfc.onnx"


def test_modelwrapper():
    lfc = get_test_model_trained("LFC", 1, 1)
    bo.export_finn_onnx(lfc, (1, 1, 28, 28), export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    assert model.check_all_tensor_shapes_specified() is False
    inp_name = model.graph.input[0].name
    inp_shape = model.get_tensor_shape(inp_name)
    assert inp_shape == [1, 1, 28, 28]
    # find first matmul node
    l0_mat_tensor_name = ""
    l0_inp_tensor_name = ""
    for node in model.graph.node:
        if node.op_type == "MatMul":
            l0_inp_tensor_name = node.input[0]
            l0_mat_tensor_name = node.input[1]
            break
    assert l0_mat_tensor_name != ""
    l0_weights = model.get_initializer(l0_mat_tensor_name)
    assert l0_weights.shape == (784, 1024)
    l0_weights_hist = Counter(l0_weights.flatten())
    assert (l0_weights_hist[1.0] + l0_weights_hist[-1.0]) == 784 * 1024
    l0_weights_rand = np.random.randn(784, 1024)
    model.set_initializer(l0_mat_tensor_name, l0_weights_rand)
    assert (model.get_initializer(l0_mat_tensor_name) == l0_weights_rand).all()
    assert l0_inp_tensor_name != ""
    inp_cons = model.find_consumer(l0_inp_tensor_name)
    assert inp_cons.op_type == "MatMul"
    out_prod = model.find_producer(l0_inp_tensor_name)
    assert out_prod.op_type == "MultiThreshold"
    inp_layout = model.get_tensor_layout(inp_name)
    assert inp_layout is None
    inp_layout = DataLayout.NCHW
    model.set_tensor_layout(inp_name, inp_layout)
    assert model.get_tensor_layout(inp_name) == inp_layout
    inp_sparsity = model.get_tensor_sparsity(inp_name)
    assert inp_sparsity is None
    inp_sparsity = {"dw": {"kernel_shape": 3}}
    model.set_tensor_sparsity(inp_name, inp_sparsity)
    assert model.get_tensor_sparsity(inp_name) == inp_sparsity
    os.remove(export_onnx_path)


def test_modelwrapper_graph_order():
    # create small network with properties to be tested
    Neg_node = onnx.helper.make_node("Neg", inputs=["in1"], outputs=["neg1"])
    Round_node = onnx.helper.make_node("Round", inputs=["neg1"], outputs=["round1"])

    Ceil_node = onnx.helper.make_node("Ceil", inputs=["neg1"], outputs=["ceil1"])
    Add_node = onnx.helper.make_node(
        "Add", inputs=["round1", "ceil1"], outputs=["out1"]
    )

    in1 = onnx.helper.make_tensor_value_info("in1", onnx.TensorProto.FLOAT, [4, 4])
    out1 = onnx.helper.make_tensor_value_info("out1", onnx.TensorProto.FLOAT, [4, 4])

    graph = onnx.helper.make_graph(
        nodes=[Neg_node, Round_node, Ceil_node, Add_node],
        name="simple_graph",
        inputs=[in1],
        outputs=[out1],
        value_info=[
            onnx.helper.make_tensor_value_info("neg1", onnx.TensorProto.FLOAT, [4, 4]),
            onnx.helper.make_tensor_value_info(
                "round1", onnx.TensorProto.FLOAT, [4, 4]
            ),
            onnx.helper.make_tensor_value_info("ceil1", onnx.TensorProto.FLOAT, [4, 4]),
        ],
    )

    onnx_model = onnx.helper.make_model(graph, producer_name="simple-model")
    model = ModelWrapper(onnx_model)

    # test graph order functions
    assert model.find_consumers("in1") == [Neg_node]
    assert model.find_consumers("neg1") == [Round_node, Ceil_node]
    assert model.find_consumers("round1") == [Add_node]
    assert model.find_consumers("ceil1") == [Add_node]
    assert model.find_consumers("out1") is None

    assert model.find_direct_successors(Neg_node) == [Round_node, Ceil_node]
    assert model.find_direct_successors(Round_node) == [Add_node]
    assert model.find_direct_successors(Ceil_node) == [Add_node]
    assert model.find_direct_successors(Add_node) is None

    assert model.find_direct_predecessors(Neg_node) is None
    assert model.find_direct_predecessors(Round_node) == [Neg_node]
    assert model.find_direct_predecessors(Ceil_node) == [Neg_node]
    assert model.find_direct_predecessors(Add_node) == [Round_node, Ceil_node]

    assert model.get_node_index(Neg_node) == 0
    assert model.get_node_index(Round_node) == 1
    assert model.get_node_index(Ceil_node) == 2
    assert model.get_node_index(Add_node) == 3


def test_modelwrapper_detect_forks_n_joins():
    # create small network with properties to be tested
    Neg_node = onnx.helper.make_node("Neg", inputs=["in1"], outputs=["neg1"])
    Round_node = onnx.helper.make_node("Round", inputs=["neg1"], outputs=["round1"])

    Ceil_node = onnx.helper.make_node("Ceil", inputs=["neg1"], outputs=["ceil1"])
    Add_node = onnx.helper.make_node(
        "Add", inputs=["round1", "ceil1"], outputs=["out1"]
    )

    in1 = onnx.helper.make_tensor_value_info("in1", onnx.TensorProto.FLOAT, [4, 4])
    out1 = onnx.helper.make_tensor_value_info("out1", onnx.TensorProto.FLOAT, [4, 4])

    graph = onnx.helper.make_graph(
        nodes=[Neg_node, Round_node, Ceil_node, Add_node],
        name="simple_graph",
        inputs=[in1],
        outputs=[out1],
        value_info=[
            onnx.helper.make_tensor_value_info("neg1", onnx.TensorProto.FLOAT, [4, 4]),
            onnx.helper.make_tensor_value_info(
                "round1", onnx.TensorProto.FLOAT, [4, 4]
            ),
            onnx.helper.make_tensor_value_info("ceil1", onnx.TensorProto.FLOAT, [4, 4]),
        ],
    )

    onnx_model = onnx.helper.make_model(graph, producer_name="simple-model")
    model = ModelWrapper(onnx_model)

    # test
    assert model.is_fork_node(Neg_node)
    assert not model.is_fork_node(Round_node)
    assert not model.is_fork_node(Ceil_node)
    assert not model.is_fork_node(Add_node)

    assert not model.is_join_node(Neg_node)
    assert not model.is_join_node(Round_node)
    assert not model.is_join_node(Ceil_node)
    assert model.is_join_node(Add_node)
