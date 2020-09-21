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

import onnx.helper as oh
from onnx import TensorProto
import brevitas.onnx as bo
from finn.util.test import get_test_model_trained
import finn.analysis.topology as ta
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.infer_shapes import InferShapes


def test_all_tensors_f32():
    top_in = oh.make_tensor_value_info("top_in", TensorProto.FLOAT, [2])
    add_param = oh.make_tensor_value_info("add_param", TensorProto.FLOAT, [2])
    mul_param = oh.make_tensor_value_info("mul_param", TensorProto.FLOAT, [2])
    top_out = oh.make_tensor_value_info("top_out", TensorProto.FLOAT, [2])
    modelproto = oh.make_model(
        oh.make_graph(
            name="test",
            inputs=[top_in],
            outputs=[top_out],
            value_info=[add_param, mul_param],
            nodes=[
                oh.make_node("Add", ["top_in", "add_param"], ["middle"]),
                oh.make_node("Mul", ["middle", "mul_param"], ["top_out"]),
            ],
        )
    )
    model = ModelWrapper(modelproto)
    model = model.transform(InferShapes())
    ret = model.analysis(ta.all_tensors_f32)
    assert ret["all_tensors_f32"] is True

    top_in = oh.make_tensor_value_info("top_in", TensorProto.FLOAT, [2])
    add_param = oh.make_tensor_value_info("add_param", TensorProto.INT8, [2])
    mul_param = oh.make_tensor_value_info("mul_param", TensorProto.FLOAT, [2])
    top_out = oh.make_tensor_value_info("top_out", TensorProto.FLOAT, [2])
    modelproto = oh.make_model(
        oh.make_graph(
            name="test",
            inputs=[top_in],
            outputs=[top_out],
            value_info=[add_param, mul_param],
            nodes=[
                oh.make_node("Add", ["top_in", "add_param"], ["middle"]),
                oh.make_node("Mul", ["middle", "mul_param"], ["top_out"]),
            ],
        )
    )
    model = ModelWrapper(modelproto)
    model = model.transform(InferShapes())
    ret = model.analysis(ta.all_tensors_f32)
    assert ret["all_tensors_f32"] is False


def test_node_inputs_in_expected_order():
    raw_m = get_data("finn", "data/onnx/mnist-conv/model.onnx")
    model = ModelWrapper(raw_m)
    model = model.transform(InferShapes())
    ret = model.analysis(ta.node_inputs_in_expected_order)
    # this model has an (unnecessary) dynamic reshape for its weight tensor
    # and so it fails the check
    assert ret["node_inputs_in_expected_order"] is False


def test_nodes_topologically_sorted():
    # test analysis pass (nodes_topologically_sorted) with different models

    # test with data/onnx/finn-hls-model/tfc_w1_a1_after_conv_to_hls.onnx
    raw_m = get_data(
        "finn", "data/onnx/finn-hls-model/tfc_w1_a1_after_conv_to_hls.onnx"
    )
    model = ModelWrapper(raw_m)
    ret = model.analysis(ta.nodes_topologically_sorted)
    assert ret["nodes_topologically_sorted"] is True

    # remove first node and add it at the end
    graph = model.graph
    first_node = graph.node[0]
    graph.node.remove(first_node)
    graph.node.append(first_node)
    ret = model.analysis(ta.nodes_topologically_sorted)
    assert ret["nodes_topologically_sorted"] is False

    # test with data/onnx/mnist-conv/model.onnx
    raw_m = get_data("finn", "data/onnx/mnist-conv/model.onnx")
    model = ModelWrapper(raw_m)
    ret = model.analysis(ta.nodes_topologically_sorted)
    assert ret["nodes_topologically_sorted"] is True

    # remove first node and add it at the end
    graph = model.graph
    first_node = graph.node[0]
    graph.node.remove(first_node)
    graph.node.append(first_node)
    ret = model.analysis(ta.nodes_topologically_sorted)
    assert ret["nodes_topologically_sorted"] is False

    # test with manually created small network
    Neg_node = oh.make_node("Neg", inputs=["in1"], outputs=["neg1"])
    Round_node = oh.make_node("Round", inputs=["neg1"], outputs=["round1"])

    Ceil_node = oh.make_node("Ceil", inputs=["neg1"], outputs=["ceil1"])
    Add_node = oh.make_node("Add", inputs=["round1", "ceil1"], outputs=["out1"])

    in1 = oh.make_tensor_value_info("in1", TensorProto.FLOAT, [4, 4])
    out1 = oh.make_tensor_value_info("out1", TensorProto.FLOAT, [4, 4])

    graph = oh.make_graph(
        nodes=[Neg_node, Round_node, Ceil_node, Add_node],
        name="simple_graph",
        inputs=[in1],
        outputs=[out1],
        value_info=[
            oh.make_tensor_value_info("neg1", TensorProto.FLOAT, [4, 4]),
            oh.make_tensor_value_info("round1", TensorProto.FLOAT, [4, 4]),
            oh.make_tensor_value_info("ceil1", TensorProto.FLOAT, [4, 4]),
        ],
    )

    onnx_model = oh.make_model(graph, producer_name="simple-model")
    model = ModelWrapper(onnx_model)

    ret = model.analysis(ta.nodes_topologically_sorted)
    assert ret["nodes_topologically_sorted"] is True

    # create same graph but with "wrong" node order
    graph = oh.make_graph(
        nodes=[Round_node, Ceil_node, Neg_node, Add_node],
        name="simple_graph",
        inputs=[in1],
        outputs=[out1],
        value_info=[
            oh.make_tensor_value_info("neg1", TensorProto.FLOAT, [4, 4]),
            oh.make_tensor_value_info("round1", TensorProto.FLOAT, [4, 4]),
            oh.make_tensor_value_info("ceil1", TensorProto.FLOAT, [4, 4]),
        ],
    )

    onnx_model = oh.make_model(graph, producer_name="simple-model")
    model = ModelWrapper(onnx_model)

    ret = model.analysis(ta.nodes_topologically_sorted)
    assert ret["nodes_topologically_sorted"] is False

    # test with data/onnx/finn-hls-model/finn-hls-onnx-model.onnx
    raw_m = get_data("finn", "data/onnx/finn-hls-model/finn-hls-onnx-model.onnx")
    model = ModelWrapper(raw_m)
    ret = model.analysis(ta.nodes_topologically_sorted)
    assert ret["nodes_topologically_sorted"] is True

    # remove first node and add it at the end
    graph = model.graph
    first_node = graph.node[0]
    graph.node.remove(first_node)
    graph.node.append(first_node)
    ret = model.analysis(ta.nodes_topologically_sorted)
    assert ret["nodes_topologically_sorted"] is False

    # test with cnv_w1a1
    build_dir = "/tmp/" + os.environ["FINN_INST_NAME"]
    cnv = get_test_model_trained("CNV", 1, 1)
    bo.export_finn_onnx(
        cnv, (1, 3, 32, 32), build_dir + "/end2end_cnv_w1a1_export.onnx"
    )
    model = ModelWrapper(build_dir + "/end2end_cnv_w1a1_export.onnx")
    ret = model.analysis(ta.nodes_topologically_sorted)
    assert ret["nodes_topologically_sorted"] is True

    # remove first node and add it at the end
    graph = model.graph
    first_node = graph.node[0]
    graph.node.remove(first_node)
    graph.node.append(first_node)
    ret = model.analysis(ta.nodes_topologically_sorted)
    assert ret["nodes_topologically_sorted"] is False
