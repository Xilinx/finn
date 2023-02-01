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

import pytest

import numpy as np
import os
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.transformation.streamline.reorder import MoveLinearPastEltwiseAdd

export_onnx_path = "test_linear_past_eltwise.onnx"
np_default_dtype = np.float32

# construct a synthetic graph to test:
# topk insertion, topk conversion to hls, add conversion to hls
# graph should just be a sum


def make_model(shape):
    inp1 = helper.make_tensor_value_info("inp1", TensorProto.FLOAT, shape)
    inp2 = helper.make_tensor_value_info("inp2", TensorProto.FLOAT, shape)
    inp1_add = helper.make_tensor_value_info("inp1_add", TensorProto.FLOAT, shape)
    inp1_add_ct = helper.make_tensor_value_info("inp1_add_ct", TensorProto.FLOAT, [1])
    inp2_add = helper.make_tensor_value_info("inp2_add", TensorProto.FLOAT, shape)
    inp2_add_ct = helper.make_tensor_value_info("inp2_add_ct", TensorProto.FLOAT, [1])
    inp1_mul = helper.make_tensor_value_info("inp1_mul", TensorProto.FLOAT, shape)
    inp1_mul_ct = helper.make_tensor_value_info("inp1_mul_ct", TensorProto.FLOAT, [1])
    inp2_mul = helper.make_tensor_value_info("inp2_mul", TensorProto.FLOAT, shape)
    inp2_mul_ct = helper.make_tensor_value_info("inp2_mul_ct", TensorProto.FLOAT, [1])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, shape)

    add1_node = helper.make_node("Add", [inp1.name, inp1_add_ct.name], [inp1_add.name])
    add2_node = helper.make_node("Add", [inp2.name, inp2_add_ct.name], [inp2_add.name])
    mul1_node = helper.make_node(
        "Mul", [inp1_add.name, inp1_mul_ct.name], [inp1_mul.name]
    )
    mul2_node = helper.make_node(
        "Mul", [inp2_add.name, inp2_mul_ct.name], [inp2_mul.name]
    )
    eltwise_add_node = helper.make_node(
        "Add", [inp1_mul.name, inp2_mul.name], [outp.name]
    )
    graph = helper.make_graph(
        nodes=[add1_node, add2_node, mul1_node, mul2_node, eltwise_add_node],
        name="graph",
        inputs=[inp1, inp2],
        outputs=[outp],
    )

    model = qonnx_make_model(graph, producer_name="add-model")
    model = ModelWrapper(model)

    # set initializers for scalar add/mul nodes
    model.set_initializer(add1_node.input[1], np.array([7.0], dtype=np_default_dtype))
    model.set_initializer(add2_node.input[1], np.array([8.0], dtype=np_default_dtype))
    model.set_initializer(mul1_node.input[1], np.array([3.0], dtype=np_default_dtype))
    model.set_initializer(mul2_node.input[1], np.array([3.0], dtype=np_default_dtype))

    return model


@pytest.mark.streamline
# channels
@pytest.mark.parametrize("ch", [64])
# ifmdim
@pytest.mark.parametrize("ifmdim", [-1, 7])
def test_linear_past_eltwise_add(ch, ifmdim):
    # generate test vectors of correct shape
    if ifmdim == -1:
        input_tensor_shape = (1, ch)
    else:
        input_tensor_shape = (1, ch, ifmdim, ifmdim)

    model = make_model(input_tensor_shape)
    model.save(export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    x1 = np.random.randn(*input_tensor_shape).astype(np.float32)
    x2 = np.random.randn(*input_tensor_shape).astype(np.float32)

    # generate expected value from streamlined net
    input_dict = {model.graph.input[0].name: x1, model.graph.input[1].name: x2}

    output_dict = oxe.execute_onnx(model, input_dict, True)
    produced_sum = output_dict[model.graph.output[0].name]
    expected_sum = 3.0 * ((x1 + x2) + 15.0)
    assert np.isclose(expected_sum, produced_sum, atol=1e-3).all()
    assert len(model.get_nodes_by_op_type("Add")) == 3
    assert len(model.get_nodes_by_op_type("Mul")) == 2

    model = model.transform(MoveLinearPastEltwiseAdd())

    # verify again, to check we didnt break anything
    output_dict = oxe.execute_onnx(model, input_dict, True)
    produced_sum = output_dict[model.graph.output[0].name]
    assert np.isclose(expected_sum, produced_sum, atol=1e-3).all()
    assert len(model.get_nodes_by_op_type("Add")) == 2
    assert len(model.get_nodes_by_op_type("Mul")) == 1

    os.remove(export_onnx_path)


@pytest.mark.streamline
@pytest.mark.parametrize("ch", [64, 1])
# ifmdim
@pytest.mark.parametrize("ifmdim", [-1, 7])
def test_linear_past_eltwise_add_multiple_forks(ch, ifmdim):
    # generate test vectors of correct shape
    if ifmdim == -1:
        input_shape = (1, ch)
    else:
        input_shape = (1, ch, ifmdim, ifmdim)

    top_in = helper.make_tensor_value_info("top_in", TensorProto.FLOAT, input_shape)
    top_out = helper.make_tensor_value_info("top_out", TensorProto.FLOAT, input_shape)

    num_of_params = 6
    value_info = []
    for i in range(num_of_params):
        value_info += [
            helper.make_tensor_value_info("p" + str(i), TensorProto.FLOAT, input_shape)
        ]

    modelproto = qonnx_make_model(
        helper.make_graph(
            name="test",
            inputs=[top_in],
            outputs=[top_out],
            value_info=value_info,
            nodes=[
                helper.make_node("Add", ["top_in", "p0"], ["fork1"]),
                helper.make_node("Mul", ["fork1", "p1"], ["t2"]),
                helper.make_node("Mul", ["fork1", "p2"], ["t3"]),
                helper.make_node("Add", ["t2", "t3"], ["t4"]),
                helper.make_node("Mul", ["t4", "p3"], ["fork2"]),
                helper.make_node("Add", ["fork2", "p4"], ["t5"]),
                helper.make_node("Add", ["fork2", "p5"], ["t6"]),
                helper.make_node("Add", ["t5", "t6"], ["top_out"]),
            ],
        )
    )
    model = ModelWrapper(modelproto)
    model = model.transform(InferShapes())

    np.random.seed(0)
    for i in range(num_of_params):
        model.set_initializer(
            "p" + str(i), np.random.rand(*input_shape).astype(np.float32)
        )

    # need equal mults:
    model.set_initializer("p2", model.get_initializer("p1"))

    # Transform
    new_model = model.transform(MoveLinearPastEltwiseAdd())
    inp_dict = {"top_in": np.random.rand(*input_shape).astype(np.float32)}

    # Test
    assert oxe.compare_execution(model, new_model, inp_dict)
    assert new_model.graph.node[0].op_type == "Add"
    assert new_model.graph.node[1].op_type == "Add"
    assert new_model.graph.node[2].op_type == "Mul"
    assert new_model.graph.node[3].op_type == "Mul"
    assert new_model.graph.node[4].op_type == "Add"
    assert new_model.graph.node[5].op_type == "Add"
    assert len(new_model.graph.node) == 6
