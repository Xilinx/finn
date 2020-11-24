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

import numpy as np
import pytest
import onnx.helper as oh
from onnx import TensorProto

import finn.core.onnx_exec as ox
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.streamline import (
    MoveScalarAddPastMatMul,
    MoveScalarMulPastMatMul,
)


def test_move_scalar_mul_past_matmul():
    top_in = oh.make_tensor_value_info("top_in", TensorProto.FLOAT, [1, 2])
    mul_param = oh.make_tensor_value_info("mul_param", TensorProto.FLOAT, [1, 1])
    matmul_param = oh.make_tensor_value_info("matmul_param", TensorProto.FLOAT, [2, 2])
    top_out = oh.make_tensor_value_info("top_out", TensorProto.FLOAT, [1, 2])
    modelproto = oh.make_model(
        oh.make_graph(
            name="test",
            inputs=[top_in],
            outputs=[top_out],
            value_info=[mul_param, matmul_param],
            nodes=[
                oh.make_node("Mul", ["top_in", "mul_param"], ["middle"]),
                oh.make_node("MatMul", ["middle", "matmul_param"], ["top_out"]),
            ],
        )
    )
    model = ModelWrapper(modelproto)
    model = model.transform(InferShapes())
    model.set_initializer("mul_param", np.asarray([[3]], dtype=np.float32))
    model.set_initializer(
        "matmul_param", np.asarray([[2, 4], [-1, 1]], dtype=np.float32)
    )
    new_model = model.transform(MoveScalarMulPastMatMul())
    inp_dict = {"top_in": np.asarray([[-1.0, 1.0]], dtype=np.float32)}
    assert ox.compare_execution(model, new_model, inp_dict)
    assert new_model.graph.node[0].op_type == "MatMul"
    assert new_model.graph.node[1].op_type == "Mul"
    assert new_model.graph.node[0].output[0] == new_model.graph.node[1].input[0]


def test_move_scalar_add_past_matmul():
    top_in = oh.make_tensor_value_info("top_in", TensorProto.FLOAT, [1, 2])
    add_param = oh.make_tensor_value_info("add_param", TensorProto.FLOAT, [1, 1])
    matmul_param = oh.make_tensor_value_info("matmul_param", TensorProto.FLOAT, [2, 2])
    top_out = oh.make_tensor_value_info("top_out", TensorProto.FLOAT, [1, 2])
    modelproto = oh.make_model(
        oh.make_graph(
            name="test",
            inputs=[top_in],
            outputs=[top_out],
            value_info=[add_param, matmul_param],
            nodes=[
                oh.make_node("Add", ["top_in", "add_param"], ["middle"]),
                oh.make_node("MatMul", ["middle", "matmul_param"], ["top_out"]),
            ],
        )
    )
    model = ModelWrapper(modelproto)
    model = model.transform(InferShapes())
    model.set_initializer("add_param", np.asarray([[3]], dtype=np.float32))
    model.set_initializer(
        "matmul_param", np.asarray([[2, 4], [-1, 1]], dtype=np.float32)
    )
    new_model = model.transform(MoveScalarAddPastMatMul())
    inp_dict = {"top_in": np.asarray([[-1.0, 1.0]], dtype=np.float32)}
    assert ox.compare_execution(model, new_model, inp_dict)
    assert new_model.graph.node[0].op_type == "MatMul"
    assert new_model.graph.node[1].op_type == "Add"
    assert new_model.graph.node[0].output[0] == new_model.graph.node[1].input[0]


@pytest.mark.parametrize(
    "test_args",
    [("Add", MoveScalarAddPastMatMul()), ("Mul", MoveScalarMulPastMatMul())],
)
def test_move_scalar_past_matmul_only_if_linear(test_args):
    scalar_op = test_args[0]
    transf_fxn = test_args[1]
    input_shape = [1, 2]
    matmul_shape = [2, 2]
    top_in = oh.make_tensor_value_info("top_in", TensorProto.FLOAT, input_shape)
    top_out = oh.make_tensor_value_info("top_out", TensorProto.FLOAT, input_shape)

    p1 = oh.make_tensor_value_info("p1", TensorProto.FLOAT, [1, 1])
    p2 = oh.make_tensor_value_info("p2", TensorProto.FLOAT, matmul_shape)
    p3 = oh.make_tensor_value_info("p3", TensorProto.FLOAT, matmul_shape)
    p4 = oh.make_tensor_value_info("p4", TensorProto.FLOAT, matmul_shape)
    modelproto = oh.make_model(
        oh.make_graph(
            name="test",
            inputs=[top_in],
            outputs=[top_out],
            value_info=[p1, p2, p3, p4],
            nodes=[
                oh.make_node(scalar_op, ["top_in", "p1"], ["t1"]),
                oh.make_node("MatMul", ["t1", "p2"], ["fork"]),
                oh.make_node("MatMul", ["fork", "p3"], ["t3"]),
                oh.make_node(scalar_op, ["t3", "fork"], ["t4"]),
                oh.make_node("MatMul", ["t4", "p4"], ["top_out"]),
            ],
        )
    )
    model = ModelWrapper(modelproto)
    model = model.transform(InferShapes())

    np.random.seed(0)
    model.set_initializer("p1", np.random.rand(1, 1).astype(np.float32))
    model.set_initializer("p2", np.random.rand(*matmul_shape).astype(np.float32))
    model.set_initializer("p3", np.random.rand(*matmul_shape).astype(np.float32))
    model.set_initializer("p4", np.random.rand(*matmul_shape).astype(np.float32))

    # Transform
    new_model = model.transform(transf_fxn)

    # Test
    inp_dict = {"top_in": np.random.rand(*input_shape).astype(np.float32)}
    assert ox.compare_execution(model, new_model, inp_dict)
    assert new_model.graph.node[0].op_type == "MatMul"
    assert new_model.graph.node[1].op_type == scalar_op
    assert new_model.graph.node[2].op_type == "MatMul"
    assert new_model.graph.node[3].op_type == scalar_op
    assert new_model.graph.node[4].op_type == "MatMul"
