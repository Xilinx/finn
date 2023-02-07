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
import onnx.helper as oh
from onnx import TensorProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model

import finn.core.onnx_exec as ox
from finn.transformation.streamline import CollapseRepeatedAdd, CollapseRepeatedMul


@pytest.mark.streamline
def test_collapse_repeated_op():
    top_in = oh.make_tensor_value_info("top_in", TensorProto.FLOAT, [2])
    add_param_0 = oh.make_tensor_value_info("add_param_0", TensorProto.FLOAT, [2])
    mul_param_0 = oh.make_tensor_value_info("mul_param_0", TensorProto.FLOAT, [2])
    add_param_1 = oh.make_tensor_value_info("add_param_1", TensorProto.FLOAT, [2])
    mul_param_1 = oh.make_tensor_value_info("mul_param_1", TensorProto.FLOAT, [2])
    top_out = oh.make_tensor_value_info("top_out", TensorProto.FLOAT, [2])
    modelproto = qonnx_make_model(
        oh.make_graph(
            name="test",
            inputs=[top_in],
            outputs=[top_out],
            value_info=[add_param_0, mul_param_0, add_param_1, mul_param_1],
            nodes=[
                oh.make_node("Add", ["top_in", "add_param_0"], ["middle_0"]),
                oh.make_node("Add", ["middle_0", "add_param_1"], ["middle_1"]),
                oh.make_node("Mul", ["middle_1", "mul_param_0"], ["middle_2"]),
                oh.make_node("Mul", ["middle_2", "mul_param_1"], ["top_out"]),
            ],
        )
    )
    model = ModelWrapper(modelproto)
    model = model.transform(InferShapes())
    model.set_initializer("add_param_0", np.asarray([1, 3], dtype=np.float32))
    model.set_initializer("add_param_1", np.asarray([-1, 3], dtype=np.float32))
    model.set_initializer("mul_param_0", np.asarray([2, 4], dtype=np.float32))
    model.set_initializer("mul_param_1", np.asarray([2, -4], dtype=np.float32))
    new_model = model.transform(CollapseRepeatedAdd())
    new_model = new_model.transform(CollapseRepeatedMul())
    inp_dict = {"top_in": np.asarray([-1.0, 1.0], dtype=np.float32)}
    assert ox.compare_execution(model, new_model, inp_dict)
    assert len(new_model.graph.node) == 2
    assert new_model.graph.node[0].op_type == "Add"
    assert new_model.graph.node[1].op_type == "Mul"


@pytest.mark.streamline
@pytest.mark.parametrize(
    "test_args",
    [("Add", CollapseRepeatedAdd()), ("Mul", CollapseRepeatedMul())],
)
def test_collapse_repeated_only_if_linear(test_args):
    scalar_op = test_args[0]
    transf_fxn = test_args[1]

    input_shape = [4, 4]
    output_shape = input_shape

    top_in = oh.make_tensor_value_info("top_in", TensorProto.FLOAT, input_shape)
    top_out = oh.make_tensor_value_info("top_out", TensorProto.FLOAT, output_shape)

    value_info = [oh.make_tensor_value_info("p1", TensorProto.FLOAT, [1])]
    value_info += [oh.make_tensor_value_info("p2", TensorProto.FLOAT, [1])]
    value_info += [oh.make_tensor_value_info("p3", TensorProto.FLOAT, [1])]
    value_info += [oh.make_tensor_value_info("p4", TensorProto.FLOAT, [1])]
    value_info += [oh.make_tensor_value_info("p5", TensorProto.FLOAT, [1])]

    modelproto = qonnx_make_model(
        oh.make_graph(
            name="test",
            inputs=[top_in],
            outputs=[top_out],
            value_info=value_info,
            nodes=[
                oh.make_node(scalar_op, ["top_in", "p2"], ["t1"]),
                oh.make_node(scalar_op, ["t1", "p1"], ["t2"]),
                oh.make_node(scalar_op, ["t2", "p3"], ["t3"]),
                oh.make_node(scalar_op, ["t2", "p4"], ["t4"]),
                oh.make_node(scalar_op, ["t3", "t4"], ["t5"]),
                oh.make_node(scalar_op, ["t5", "p5"], ["top_out"]),
            ],
        )
    )
    model = ModelWrapper(modelproto)
    model = model.transform(InferShapes())

    np.random.seed(0)
    model.set_initializer("p1", *np.random.rand(1).astype(np.float32))
    model.set_initializer("p2", *np.random.rand(1).astype(np.float32))
    model.set_initializer("p3", *np.random.rand(1).astype(np.float32))
    model.set_initializer("p4", *np.random.rand(1).astype(np.float32))
    model.set_initializer("p5", *np.random.rand(1).astype(np.float32))

    # Transform
    new_model = model.transform(transf_fxn)

    # Test
    inp_dict = {"top_in": np.random.rand(*input_shape).astype(np.float32)}
    assert ox.compare_execution(model, new_model, inp_dict)
    assert len(new_model.graph.node) == 5
