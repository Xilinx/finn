# Copyright (C) 2025, Advanced Micro Devices, Inc.
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
from onnx import TensorProto
from onnx import helper as oh
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.transformation.streamline.reorder import (
    MoveAddPastJoinAdd,
    MoveMulPastJoinAdd,
    MoveTransposePastJoinAdd,
)

moveop_details = {
    "Transpose_0231_nocast": {
        "op_type": "Transpose",
        "transform": MoveTransposePastJoinAdd(),
        "in1_shape": [1, 64, 10, 9],
        "op1_shape": [1, 10, 9, 64],
        "in2_shape": [1, 64, 10, 9],
        "op2_shape": [1, 10, 9, 64],
        "out_shape": [1, 10, 9, 64],
        "perm": [0, 2, 3, 1],
    },
    "Transpose_0231_bcast1": {
        "op_type": "Transpose",
        "transform": MoveTransposePastJoinAdd(),
        "in1_shape": [1, 1, 10, 9],
        "op1_shape": [1, 10, 9, 1],
        "in2_shape": [1, 64, 10, 9],
        "op2_shape": [1, 10, 9, 64],
        "out_shape": [1, 10, 9, 64],
        "perm": [0, 2, 3, 1],
    },
    "Transpose_0231_bcast2": {
        "op_type": "Transpose",
        "transform": MoveTransposePastJoinAdd(),
        "in1_shape": [1, 64, 10, 9],
        "op1_shape": [1, 10, 9, 64],
        "in2_shape": [1, 64, 1, 9],
        "op2_shape": [1, 1, 9, 64],
        "out_shape": [1, 10, 9, 64],
        "perm": [0, 2, 3, 1],
    },
    "Transpose_0312_nocast": {
        "op_type": "Transpose",
        "transform": MoveTransposePastJoinAdd(),
        "in1_shape": [1, 10, 9, 64],
        "op1_shape": [1, 64, 10, 9],
        "in2_shape": [1, 10, 9, 64],
        "op2_shape": [1, 64, 10, 9],
        "out_shape": [1, 64, 10, 9],
        "perm": [0, 3, 1, 2],
    },
    "Transpose_0312_bcast1": {
        "op_type": "Transpose",
        "transform": MoveTransposePastJoinAdd(),
        "in1_shape": [1, 10, 9, 1],
        "op1_shape": [1, 1, 10, 9],
        "in2_shape": [1, 10, 9, 64],
        "op2_shape": [1, 64, 10, 9],
        "out_shape": [1, 64, 10, 9],
        "perm": [0, 3, 1, 2],
    },
    "Transpose_0312_bcast2": {
        "op_type": "Transpose",
        "transform": MoveTransposePastJoinAdd(),
        "in1_shape": [1, 10, 9, 64],
        "op1_shape": [1, 64, 10, 9],
        "in2_shape": [1, 1, 9, 64],
        "op2_shape": [1, 64, 1, 9],
        "out_shape": [1, 64, 10, 9],
        "perm": [0, 3, 1, 2],
    },
    "Mul_nocast": {
        "op_type": "Mul",
        "transform": MoveMulPastJoinAdd(),
        "in1_shape": [1, 64, 10, 9],
        "op1_shape": [1, 64, 10, 9],
        "in2_shape": [1, 64, 10, 9],
        "op2_shape": [1, 64, 10, 9],
        "out_shape": [1, 64, 10, 9],
        "op1_val": 1.415,
        "op2_val": 1.415,
    },
    "Mul_bcast1": {
        "op_type": "Mul",
        "transform": MoveMulPastJoinAdd(),
        "in1_shape": [1, 1, 10, 9],
        "op1_shape": [1, 1, 10, 9],
        "in2_shape": [1, 64, 10, 9],
        "op2_shape": [1, 64, 10, 9],
        "out_shape": [1, 64, 10, 9],
        "op1_val": -0.476,
        "op2_val": -0.476,
    },
    "Mul_bcast2": {
        "op_type": "Mul",
        "transform": MoveMulPastJoinAdd(),
        "in1_shape": [1, 64, 10, 9],
        "op1_shape": [1, 64, 10, 9],
        "in2_shape": [1, 64, 1, 9],
        "op2_shape": [1, 64, 1, 9],
        "out_shape": [1, 64, 10, 9],
        "op1_val": -1.374,
        "op2_val": -1.374,
    },
    "Add_nocast": {
        "op_type": "Add",
        "transform": MoveAddPastJoinAdd(),
        "in1_shape": [1, 64, 10, 9],
        "op1_shape": [1, 64, 10, 9],
        "in2_shape": [1, 64, 10, 9],
        "op2_shape": [1, 64, 10, 9],
        "out_shape": [1, 64, 10, 9],
        "op1_val": 1.415,
        "op2_val": 0.745,
    },
    "Add_bcast1": {
        "op_type": "Add",
        "transform": MoveAddPastJoinAdd(),
        "in1_shape": [1, 1, 10, 9],
        "op1_shape": [1, 1, 10, 9],
        "in2_shape": [1, 64, 10, 9],
        "op2_shape": [1, 64, 10, 9],
        "out_shape": [1, 64, 10, 9],
        "op1_val": -0.476,
        "op2_val": 1.539,
    },
    "Add_bcast2": {
        "op_type": "Add",
        "transform": MoveAddPastJoinAdd(),
        "in1_shape": [1, 64, 10, 9],
        "op1_shape": [1, 64, 10, 9],
        "in2_shape": [1, 64, 1, 9],
        "op2_shape": [1, 64, 1, 9],
        "out_shape": [1, 64, 10, 9],
        "op1_val": -1.374,
        "op2_val": -0.295,
    },
}


def create_add_model(test_details):
    op1_node = oh.make_node(test_details["op_type"], inputs=["in1"], outputs=["op1_out"])
    op2_node = oh.make_node(test_details["op_type"], inputs=["in2"], outputs=["op2_out"])

    if test_details["op_type"] == "Transpose":
        new_attr = oh.make_attribute("perm", test_details["perm"])
        op1_node.attribute.append(new_attr)
        op2_node.attribute.append(new_attr)
    elif test_details["op_type"] == "Mul" or test_details["op_type"] == "Add":
        op1_init = oh.make_tensor_value_info("op1_param", TensorProto.FLOAT, [1])
        op2_init = oh.make_tensor_value_info("op2_param", TensorProto.FLOAT, [1])
        op1_node.input.append(op1_init.name)
        op2_node.input.append(op2_init.name)

    add_node = oh.make_node("Add", inputs=["op1_out", "op2_out"], outputs=["out_join1"])

    in1 = oh.make_tensor_value_info("in1", TensorProto.FLOAT, test_details["in1_shape"])
    in2 = oh.make_tensor_value_info("in2", TensorProto.FLOAT, test_details["in2_shape"])
    op1_out = oh.make_tensor_value_info("op1_out", TensorProto.FLOAT, test_details["op1_shape"])
    op2_out = oh.make_tensor_value_info("op2_out", TensorProto.FLOAT, test_details["op2_shape"])
    out_join1 = oh.make_tensor_value_info("out_join1", TensorProto.FLOAT, test_details["out_shape"])

    graph = oh.make_graph(
        nodes=[op1_node, op2_node, add_node],
        name="test_graph",
        inputs=[in1, in2],
        outputs=[out_join1],
        value_info=[
            op1_out,
            op2_out,
        ],
    )

    onnx_model = qonnx_make_model(graph, producer_name="test_model")
    model = ModelWrapper(onnx_model)
    if test_details["op_type"] == "Mul" or test_details["op_type"] == "Add":
        model.set_initializer("op1_param", np.array(test_details["op1_val"]).astype(np.float32))
        model.set_initializer("op2_param", np.array(test_details["op2_val"]).astype(np.float32))

    return model


@pytest.mark.streamline
@pytest.mark.parametrize("moveop_key", moveop_details.keys())
def test_move_identical_op_past_join_op(moveop_key):
    test_details = moveop_details[moveop_key]

    model = create_add_model(test_details)

    # Create input data
    input0_tensor_name = model.graph.input[0].name
    input1_tensor_name = model.graph.input[1].name

    input0_shape = model.get_tensor_shape(input0_tensor_name)
    input0_dtype = model.get_tensor_datatype(input0_tensor_name)
    input1_shape = model.get_tensor_shape(input1_tensor_name)
    input1_dtype = model.get_tensor_datatype(input1_tensor_name)
    input0_val = gen_finn_dt_tensor(input0_dtype, input0_shape)
    input1_val = gen_finn_dt_tensor(input1_dtype, input1_shape)
    input_dict = {}
    input_dict[input0_tensor_name] = input0_val
    input_dict[input1_tensor_name] = input1_val

    model_transformed = model.transform(test_details["transform"])

    assert oxe.compare_execution(model, model_transformed, input_dict)

    # Check if order changed
    node0_optype_model = model.find_consumers(model.graph.input[0].name)[0].op_type
    node1_optype_model = model.find_consumers(model.graph.input[1].name)[0].op_type
    node0_optype_model_transformed = model_transformed.find_consumers(
        model_transformed.graph.input[0].name
    )[0].op_type
    node1_optype_model_transformed = model_transformed.find_consumers(
        model_transformed.graph.input[1].name
    )[0].op_type
    last_node_optype_model_transformed = model_transformed.find_producer(
        model_transformed.graph.output[0].name
    ).op_type
    assert node0_optype_model == last_node_optype_model_transformed
    assert node1_optype_model == last_node_optype_model_transformed
    assert node0_optype_model_transformed == node1_optype_model_transformed == "Add"
