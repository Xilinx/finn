# Copyright (C) 2024, Advanced Micro Devices, Inc.
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
from onnx import TensorProto
from onnx import helper as oh
from os.path import join
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.transformation.streamline.reorder import (
    MoveAddPastJoinConcat,
    MoveMulPastJoinConcat,
    MoveTransposePastJoinConcat,
)


def create_concat_model(identical_op):
    perm = None
    channelwise = False
    if "Transpose" in identical_op:
        perm = identical_op.split("_")[1]
        identical_op = identical_op.split("_")[0]
        perm = [int(char) for char in perm]
    if "channelwise" in identical_op:
        channelwise = True
        identical_op = identical_op.split("_")[0]
    if perm == [0, 2, 3, 1]:
        in_shape1 = [1, 64, 10, 9]
        in_shape2 = [1, 32, 10, 9]
        out_shape1 = [1, 10, 9, 64]
        out_shape2 = [1, 10, 9, 32]
        out_join_shape = [1, 10, 9, 96]
        concat_axis = 3
    elif perm == [0, 3, 1, 2]:
        in_shape1 = [1, 10, 9, 64]
        in_shape2 = [1, 10, 9, 32]
        out_shape1 = [1, 64, 10, 9]
        out_shape2 = [1, 32, 10, 9]
        out_join_shape = [1, 96, 10, 9]
        concat_axis = 1
    else:
        in_shape1 = [1, 64, 10, 9]
        in_shape2 = [1, 32, 10, 9]
        out_shape1 = in_shape1
        out_shape2 = in_shape2
        out_join_shape = [1, 96, 10, 9]
        concat_axis = 1
        if channelwise:
            op1_param_shape = [1, 64, 1, 1]
            op2_param_shape = [1, 32, 1, 1]
            op1_param = np.ones((1, 64, 1, 1)) * 2
            op2_param = np.ones((1, 32, 1, 1)) * 3
        else:
            op1_param_shape = [1]
            op2_param_shape = [1]
            op1_param = 1.5
            op2_param = 1.5

    op1_node = oh.make_node(identical_op, inputs=["in1"], outputs=["op1_out"])

    op2_node = oh.make_node(identical_op, inputs=["in2"], outputs=["op2_out"])

    if identical_op == "Transpose":
        new_attr = oh.make_attribute("perm", perm)
        op1_node.attribute.append(new_attr)
        op2_node.attribute.append(new_attr)
    elif identical_op == "Mul" or identical_op == "Add":
        op1_init = oh.make_tensor_value_info("op1_param", TensorProto.FLOAT, op1_param_shape)
        op2_init = oh.make_tensor_value_info("op2_param", TensorProto.FLOAT, op2_param_shape)
        op1_node.input.append(op1_init.name)
        op2_node.input.append(op2_init.name)

    concat_node = oh.make_node(
        "Concat", inputs=["op1_out", "op2_out"], outputs=["out_join1"], axis=concat_axis
    )

    in1 = oh.make_tensor_value_info("in1", TensorProto.FLOAT, in_shape1)
    in2 = oh.make_tensor_value_info("in2", TensorProto.FLOAT, in_shape2)
    op1_out = oh.make_tensor_value_info("op1_out", TensorProto.FLOAT, out_shape1)
    op2_out = oh.make_tensor_value_info("op2_out", TensorProto.FLOAT, out_shape2)
    out_join1 = oh.make_tensor_value_info("out_join1", TensorProto.FLOAT, out_join_shape)

    graph = oh.make_graph(
        nodes=[op1_node, op2_node, concat_node],
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
    if identical_op == "Mul" or identical_op == "Add":
        model.set_initializer("op1_param", np.array(op1_param).astype(np.float32))
        model.set_initializer("op2_param", np.array(op2_param).astype(np.float32))

    return model


transform_dict = {
    "Transpose_0231": MoveTransposePastJoinConcat(),
    "Transpose_0312": MoveTransposePastJoinConcat(),
    "Mul": MoveMulPastJoinConcat(),
    "Mul_channelwise": MoveMulPastJoinConcat(),
    "Add": MoveAddPastJoinConcat(),
    "Add_channelwise": MoveAddPastJoinConcat(),
}


@pytest.mark.streamline
# Permutation of transpose node
@pytest.mark.parametrize(
    "identical_op",
    ["Transpose_0231", "Transpose_0312", "Mul", "Add", "Mul_channelwise", "Add_channelwise"],
)
def test_move_identical_op_past_join_concat(identical_op):
    model = create_concat_model(identical_op)
    build_dir = os.environ["FINN_BUILD_DIR"]
    model.save(join(build_dir, "concat_pytest_model_{}.onnx".format(identical_op)))

    # Create input data
    input0_tensor_name = model.graph.input[0].name
    input1_tensor_name = model.graph.input[1].name

    # Note: it is assumed that both tensors have the same shape and data type
    input_dict = {}
    input_dict[input0_tensor_name] = gen_finn_dt_tensor(
        model.get_tensor_datatype(input0_tensor_name), model.get_tensor_shape(input0_tensor_name)
    )
    input_dict[input1_tensor_name] = gen_finn_dt_tensor(
        model.get_tensor_datatype(input1_tensor_name), model.get_tensor_shape(input1_tensor_name)
    )

    model_transformed = model.transform(transform_dict[identical_op])
    model_transformed.save(
        join(build_dir, "concat_pytest_model_{}_trans.onnx".format(identical_op))
    )

    assert oxe.compare_execution(model, model_transformed, input_dict)

    # Check if order changed
    node0_input0_model = model.find_consumers(model.graph.input[0].name)[0].op_type
    node1_input1_model = model.find_consumers(model.graph.input[1].name)[0].op_type
    node0_input0_model_transformed = model_transformed.find_consumers(
        model_transformed.graph.input[0].name
    )[0].op_type
    node1_input1_model_transformed = model_transformed.find_consumers(
        model_transformed.graph.input[1].name
    )[0].op_type
    assert node0_input0_model != node0_input0_model_transformed
    assert node1_input1_model != node1_input1_model_transformed
