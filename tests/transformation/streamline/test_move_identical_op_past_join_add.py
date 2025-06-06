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


def create_add_model(identical_op):
    perm = None
    if "Transpose" in identical_op:
        perm = identical_op.split("_")[1]
        identical_op = identical_op.split("_")[0]
        perm = [int(char) for char in perm]
    if perm == [0, 2, 3, 1]:
        in_shape = [1, 64, 10, 9]
        out_shape = [1, 10, 9, 64]
    elif perm == [0, 3, 1, 2]:
        in_shape = [1, 10, 9, 64]
        out_shape = [1, 64, 10, 9]
    else:
        in_shape = [1, 64, 10, 9]
        out_shape = in_shape
    op_value = 1.5

    op1_node = oh.make_node(identical_op, inputs=["in1"], outputs=["op1_out"])

    op2_node = oh.make_node(identical_op, inputs=["in2"], outputs=["op2_out"])

    if identical_op == "Transpose":
        new_attr = oh.make_attribute("perm", perm)
        op1_node.attribute.append(new_attr)
        op2_node.attribute.append(new_attr)
    elif identical_op == "Mul" or identical_op == "Add":
        op1_init = oh.make_tensor_value_info("op1_param", TensorProto.FLOAT, [1])
        op2_init = oh.make_tensor_value_info("op2_param", TensorProto.FLOAT, [1])
        op1_node.input.append(op1_init.name)
        op2_node.input.append(op2_init.name)

    add_node = oh.make_node("Add", inputs=["op1_out", "op2_out"], outputs=["out_join1"])

    in1 = oh.make_tensor_value_info("in1", TensorProto.FLOAT, in_shape)
    in2 = oh.make_tensor_value_info("in2", TensorProto.FLOAT, in_shape)
    op1_out = oh.make_tensor_value_info("op1_out", TensorProto.FLOAT, out_shape)
    op2_out = oh.make_tensor_value_info("op2_out", TensorProto.FLOAT, out_shape)
    out_join1 = oh.make_tensor_value_info("out_join1", TensorProto.FLOAT, out_shape)

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
    if identical_op == "Mul" or identical_op == "Add":
        model.set_initializer("op1_param", np.array(op_value).astype(np.float32))
        model.set_initializer("op2_param", np.array(op_value).astype(np.float32))

    return model


transform_dict = {
    "Transpose_0231": MoveTransposePastJoinAdd(),
    "Transpose_0312": MoveTransposePastJoinAdd(),
    "Mul": MoveMulPastJoinAdd(),
    "Add": MoveAddPastJoinAdd(),
}


@pytest.mark.streamline
# Permutation of transpose node
@pytest.mark.parametrize("identical_op", ["Transpose_0231", "Transpose_0312", "Mul", "Add"])
def test_move_identical_op_past_join_op(identical_op):
    model = create_add_model(identical_op)
    # build_dir = os.environ["FINN_BUILD_DIR"]
    # model.save(join(build_dir, "add_pytest_model_{}.onnx".format(identical_op)))

    # Create input data
    input0_tensor_name = model.graph.input[0].name
    input1_tensor_name = model.graph.input[1].name

    # Note: it is assumed that both tensors have the same shape and data type
    input_shape = model.get_tensor_shape(input0_tensor_name)
    input_dtype = model.get_tensor_datatype(input0_tensor_name)
    input_val = gen_finn_dt_tensor(input_dtype, input_shape)
    input_dict = {}
    input_dict[input0_tensor_name] = input_val
    input_dict[input1_tensor_name] = input_val

    model_transformed = model.transform(transform_dict[identical_op])
    # model_transformed.save(join(build_dir, "add_pytest_model_{}_trans.onnx".format(identical_op)))

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
