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

from onnx import TensorProto
from onnx import helper as oh
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import gen_finn_dt_tensor

import finn.core.onnx_exec as oxe
from finn.transformation.streamline.reorder import MoveTransposePastJoinAdd


def create_model(perm):
    if perm == [0, 3, 1, 2]:
        in_shape = [1, 128, 1, 256]
        out_shape = [1, 256, 128, 1]
    if perm == [0, 2, 3, 1]:
        in_shape = [1, 256, 128, 1]
        out_shape = [1, 128, 1, 256]

    Transpose1_node = oh.make_node(
        "Transpose", inputs=["in_transpose1"], outputs=["out_transpose1"], perm=perm
    )

    Transpose2_node = oh.make_node(
        "Transpose", inputs=["in_transpose2"], outputs=["out_transpose2"], perm=perm
    )

    Join1_node = oh.make_node(
        "Add", inputs=["out_transpose1", "out_transpose2"], outputs=["out_join1"]
    )

    in_transpose1 = oh.make_tensor_value_info(
        "in_transpose1", TensorProto.FLOAT, in_shape
    )
    in_transpose2 = oh.make_tensor_value_info(
        "in_transpose2", TensorProto.FLOAT, in_shape
    )
    out_transpose1 = oh.make_tensor_value_info(
        "out_transpose1", TensorProto.FLOAT, out_shape
    )
    out_transpose2 = oh.make_tensor_value_info(
        "out_transpose2", TensorProto.FLOAT, out_shape
    )
    out_join1 = oh.make_tensor_value_info("out_join1", TensorProto.FLOAT, out_shape)

    graph = oh.make_graph(
        nodes=[Transpose1_node, Transpose2_node, Join1_node],
        name="test_graph",
        inputs=[in_transpose1, in_transpose2],
        outputs=[out_join1],
        value_info=[
            out_transpose1,
            out_transpose2,
        ],
    )

    onnx_model = oh.make_model(graph, producer_name="test_model")
    model = ModelWrapper(onnx_model)

    return model


@pytest.mark.streamline
# Permutation of transpose node
@pytest.mark.parametrize("perm", [[0, 3, 1, 2], [0, 2, 3, 1]])
def test_move_identical_op_past_join_op(perm):
    model = create_model(perm)

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

    model_transformed = model.transform(MoveTransposePastJoinAdd())

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
