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
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.transformation.streamline.reorder import (
    MoveScalarLinearPastSplit,
    MoveTransposePastSplit,
)


def create_split_model(identical_op):
    perm = None
    if "Transpose" in identical_op:
        perm = identical_op.split("_")[1]
        identical_op = identical_op.split("_")[0]
        perm = [int(char) for char in perm]
    if perm == [0, 2, 3, 1]:
        in_shape = [1, 96, 10, 9]
        out_shape = [1, 10, 9, 96]
        out1_split_shape = [1, 10, 9, 32]
        out2_split_shape = [1, 10, 9, 64]
        split_axis = 3
    elif perm == [0, 3, 1, 2]:
        in_shape = [1, 10, 9, 96]
        out_shape = [1, 96, 10, 9]
        out1_split_shape = [1, 32, 10, 9]
        out2_split_shape = [1, 64, 10, 9]
        split_axis = 1
    else:
        in_shape = [1, 96, 10, 9]
        out_shape = in_shape
        out1_split_shape = [1, 32, 10, 9]
        out2_split_shape = [1, 64, 10, 9]
        split_axis = 1
    op_value = 1.5
    split = [32, 64]

    op_node = oh.make_node(identical_op, inputs=["in1"], outputs=["op_out"])

    if identical_op == "Transpose":
        new_attr = oh.make_attribute("perm", perm)
        op_node.attribute.append(new_attr)
    elif identical_op == "Mul" or identical_op == "Add":
        op_init = oh.make_tensor_value_info("op_param", TensorProto.FLOAT, [1])
        op_node.input.append(op_init.name)

    in1 = oh.make_tensor_value_info("in1", TensorProto.FLOAT, in_shape)
    op_out = oh.make_tensor_value_info("op_out", TensorProto.FLOAT, out_shape)
    out1_split = oh.make_tensor_value_info("out1_split", TensorProto.FLOAT, out1_split_shape)
    out2_split = oh.make_tensor_value_info("out2_split", TensorProto.FLOAT, out2_split_shape)
    split_init = oh.make_tensor_value_info("split", TensorProto.INT64, [2])

    split_node = oh.make_node(
        "Split", [op_out.name, split_init.name], [out1_split.name, out2_split.name], axis=split_axis
    )

    graph = oh.make_graph(
        nodes=[op_node, split_node],
        name="test_graph",
        inputs=[in1],
        outputs=[out1_split, out2_split],
        value_info=[op_out],
    )

    # set opset version to 13 for specific Split configuration
    opset_imports = [oh.make_opsetid("", 13)]
    model = qonnx_make_model(graph, opset_imports=opset_imports)
    model = ModelWrapper(model)
    model.set_initializer(split_init.name, np.array(split, dtype=np.int64))
    if identical_op == "Mul" or identical_op == "Add":
        model.set_initializer(op_init.name, np.array(op_value).astype(np.float32))
    model = model.transform(GiveUniqueNodeNames())

    return model


transform_dict = {
    "Transpose_0231": MoveTransposePastSplit(),
    "Transpose_0312": MoveTransposePastSplit(),
    "Mul": MoveScalarLinearPastSplit(),
    "Add": MoveScalarLinearPastSplit(),
}


@pytest.mark.streamline
# Permutation of transpose node
@pytest.mark.parametrize("identical_op", ["Transpose_0231", "Transpose_0312", "Mul", "Add"])
def test_move_identical_op_past_split(identical_op):
    model = create_split_model(identical_op)

    # Create input data
    input0_tensor_name = model.graph.input[0].name

    # Note: it is assumed that both tensors have the same shape and data type
    input_dict = {}
    input_dict[input0_tensor_name] = gen_finn_dt_tensor(
        model.get_tensor_datatype(input0_tensor_name), model.get_tensor_shape(input0_tensor_name)
    )

    model_transformed = model.transform(transform_dict[identical_op])

    assert oxe.compare_execution(model, model_transformed, input_dict)

    # Check if order changed
    node0_input0_model = model.find_consumers(model.graph.input[0].name)[0].op_type
    node0_input0_model_transformed = model_transformed.find_consumers(
        model_transformed.graph.input[0].name
    )[0].op_type
    assert node0_input0_model != node0_input0_model_transformed
