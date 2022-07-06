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
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes

import finn.core.onnx_exec as oxe
from finn.transformation.streamline.reorder import MoveLinearPastFork


@pytest.mark.streamline
@pytest.mark.parametrize("ch", [64, 1])
# ifmdim
@pytest.mark.parametrize("ifmdim", [-1, 7])
def test_move_past_fork(ch, ifmdim):
    # generate test vectors of correct shape
    if ifmdim == -1:
        input_shape = (1, ch)
    else:
        input_shape = (1, ch, ifmdim, ifmdim)

    top_in = helper.make_tensor_value_info("top_in", TensorProto.FLOAT, input_shape)
    top_out = helper.make_tensor_value_info("top_out", TensorProto.FLOAT, input_shape)

    num_of_params = 8
    value_info = []
    for i in range(num_of_params):
        value_info += [
            helper.make_tensor_value_info("p" + str(i), TensorProto.FLOAT, input_shape)
        ]

    add_1_to_move = helper.make_node("Add", ["top_in", "p0"], ["fork1"])
    mul_1_to_move = helper.make_node("Mul", ["t5", "p4"], ["fork2"])
    add_2_to_move = helper.make_node("Add", ["fork2", "p5"], ["t6"])
    mul_1_not_to_move = helper.make_node("Mul", ["t8", "p7"], ["fork3"])
    modelproto = helper.make_model(
        helper.make_graph(
            name="test",
            inputs=[top_in],
            outputs=[top_out],
            value_info=value_info,
            nodes=[
                # fork1
                add_1_to_move,
                helper.make_node("Mul", ["fork1", "p1"], ["t2"]),
                helper.make_node("Mul", ["fork1", "p2"], ["t3"]),
                helper.make_node("Add", ["t2", "t3"], ["t4"]),
                helper.make_node("Add", ["t4", "p3"], ["t5"]),
                # fork2
                mul_1_to_move,
                add_2_to_move,
                helper.make_node("Add", ["fork2", "p6"], ["t7"]),
                helper.make_node("Add", ["t6", "t7"], ["t8"]),
                # empty branches: do nothing
                mul_1_not_to_move,
                helper.make_node("Add", ["fork3", "fork3"], ["top_out"]),
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

    # Transform
    new_model = model.transform(MoveLinearPastFork())
    inp_dict = {"top_in": np.random.rand(*input_shape).astype(np.float32)}

    # Test
    assert oxe.compare_execution(model, new_model, inp_dict)
    assert not new_model.is_fork_node(add_1_to_move)
    assert not new_model.is_fork_node(mul_1_to_move)
    assert not new_model.is_fork_node(add_2_to_move)
    assert new_model.is_fork_node(mul_1_not_to_move)
    assert len(new_model.graph.node) == 14
