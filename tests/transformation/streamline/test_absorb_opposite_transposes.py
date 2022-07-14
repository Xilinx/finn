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

import finn.core.onnx_exec as ox
from finn.transformation.streamline.absorb import AbsorbConsecutiveTransposes


@pytest.mark.streamline
def test_absorb_opposite_transposes():
    np.random.seed(0)
    input_shape = [1, 3, 4, 2]
    top_in = oh.make_tensor_value_info("top_in", TensorProto.FLOAT, input_shape)
    top_out = oh.make_tensor_value_info("top_out", TensorProto.FLOAT, input_shape)
    value_info = [oh.make_tensor_value_info("add_param_0", TensorProto.FLOAT, [1])]
    value_info += [oh.make_tensor_value_info("add_param_1", TensorProto.FLOAT, [1])]
    value_info += [oh.make_tensor_value_info("mul_param_0", TensorProto.FLOAT, [1])]
    modelproto = oh.make_model(
        oh.make_graph(
            name="test",
            inputs=[top_in],
            outputs=[top_out],
            value_info=value_info,
            nodes=[
                oh.make_node("Add", ["top_in", "add_param_0"], ["t0"]),
                oh.make_node("Transpose", ["t0"], ["t1"], perm=[0, 2, 3, 1]),
                oh.make_node("Transpose", ["t1"], ["t2"], perm=[0, 3, 1, 2]),
                oh.make_node("Add", ["t2", "add_param_1"], ["t3"]),
                oh.make_node("Transpose", ["t3"], ["t4"], perm=[0, 2, 3, 1]),
                oh.make_node("Transpose", ["t4"], ["t5"], perm=[0, 3, 1, 2]),
                oh.make_node("Add", ["t5", "t2"], ["t6"]),
                oh.make_node("Mul", ["t6", "mul_param_0"], ["top_out"]),
            ],
        )
    )
    model = ModelWrapper(modelproto)
    model = model.transform(InferShapes())
    model.set_initializer("add_param_0", np.asarray([1], dtype=np.float32))
    model.set_initializer("add_param_1", np.asarray([3], dtype=np.float32))
    model.set_initializer("mul_param_0", np.asarray([2], dtype=np.float32))
    new_model = model.transform(AbsorbConsecutiveTransposes())
    new_model = new_model.transform(InferShapes())
    inp_dict = {"top_in": np.random.rand(*input_shape).astype(np.float32)}
    assert ox.compare_execution(model, model, inp_dict)
    assert len(new_model.graph.node) == 4
    for n in new_model.graph.node:
        assert new_model.graph.node[0].op_type != "Transpose"
