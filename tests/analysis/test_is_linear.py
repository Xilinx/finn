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

import onnx.helper as oh
from onnx import TensorProto

import finn.analysis.topology as ta
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.infer_shapes import InferShapes


def test_is_linear_linear():
    top_in = oh.make_tensor_value_info("top_in", TensorProto.FLOAT, [2])
    add_param = oh.make_tensor_value_info("add_param", TensorProto.FLOAT, [2])
    mul_param = oh.make_tensor_value_info("mul_param", TensorProto.FLOAT, [2])
    top_out = oh.make_tensor_value_info("top_out", TensorProto.FLOAT, [2])
    modelproto = oh.make_model(
        oh.make_graph(
            name="test",
            inputs=[top_in],
            outputs=[top_out],
            value_info=[add_param, mul_param],
            nodes=[
                oh.make_node("Add", ["top_in", "add_param"], ["middle"]),
                oh.make_node("Mul", ["middle", "mul_param"], ["top_out"]),
            ],
        )
    )
    model = ModelWrapper(modelproto)
    model = model.transform(InferShapes())
    ret = model.analysis(ta.is_linear)
    assert ret["is_linear"] is True


def test_is_linear_forked_node_output():
    top_in = oh.make_tensor_value_info("top_in", TensorProto.FLOAT, [2])
    add_param = oh.make_tensor_value_info("add_param", TensorProto.FLOAT, [2])
    mul0_param = oh.make_tensor_value_info("mul0_param", TensorProto.FLOAT, [2])
    mul1_param = oh.make_tensor_value_info("mul1_param", TensorProto.FLOAT, [2])
    mul0_res = oh.make_tensor_value_info("mul0_res", TensorProto.FLOAT, [2])
    mul1_res = oh.make_tensor_value_info("mul1_res", TensorProto.FLOAT, [2])
    top_out = oh.make_tensor_value_info("top_out", TensorProto.FLOAT, [2])
    modelproto = oh.make_model(
        oh.make_graph(
            name="test",
            inputs=[top_in],
            outputs=[top_out],
            value_info=[add_param, mul0_param, mul1_param, mul0_res, mul1_res],
            nodes=[
                oh.make_node("Add", ["top_in", "add_param"], ["middle"]),
                oh.make_node("Mul", ["middle", "mul0_param"], ["mul0_res"]),
                oh.make_node("Mul", ["middle", "mul1_param"], ["mul1_res"]),
                oh.make_node("Add", ["mul0_res", "mul1_res"], ["top_out"]),
            ],
        )
    )
    model = ModelWrapper(modelproto)
    model = model.transform(InferShapes())
    ret = model.analysis(ta.is_linear)
    assert ret["is_linear"] is False
