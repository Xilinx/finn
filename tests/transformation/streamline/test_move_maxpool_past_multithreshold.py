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
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes

import finn.core.onnx_exec as oxe
from finn.transformation.streamline.reorder import MoveMaxPoolPastMultiThreshold


def get_multithreshold_rand_params(channels, num_of_thres, seed=None):
    if seed is not None:
        np.random.seed(seed)
    steps = np.random.rand(channels, 1) * 2
    bias = np.random.rand(channels, 1) * 10
    thres = [np.arange(num_of_thres) for chn in range(channels)]
    thres = ((thres - bias) * steps).astype(np.float32)
    return thres


@pytest.mark.streamline
def test_move_maxpool_past_multithreshold():
    # generate test vectors of correct shape
    ch = 64
    ifmdim = 16
    ofmdim = 16 // 4
    input_shape = (1, ch, ifmdim, ifmdim)
    output_shape = (1, ch, ofmdim, ofmdim)

    top_in = helper.make_tensor_value_info("top_in", TensorProto.FLOAT, input_shape)
    top_out = helper.make_tensor_value_info("top_out", TensorProto.FLOAT, output_shape)

    maxpool_config = {}
    maxpool_config["pads"] = [1, 1, 1, 1]
    maxpool_config["kernel_shape"] = [3, 3]
    maxpool_config["strides"] = [2, 2]

    value_info = []
    thres1_shape = [1, 1]
    value_info += [
        helper.make_tensor_value_info("thres1", TensorProto.FLOAT, thres1_shape)
    ]

    thres2_shape = [ch, 14]
    value_info += [
        helper.make_tensor_value_info("thres2", TensorProto.FLOAT, thres2_shape)
    ]

    nodes = []
    nodes += [helper.make_node("MaxPool", ["top_in"], ["t1"], **maxpool_config)]
    nodes += [
        helper.make_node(
            "MultiThreshold",
            ["t1", "thres1"],
            ["t2"],
            domain="finn.custom_op.general",
            out_dtype="BIPOLAR",
            out_bias=-1.0,
            out_scale=1.0,
        )
    ]
    nodes += [helper.make_node("MaxPool", ["t2"], ["t3"], **maxpool_config)]
    nodes += [
        helper.make_node(
            "MultiThreshold",
            ["t3", "thres2"],
            ["top_out"],
            domain="finn.custom_op.general",
            out_dtype="UINT4",
        )
    ]

    modelproto = helper.make_model(
        helper.make_graph(
            name="test",
            inputs=[top_in],
            outputs=[top_out],
            value_info=value_info,
            nodes=nodes,
        )
    )
    model = ModelWrapper(modelproto)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    model.set_initializer("thres1", np.array([[0]]))
    model.set_initializer(
        "thres2", get_multithreshold_rand_params(*thres2_shape, seed=0)
    )

    # Transform
    new_model = model.transform(MoveMaxPoolPastMultiThreshold())
    inp_dict = {"top_in": np.random.rand(*input_shape).astype(np.float32)}

    # Test
    assert oxe.compare_execution(model, new_model, inp_dict)
    assert new_model.graph.node[0].op_type == "MaxPool"
    assert new_model.graph.node[1].op_type == "MultiThreshold"
    assert new_model.graph.node[2].op_type == "MultiThreshold"
    assert new_model.graph.node[3].op_type == "MaxPool"
    assert len(new_model.graph.node) == 4
