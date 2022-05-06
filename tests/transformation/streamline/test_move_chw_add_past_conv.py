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

import finn.core.onnx_exec as oxe
from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.general.im2col import compute_conv_output_dim
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.streamline.reorder import MoveAddPastConv


@pytest.mark.streamline
# input dimension
@pytest.mark.parametrize("idim", [4, 7])
# kernel size
@pytest.mark.parametrize("k", [2, 3])
# stride
@pytest.mark.parametrize("s", [1, 2])
# input channels
@pytest.mark.parametrize("ich", [2, 4])
# output channels
@pytest.mark.parametrize("och", [2, 3])
def test_move_chw_add_past_conv(idim, k, s, ich, och):
    odim = compute_conv_output_dim(idim, k, s)

    ishape = [1, ich, idim, idim]
    oshape = [1, och, odim, odim]
    add_param_shape = [1, ich, 1, 1]
    conv_param_shape = [och, ich, k, k]

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, ishape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, oshape)
    a0 = helper.make_tensor_value_info("a0", TensorProto.FLOAT, add_param_shape)
    a1 = helper.make_tensor_value_info("a1", TensorProto.FLOAT, conv_param_shape)

    conv_config = {}
    conv_config["dilations"] = [1, 1]
    conv_config["group"] = 1
    conv_config["kernel_shape"] = [k, k]
    conv_config["pads"] = [0, 0, 0, 0]
    conv_config["strides"] = [s, s]

    add_node = helper.make_node("Add", ["inp", "a0"], ["add_out"])
    conv_node = helper.make_node("Conv", ["add_out", "a1"], ["outp"], **conv_config)

    model = helper.make_model(
        helper.make_graph(
            nodes=[add_node, conv_node],
            name="move-add-graph",
            inputs=[inp],
            outputs=[outp],
            value_info=[a0, a1],
        )
    )

    model = ModelWrapper(model)
    # initialize model
    a0_values = np.random.uniform(low=0, high=1, size=tuple(add_param_shape)).astype(
        np.float32
    )
    model.set_initializer("a0", a0_values)
    a1_values = np.random.uniform(low=0, high=1, size=tuple(conv_param_shape)).astype(
        np.float32
    )
    model.set_initializer("a1", a1_values)

    model = model.transform(InferShapes())

    # execution before transformation
    inp_values = np.random.uniform(low=0, high=1, size=tuple(ishape)).astype(np.float32)
    idict = {model.graph.input[0].name: inp_values}
    odict = oxe.execute_onnx(model, idict)
    y_before = odict[model.graph.output[0].name]

    model = model.transform(MoveAddPastConv())
    odict = oxe.execute_onnx(model, idict)
    y_after = odict[model.graph.output[0].name]

    assert np.isclose(y_before, y_after).all()
    assert model.graph.node[0].op_type == "Conv"
    assert model.graph.node[1].op_type == "Add"
