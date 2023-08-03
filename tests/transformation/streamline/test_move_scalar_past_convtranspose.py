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
from qonnx.util.basic import qonnx_make_model

import finn.core.onnx_exec as ox
from finn.transformation.streamline.reorder import MoveScalarMulPastConvTranspose


@pytest.mark.streamline
# input image dimension
@pytest.mark.parametrize("idim", [[8, 8], [10, 8]])
# number of rows and number of cols to add
@pytest.mark.parametrize("stride", [[2, 2], [2, 3]])
# number of channels
@pytest.mark.parametrize("ifm_ch", [2, 4])
# number of channels
@pytest.mark.parametrize("ofm_ch", [2, 4])
# kernel size
@pytest.mark.parametrize("k", [2, 4])
# padding
@pytest.mark.parametrize("padding", [False, True])
def test_move_scalar_past_conv(idim, stride, ifm_ch, ofm_ch, k, padding):
    idim_h, idim_w = idim
    stride_h, stride_w = stride

    odim_h = (idim_h - 1) * stride_h - 2 * padding + (k - 1) + 1
    odim_w = (idim_w - 1) * stride_w - 2 * padding + (k - 1) + 1

    input_shape = [1, ifm_ch, idim_h, idim_w]
    output_shape = [1, ofm_ch, odim_h, odim_w]

    conv_param_shape = [ifm_ch, ofm_ch, k, k]

    conv_config = {}
    conv_config["dilations"] = [1, 1]
    conv_config["group"] = 1
    conv_config["kernel_shape"] = [k, k]
    if padding:
        conv_config["pads"] = [1, 1, 1, 1]
    else:
        conv_config["pads"] = [0, 0, 0, 0]
    conv_config["strides"] = [stride_h, stride_w]

    top_in = oh.make_tensor_value_info("top_in", TensorProto.FLOAT, input_shape)
    top_out = oh.make_tensor_value_info("top_out", TensorProto.FLOAT, output_shape)

    value_info = [oh.make_tensor_value_info("p1", TensorProto.FLOAT, [1])]
    value_info += [oh.make_tensor_value_info("p2", TensorProto.FLOAT, conv_param_shape)]

    modelproto = qonnx_make_model(
        oh.make_graph(
            name="test",
            inputs=[top_in],
            outputs=[top_out],
            value_info=value_info,
            nodes=[
                oh.make_node("Mul", ["top_in", "p1"], ["t1"]),
                oh.make_node("ConvTranspose", ["t1", "p2"], ["top_out"], **conv_config),
            ],
        )
    )
    model = ModelWrapper(modelproto)
    model = model.transform(InferShapes())

    np.random.seed(0)
    model.set_initializer("p1", *np.random.rand(1).astype(np.float32))
    model.set_initializer("p2", np.random.rand(*conv_param_shape).astype(np.float32))
    new_model = model.transform(MoveScalarMulPastConvTranspose())
    inp_dict = {"top_in": np.random.rand(*input_shape).astype(np.float32)}

    assert ox.compare_execution(model, new_model, inp_dict)
    assert new_model.graph.node[0].op_type == "ConvTranspose"
    assert new_model.graph.node[1].op_type == "Mul"
