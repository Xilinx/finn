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

from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.im2col import compute_conv_output_dim
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor

import finn.core.onnx_exec as oxe
from finn.transformation.streamline.reorder import MoveMulPastDWConv


@pytest.mark.streamline
# input dimension
@pytest.mark.parametrize("ifm_dim", [4, 7])
# input channels
@pytest.mark.parametrize("ifm_ch", [2, 3])
# kernel size
@pytest.mark.parametrize("k", [2, 3])
# stride
@pytest.mark.parametrize("stride", [1, 2])
# padding
@pytest.mark.parametrize("pad_amt", [0, 1])
# depthwise
@pytest.mark.parametrize("dw", [0, 1])
def test_move_mul_past_dw_conv(ifm_dim, ifm_ch, k, stride, pad_amt, dw):
    if dw == 1:
        ofm_ch = ifm_ch
        groups = ifm_ch
        W_shape = [ofm_ch, 1, k, k]
    else:
        ofm_ch = ifm_ch + 2
        groups = 1
        W_shape = [ofm_ch, ifm_ch, k, k]
    total_pad = 2 * pad_amt
    ofm_dim = compute_conv_output_dim(ifm_dim, k, stride, total_pad)

    # set up onnx model
    inp = helper.make_tensor_value_info(
        "inp", TensorProto.FLOAT, [1, ifm_ch, ifm_dim, ifm_dim]
    )
    mul = helper.make_tensor_value_info("mul", TensorProto.FLOAT, [1, ifm_ch, 1, 1])
    W = helper.make_tensor_value_info("W", TensorProto.FLOAT, W_shape)
    outp = helper.make_tensor_value_info(
        "outp", TensorProto.FLOAT, [1, ofm_ch, ofm_dim, ofm_dim]
    )

    Mul_node = helper.make_node("Mul", ["inp", "mul"], ["mul_out"])

    Conv_node = helper.make_node(
        "Conv",
        ["mul_out", "W"],
        ["outp"],
        group=groups,
        kernel_shape=[k, k],
        pads=[pad_amt, pad_amt, pad_amt, pad_amt],
        strides=[stride, stride],
    )

    graph = helper.make_graph(
        nodes=[Mul_node, Conv_node],
        name="mulpastconv_graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[mul, W],
    )

    model = helper.make_model(graph, producer_name="mulpastconv-model")
    model = ModelWrapper(model)
    inp_values = gen_finn_dt_tensor(DataType["INT2"], [1, ifm_ch, ifm_dim, ifm_dim])
    mul_values = gen_finn_dt_tensor(DataType["INT2"], [1, ifm_ch, 1, 1])
    W_values = gen_finn_dt_tensor(DataType["INT2"], W_shape)
    model.set_initializer("W", W_values)
    model.set_initializer("mul", mul_values)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    idict = {"inp": inp_values}
    odict = oxe.execute_onnx(model, idict, True)
    out_before = odict["outp"]

    # move channelwise multiplication past depthwise conv
    model_transformed = model.transform(MoveMulPastDWConv())
    odict = oxe.execute_onnx(model_transformed, idict, True)
    out_after = odict["outp"]

    assert (out_before == out_after).all()

    if dw == 0:
        assert model.graph.node[0].op_type == model_transformed.graph.node[0].op_type
        assert model.graph.node[1].op_type == model_transformed.graph.node[1].op_type
    else:
        assert model.graph.node[0].op_type == model_transformed.graph.node[1].op_type
        assert model.graph.node[1].op_type == model_transformed.graph.node[0].op_type
