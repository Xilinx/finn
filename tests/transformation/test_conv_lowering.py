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
import os
from onnx import helper, TensorProto
import pkg_resources as pk
import brevitas.onnx as bo
import numpy as np

from finn.core.modelwrapper import ModelWrapper
from finn.core.datatype import DataType
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.infer_shapes import InferShapes
from finn.util.test import get_test_model_trained
from finn.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from finn.transformation.double_to_single_float import DoubleToSingleFloat
import finn.core.onnx_exec as oxe
from finn.custom_op.im2col import compute_conv_output_dim
from finn.util.basic import gen_finn_dt_tensor
from finn.custom_op.registry import getCustomOp

export_onnx_path = "test_output_cnv.onnx"


def test_conv_lowering_cnv_w1a1():
    cnv = get_test_model_trained("CNV", 1, 1)
    bo.export_finn_onnx(cnv, (1, 3, 32, 32), export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    model = model.transform(DoubleToSingleFloat())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    fn = pk.resource_filename("finn", "data/cifar10/cifar10-test-data-class3.npz")
    input_tensor = np.load(fn)["arr_0"].astype(np.float32)
    input_tensor = input_tensor / 255
    assert input_tensor.shape == (1, 3, 32, 32)
    # execute imported model to get expected answer
    input_dict = {"0": input_tensor}
    output_dict_e = oxe.execute_onnx(model, input_dict)
    expected = output_dict_e[list(output_dict_e.keys())[0]]
    # execute transformed model and compare
    model = model.transform(LowerConvsToMatMul())
    output_dict_p = oxe.execute_onnx(model, input_dict)
    produced = output_dict_p[list(output_dict_p.keys())[0]]
    assert np.isclose(produced, expected).all()
    assert np.argmax(produced) == 3
    os.remove(export_onnx_path)


# input datatype
@pytest.mark.parametrize("idt", [DataType.INT2, DataType.INT4])
# kernel size
@pytest.mark.parametrize("k", [2, 4])
# input dimension
@pytest.mark.parametrize("ifm_dim", [4, 6])
# input channels
@pytest.mark.parametrize("ifm_ch", [2, 3])
# stride
@pytest.mark.parametrize("stride", [1, 2])
# padding
@pytest.mark.parametrize("padding", [[0, 0, 0, 0], [1, 1, 1, 1]])
def test_depthwise_conv_lowering(idt, k, ifm_dim, ifm_ch, stride, padding):
    odt = wdt = idt
    ofm_ch = ifm_ch
    ofm_dim = compute_conv_output_dim(ifm_dim, k, stride, pad=padding[0])

    # set up onnx model
    inp = helper.make_tensor_value_info(
        "inp", TensorProto.FLOAT, [1, ifm_ch, ifm_dim, ifm_dim]
    )
    outp = helper.make_tensor_value_info(
        "outp", TensorProto.FLOAT, [1, ofm_ch, ofm_dim, ofm_dim]
    )

    W = helper.make_tensor_value_info("W", TensorProto.FLOAT, [ofm_ch, 1, k, k])

    dw_cnv = helper.make_node(
        "Conv",
        inputs=["inp", "W"],
        outputs=["outp"],
        kernel_shape=[k, k],
        pads=padding,
        strides=[stride, stride],
        group=ifm_ch,
    )
    graph = helper.make_graph(
        nodes=[dw_cnv],
        name="dw_cnv_graph",
        inputs=[inp],
        outputs=[outp],
        value_info=[W],
    )

    model = helper.make_model(graph, producer_name="dws_cnv-model")
    model = ModelWrapper(model)
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("W", wdt)
    w_tensor = gen_finn_dt_tensor(wdt, [ofm_ch, 1, k, k])
    model.set_initializer("W", w_tensor)
    model = model.transform(InferShapes())

    input_tensor = gen_finn_dt_tensor(idt, [1, ifm_ch, ifm_dim, ifm_dim])
    input_dict = {"inp": input_tensor}
    output_dict = oxe.execute_onnx(model, input_dict)
    expected = output_dict["outp"]

    model = model.transform(LowerConvsToMatMul())
    output_dict = oxe.execute_onnx(model, input_dict)
    produced = output_dict["outp"]
    assert (produced == expected).all()

    # check if created nodes have attributes that indicate depthwise conv
    assert model.get_tensor_sparsity("W") is not None
    im2col_node = getCustomOp(model.graph.node[1])
    assert im2col_node.get_nodeattr("dw") == 1


def test_conv_lowering_conv_1x1():
    np.random.seed(0)

    in_feature_dim = 7
    in_chn = 3
    kernel_size = 1
    out_feature_dim = in_feature_dim

    input_shape = [1, in_chn, in_feature_dim, in_feature_dim]
    output_shape = [1, in_chn, out_feature_dim, out_feature_dim]

    conv_param_shape = [in_chn, in_chn, kernel_size, kernel_size]

    conv_config = {}
    conv_config["dilations"] = [1, 1]
    conv_config["group"] = 1
    conv_config["kernel_shape"] = [kernel_size, kernel_size]
    conv_config["pads"] = [0, 0, 0, 0]
    conv_config["strides"] = [1, 1]

    top_in = helper.make_tensor_value_info("top_in", TensorProto.FLOAT, input_shape)
    top_out = helper.make_tensor_value_info("top_out", TensorProto.FLOAT, output_shape)

    value_info = [
        helper.make_tensor_value_info("p1", TensorProto.FLOAT, conv_param_shape)
    ]

    modelproto = helper.make_model(
        helper.make_graph(
            name="test",
            inputs=[top_in],
            outputs=[top_out],
            value_info=value_info,
            nodes=[
                helper.make_node("Conv", ["top_in", "p1"], ["top_out"], **conv_config)
            ],
        )
    )
    model = ModelWrapper(modelproto)
    model = model.transform(InferShapes())
    model.set_initializer("p1", np.random.rand(*conv_param_shape).astype(np.float32))

    new_model = model.transform(LowerConvsToMatMul())
    inp_dict = {"top_in": np.random.rand(*input_shape).astype(np.float32)}

    assert oxe.compare_execution(model, new_model, inp_dict)
    assert new_model.graph.node[0].op_type == "Transpose"
    assert new_model.graph.node[1].op_type == "MatMul"
    assert new_model.graph.node[2].op_type == "Transpose"
    assert len(new_model.graph.node) == 3
