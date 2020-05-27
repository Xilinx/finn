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


# @pytest.mark.parametrize
def test_depthwise_conv_lowering():
    idt = odt = wdt = DataType.INT4
    k = 3
    stride = 1
    ifm_dim = 4
    ifm_ch = 2
    ofm_ch = ifm_ch
    ofm_dim = compute_conv_output_dim(ifm_dim, k, stride)

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
        pads=[0, 0, 0, 0],
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

    input_tensor = gen_finn_dt_tensor(idt, [1, ifm_ch, ifm_dim, ifm_dim])
    input_dict = {"inp": input_tensor}
    output_dict = oxe.execute_onnx(model, input_dict)
    expected = output_dict["outp"]

    model = model.transform(LowerConvsToMatMul())
    output_dict = oxe.execute_onnx(model, input_dict)
    produced = output_dict["outp"]
    assert (produced == expected).all()
