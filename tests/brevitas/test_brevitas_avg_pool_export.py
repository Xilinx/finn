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

import onnx  # noqa
import torch
import numpy as np
import brevitas.onnx as bo
from brevitas.nn import QuantAvgPool2d
from brevitas.quant_tensor import QuantTensor
from brevitas.core.quant import QuantType
from finn.core.modelwrapper import ModelWrapper
from finn.core.datatype import DataType
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.infer_datatypes import InferDataTypes
from finn.util.basic import gen_finn_dt_tensor
import finn.core.onnx_exec as oxe

import pytest

export_onnx_path = "test_brevitas_avg_pool_export.onnx"


@pytest.mark.parametrize("kernel_size", [2, 3])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("bit_width", [2, 4])
@pytest.mark.parametrize("input_bit_width", [4, 8, 16])
@pytest.mark.parametrize("channels", [2, 4])
@pytest.mark.parametrize("idim", [7, 8])
def test_brevitas_avg_pool_export(
    kernel_size, stride, signed, bit_width, input_bit_width, channels, idim
):
    ishape = (1, channels, idim, idim)
    ibw_tensor = torch.Tensor([input_bit_width])

    b_avgpool = QuantAvgPool2d(
        kernel_size=kernel_size,
        stride=stride,
        bit_width=bit_width,
        quant_type=QuantType.INT,
    )
    # call forward pass manually once to cache scale factor and bitwidth
    input_tensor = torch.from_numpy(np.zeros(ishape)).float()
    scale = np.ones((1, channels, 1, 1))
    zpt = torch.from_numpy(np.zeros((1))).float()
    output_scale = torch.from_numpy(scale).float()
    input_quant_tensor = QuantTensor(
        value=input_tensor,
        scale=output_scale,
        bit_width=ibw_tensor,
        signed=signed,
        zero_point=zpt,
    )
    bo.export_finn_onnx(
        b_avgpool, export_path=export_onnx_path, input_t=input_quant_tensor
    )
    model = ModelWrapper(export_onnx_path)

    # determine input FINN datatype
    if signed is True:
        prefix = "INT"
    else:
        prefix = "UINT"
    dt_name = prefix + str(input_bit_width)
    dtype = DataType[dt_name]
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    # execution with input tensor using integers and scale = 1
    # calculate golden output
    inp = gen_finn_dt_tensor(dtype, ishape)
    input_tensor = torch.from_numpy(inp).float()
    input_quant_tensor = QuantTensor(
        value=input_tensor,
        scale=output_scale,
        bit_width=ibw_tensor,
        signed=signed,
        zero_point=zpt,
    )
    b_avgpool.eval()
    expected = b_avgpool.forward(input_quant_tensor).tensor.detach().numpy()

    # finn execution
    idict = {model.graph.input[0].name: inp}
    odict = oxe.execute_onnx(model, idict, True)
    produced = odict[model.graph.output[0].name]
    assert (expected == produced).all()

    # execution with input tensor using float and scale != 1
    scale = np.random.uniform(low=0, high=1, size=(1, channels, 1, 1)).astype(
        np.float32
    )
    inp_tensor = inp * scale
    input_tensor = torch.from_numpy(inp_tensor).float()
    input_scale = torch.from_numpy(scale).float()
    input_quant_tensor = QuantTensor(
        value=input_tensor,
        scale=input_scale,
        bit_width=ibw_tensor,
        signed=signed,
        zero_point=zpt,
    )
    # export again to set the scale values correctly
    bo.export_finn_onnx(
        b_avgpool, export_path=export_onnx_path, input_t=input_quant_tensor
    )
    model = ModelWrapper(export_onnx_path)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    b_avgpool.eval()
    expected = b_avgpool.forward(input_quant_tensor).tensor.detach().numpy()
    # finn execution
    idict = {model.graph.input[0].name: inp_tensor}
    odict = oxe.execute_onnx(model, idict, True)
    produced = odict[model.graph.output[0].name]

    assert np.isclose(expected, produced).all()

    os.remove(export_onnx_path)
