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
import os
import torch
from brevitas.export import FINNManager
from brevitas.export.onnx.generic.manager import BrevitasONNXManager
from brevitas.nn import QuantAvgPool2d, QuantIdentity, QuantReLU
from qonnx.util.cleanup import cleanup as qonnx_cleanup
from torch import nn

import finn.core.onnx_exec as oxe
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.util.basic import gen_finn_dt_tensor

base_export_onnx_path = "test_brevitas_avg_pool_export.onnx"


@pytest.mark.parametrize("QONNX_export", [False, True])
@pytest.mark.parametrize("signed", [True, False])
@pytest.mark.parametrize("kernel_size", [2, 3])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("bit_width", [2, 4])
@pytest.mark.parametrize("input_bit_width", [4, 8, 16])
@pytest.mark.parametrize("channels", [2, 4])
@pytest.mark.parametrize("idim", [7, 8])
def test_brevitas_avg_pool_export(
    kernel_size,
    stride,
    signed,
    bit_width,
    input_bit_width,
    channels,
    idim,
    QONNX_export,
):
    export_onnx_path = base_export_onnx_path.replace(
        ".onnx", f"test_QONNX-{QONNX_export}.onnx"
    )
    # To do a proper static export Brevitas requires a quantized input tensor.
    # For the BrevitasONNXManager these requirements are even more stringent,
    # such that in-model quantization and de-quantization at the end are required.
    if signed:
        # Signed is only supported by the QuantIdentity in FINN
        act_layer = QuantIdentity(
            signed=signed, bit_width=input_bit_width, return_quant_tensor=True
        )
    else:
        # Unsigned is only supported by the QuantReLU in FINN
        act_layer = QuantReLU(
            signed=signed, bit_width=input_bit_width, return_quant_tensor=True
        )
    quant_avgpool = nn.Sequential(
        act_layer,
        QuantAvgPool2d(
            kernel_size=kernel_size,
            stride=stride,
            bit_width=bit_width,
            return_quant_tensor=False,
        ),
    )
    quant_avgpool.eval()

    # determine input
    prefix = "INT" if signed else "UINT"
    dt_name = prefix + str(input_bit_width)
    dtype = DataType[dt_name]
    input_shape = (1, channels, idim, idim)
    input_array = gen_finn_dt_tensor(dtype, input_shape)
    input_tensor = torch.from_numpy(input_array).float()

    # export
    if QONNX_export:
        BrevitasONNXManager.export(
            quant_avgpool,
            input_shape,
            export_path=export_onnx_path,
        )
        qonnx_cleanup(export_onnx_path, out_file=export_onnx_path)
        model = ModelWrapper(export_onnx_path)
        model = model.transform(ConvertQONNXtoFINN())
        model.save(export_onnx_path)
    else:
        FINNManager.export(quant_avgpool, input_shape, export_path=export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    # reference brevitas output
    ref_output_array = quant_avgpool(input_tensor).detach().numpy()
    # finn output
    idict = {model.graph.input[0].name: input_tensor.detach().numpy()}
    odict = oxe.execute_onnx(model, idict, True)
    finn_output = odict[model.graph.output[0].name]
    # compare outputs
    assert np.isclose(ref_output_array, finn_output).all()
    # cleanup
    os.remove(export_onnx_path)
