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
from brevitas.export import export_qonnx
from brevitas.nn import QuantIdentity, QuantReLU, TruncAvgPool2d
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor
from qonnx.util.cleanup import cleanup as qonnx_cleanup

import finn.core.onnx_exec as oxe
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.util.basic import make_build_dir


@pytest.mark.brevitas_export
@pytest.mark.parametrize("kernel_size", [2, 3])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("signed", [True])  # TODO: Add unsigned test case
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
):
    build_dir = make_build_dir(prefix="test_brevitas_avg_pool_export")
    export_onnx_path = os.path.join(build_dir, "test.onnx")
    if signed:
        quant_node = QuantIdentity(
            bit_width=input_bit_width,
            return_quant_tensor=True,
        )
    else:
        quant_node = QuantReLU(
            bit_width=input_bit_width,
            return_quant_tensor=True,
        )
    quant_avgpool = TruncAvgPool2d(
        kernel_size=kernel_size,
        stride=stride,
        bit_width=bit_width,
        return_quant_tensor=False,
        float_to_int_impl_type="FLOOR",
    )
    model_brevitas = torch.nn.Sequential(quant_node, quant_avgpool)
    model_brevitas.eval()

    # determine input
    input_shape = (1, channels, idim, idim)
    input_array = gen_finn_dt_tensor(DataType["FLOAT32"], input_shape)

    input_tensor = torch.from_numpy(input_array).float()

    # export
    export_qonnx(
        model_brevitas,
        export_path=export_onnx_path,
        input_t=input_tensor,
    )
    model = ModelWrapper(export_onnx_path)
    model.save(export_onnx_path)

    qonnx_cleanup(export_onnx_path, out_file=export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    model = model.transform(ConvertQONNXtoFINN())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    # reference brevitas output
    ref_output_array = model_brevitas(input_tensor).detach().numpy()
    # finn output
    idict = {model.graph.input[0].name: input_array}
    odict = oxe.execute_onnx(model, idict, True)
    finn_output = odict[model.graph.output[0].name]
    # compare outputs
    assert np.isclose(ref_output_array, finn_output).all()
