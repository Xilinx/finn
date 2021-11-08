# Copyright (c) 2021, Xilinx
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

import brevitas.onnx as bo
import numpy as np
import os
import torch
from brevitas.core.quant import QuantType
from brevitas.export.onnx.generic.manager import BrevitasONNXManager
from brevitas.nn import QuantLinear
from qonnx.util.cleanup import cleanup as qonnx_cleanup

import finn.core.onnx_exec as oxe
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.util.basic import gen_finn_dt_tensor

export_onnx_path = "test_brevitas_qlinear.onnx"


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("out_features", [4])
@pytest.mark.parametrize("in_features", [3])
@pytest.mark.parametrize("w_bits", [4])
@pytest.mark.parametrize("i_dtype", [DataType["UINT4"]])
@pytest.mark.parametrize("QONNX_export", [False, True])
def test_brevitas_qlinear(
    bias, out_features, in_features, w_bits, i_dtype, QONNX_export
):
    i_shape = (1, in_features)
    w_shape = (out_features, in_features)
    b_linear = QuantLinear(
        out_features=out_features,
        in_features=in_features,
        bias=bias,
        bias_quant_type=QuantType.FP,
        weight_bit_width=w_bits,
        weight_quant_type=QuantType.INT,
        weight_scaling_per_output_channel=True,
    )
    weight_tensor_fp = np.random.uniform(low=-1.0, high=1.0, size=w_shape).astype(
        np.float32
    )
    b_linear.weight.data = torch.from_numpy(weight_tensor_fp)
    b_linear.eval()
    if QONNX_export:
        m_path = export_onnx_path
        BrevitasONNXManager.export(b_linear, i_shape, m_path)
        qonnx_cleanup(m_path, out_file=m_path)
        model = ModelWrapper(m_path)
        model = model.transform(ConvertQONNXtoFINN())
        model.save(m_path)
    else:
        bo.export_finn_onnx(b_linear, i_shape, export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    model = model.transform(InferShapes())
    inp_tensor = gen_finn_dt_tensor(i_dtype, i_shape)
    idict = {model.graph.input[0].name: inp_tensor}
    odict = oxe.execute_onnx(model, idict, True)
    produced = odict[model.graph.output[0].name]
    inp_tensor = torch.from_numpy(inp_tensor).float()
    expected = b_linear.forward(inp_tensor).detach().numpy()

    assert np.isclose(produced, expected, atol=1e-3).all()
    os.remove(export_onnx_path)
