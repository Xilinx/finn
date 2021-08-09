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

import brevitas.onnx as bo
import numpy as np
import os
import torch
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.stats import StatsOp
from brevitas.nn import QuantConv2d

import finn.core.onnx_exec as oxe
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.infer_shapes import InferShapes
from finn.util.basic import gen_finn_dt_tensor

export_onnx_path = "test_brevitas_conv.onnx"


@pytest.mark.parametrize("dw", [False, True])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("in_channels", [32])
def test_brevitas_QConv2d(dw, bias, in_channels):
    ishape = (1, 32, 111, 111)
    if dw is True:
        groups = in_channels
        out_channels = in_channels
        kernel_size = 3
        padding = 1
        stride = 1
        w_shape = (32, 1, 3, 3)

    else:
        groups = 1
        out_channels = 64
        kernel_size = 1
        padding = 0
        stride = 1
        w_shape = (64, 32, 1, 1)

    b_conv = QuantConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        groups=groups,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
        bias=bias,
        bias_quant_type=QuantType.FP,
        weight_bit_width=4,
        weight_quant_type=QuantType.INT,
        weight_scaling_impl_type=ScalingImplType.STATS,
        weight_scaling_stats_op=StatsOp.MAX,
        weight_scaling_per_output_channel=True,
        weight_restrict_scaling_type=RestrictValueType.LOG_FP,
        weight_narrow_range=True,
        weight_scaling_min_val=2e-16,
    )
    weight_tensor = gen_finn_dt_tensor(DataType.INT4, w_shape)
    b_conv.weight = torch.nn.Parameter(torch.from_numpy(weight_tensor).float())
    b_conv.eval()
    bo.export_finn_onnx(b_conv, ishape, export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    model = model.transform(InferShapes())
    inp_tensor = np.random.uniform(low=-1.0, high=1.0, size=ishape).astype(np.float32)
    idict = {model.graph.input[0].name: inp_tensor}
    odict = oxe.execute_onnx(model, idict, True)
    produced = odict[model.graph.output[0].name]
    inp_tensor = torch.from_numpy(inp_tensor).float()
    expected = b_conv.forward(inp_tensor).detach().numpy()

    assert np.isclose(produced, expected, atol=1e-3).all()
    os.remove(export_onnx_path)
