# Copyright (c) 2023, Advanced Micro Devices, Inc.
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
# * Neither the name of Xilinx nor the names of its
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

import brevitas.nn as qnn
import numpy as np
import os
import torch
from brevitas.export import export_qonnx
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.cleanup import cleanup as qonnx_cleanup

import finn.core.onnx_exec as oxe
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.util.basic import make_build_dir


@pytest.mark.brevitas_export
@pytest.mark.parametrize("ifm_ch", [3])
@pytest.mark.parametrize("ofm_ch", [5])
@pytest.mark.parametrize("mh", [4])
@pytest.mark.parametrize("mw", [4])
@pytest.mark.parametrize("padding", [1])
@pytest.mark.parametrize("stride", [2])
@pytest.mark.parametrize("kw", [4])
@pytest.mark.parametrize("bias", [False])
def test_brevitas_QTransposeConv(ifm_ch, ofm_ch, mh, mw, padding, stride, kw, bias):
    kh = kw
    oh = stride * (mh - 1) - (2 * padding) + kh
    if oh % mh != 0:
        pytest.skip("Skip test because oh needs to be divisible by mh")
    ishape = (1, ifm_ch, mh, mw)  # NCHW
    inp = torch.randn(ishape)
    b_deconv = qnn.QuantConvTranspose2d(
        in_channels=ifm_ch,
        out_channels=ofm_ch,
        kernel_size=kw,
        stride=stride,
        padding=padding,
        bias=bias,
    )
    build_dir = make_build_dir("test_brevitas_QTransposeConv")
    export_path = os.path.join(build_dir, "test_brevitas_deconv.onnx")
    # outp = el(inp) # expects NCHW data format
    export_qonnx(b_deconv, input_t=inp, export_path=export_path, opset_version=11)
    qonnx_cleanup(export_path, out_file=export_path)
    model = ModelWrapper(export_path)
    model = model.transform(ConvertQONNXtoFINN())
    model = model.transform(InferShapes())
    inp_tensor = np.random.uniform(low=-1.0, high=1.0, size=ishape).astype(np.float32)
    idict = {model.graph.input[0].name: inp_tensor}
    odict = oxe.execute_onnx(model, idict, True)
    produced = odict[model.graph.output[0].name]
    inp_tensor = torch.from_numpy(inp_tensor).float()
    expected = b_deconv.forward(inp_tensor).detach().numpy()
    assert np.isclose(produced, expected, atol=1e-3).all()
