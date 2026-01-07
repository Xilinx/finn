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
import onnx  # noqa
import os
import torch
from brevitas.core.scaling import ScalingImplType
from brevitas.export import export_qonnx
from brevitas.nn import QuantReLU
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.cleanup import cleanup as qonnx_cleanup

import finn.core.onnx_exec as oxe
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.util.basic import make_build_dir


@pytest.mark.brevitas_export
@pytest.mark.parametrize("abits", [2, 4, 8])
@pytest.mark.parametrize("ishape", [(1, 15), (1, 32, 1, 1)])
def test_brevitas_act_export_relu(
    abits,
    ishape,
):
    b_act = QuantReLU(
        bit_width=abits,
    )
    build_dir = make_build_dir(prefix="test_brevitas_act_export_relu")
    m_path = os.path.join(build_dir, "test_brevitas_relu_act_export.onnx")
    export_qonnx(b_act, torch.randn(ishape), m_path)
    qonnx_cleanup(m_path, out_file=m_path)
    model = ModelWrapper(m_path)
    model = model.transform(ConvertQONNXtoFINN())
    model = model.transform(InferShapes())
    inp_tensor = np.random.uniform(low=-1.0, high=6.0, size=ishape).astype(np.float32)
    idict = {model.graph.input[0].name: inp_tensor}
    odict = oxe.execute_onnx(model, idict, True)
    produced = odict[model.graph.output[0].name]
    inp_tensor = torch.from_numpy(inp_tensor).float()
    b_act.eval()
    expected = b_act.forward(inp_tensor).detach().numpy()

    assert np.isclose(produced, expected, atol=1e-3).all()


@pytest.mark.brevitas_export
@pytest.mark.parametrize("abits", [2, 4, 8])
@pytest.mark.parametrize("ishape", [(1, 15, 4, 4), (1, 32, 1, 1)])
def test_brevitas_act_export_relu_channel(
    abits,
    ishape,
):
    ch = ishape[1]
    b_act = QuantReLU(
        bit_width=abits,
        max_val=6.0,
        scaling_impl_type=ScalingImplType.CONST,
        scaling_per_output_channel=True,
        per_channel_broadcastable_shape=(1, ch, 1, 1),
    )
    build_dir = make_build_dir(prefix="test_brevitas_act_export_relu_channel")
    m_path = os.path.join(build_dir, "test_brevitas_relu_act_export.onnx")
    export_qonnx(b_act, torch.randn(ishape), m_path)
    qonnx_cleanup(m_path, out_file=m_path)
    model = ModelWrapper(m_path)
    model = model.transform(ConvertQONNXtoFINN())
    model = model.transform(InferShapes())
    inp_tensor = np.random.uniform(low=-1.0, high=6.0, size=ishape).astype(np.float32)
    idict = {model.graph.input[0].name: inp_tensor}
    odict = oxe.execute_onnx(model, idict, True)
    produced = odict[model.graph.output[0].name]
    inp_tensor = torch.from_numpy(inp_tensor).float()
    b_act.eval()
    expected = b_act.forward(inp_tensor).detach().numpy()

    assert np.isclose(produced, expected, atol=1e-3).all()
