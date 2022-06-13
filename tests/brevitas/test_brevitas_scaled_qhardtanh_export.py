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
import onnx  # noqa
import os
import torch
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType
from brevitas.export.onnx.generic.manager import BrevitasONNXManager
from brevitas.nn import QuantHardTanh
from qonnx.util.cleanup import cleanup as qonnx_cleanup

import finn.core.onnx_exec as oxe
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN

export_onnx_path = "test_brevitas_scaled_QHardTanh_export.onnx"

@pytest.mark.brevitas_export
@pytest.mark.parametrize("abits", [2, 4, 8])
@pytest.mark.parametrize("narrow_range", [False, True])
@pytest.mark.parametrize("min_val", [-1.0, -(1 - 2 ** (-7)), -2])
@pytest.mark.parametrize("max_val", [1.0, 1 - 2 ** (-7), 2])
@pytest.mark.parametrize(
    "scaling_impl_type", [ScalingImplType.CONST, ScalingImplType.PARAMETER]
)
@pytest.mark.parametrize("QONNX_export", [False, True])
def test_brevitas_act_export_qhardtanh_scaled(
    abits, narrow_range, min_val, max_val, scaling_impl_type, QONNX_export
):
    def get_quant_type(bit_width):
        if bit_width is None:
            return QuantType.FP
        elif bit_width == 1:
            return QuantType.BINARY
        else:
            return QuantType.INT

    act_quant_type = get_quant_type(abits)
    ishape = (1, 15)
    b_act = QuantHardTanh(
        bit_width=abits,
        quant_type=act_quant_type,
        max_val=max_val,
        min_val=min_val,
        restrict_scaling_type=RestrictValueType.LOG_FP,
        scaling_impl_type=scaling_impl_type,
        narrow_range=narrow_range,
    )
    if scaling_impl_type == ScalingImplType.PARAMETER:
        checkpoint = {
            "act_quant_proxy.fused_activation_quant_proxy.\
tensor_quant.scaling_impl.learned_value": torch.tensor(
                0.49
            ).type(
                torch.FloatTensor
            )
        }
        b_act.load_state_dict(checkpoint)
    if QONNX_export:
        m_path = export_onnx_path
        BrevitasONNXManager.export(b_act, ishape, m_path)
        qonnx_cleanup(m_path, out_file=m_path)
        model = ModelWrapper(m_path)
        model = model.transform(ConvertQONNXtoFINN())
        model.save(m_path)
    else:
        bo.export_finn_onnx(b_act, ishape, export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    model = model.transform(InferShapes())
    inp_tensor = np.random.uniform(low=min_val, high=max_val, size=ishape).astype(
        np.float32
    )
    idict = {model.graph.input[0].name: inp_tensor}
    odict = oxe.execute_onnx(model, idict, True)
    produced = odict[model.graph.output[0].name]
    inp_tensor = torch.from_numpy(inp_tensor).float()
    b_act.eval()
    expected = b_act.forward(inp_tensor).detach().numpy()
    if not np.isclose(produced, expected, atol=1e-3).all():
        print(
            "abits: ",
            abits,
            " | narrow_range: ",
            narrow_range,
            " | min_val: ",
            min_val,
            " | max_val: ",
            max_val,
        )
        print("layer scale: ", b_act.quant_act_scale().type(torch.FloatTensor).detach())
        print("export scale: ", b_act.export_act_scale)
        if abits < 5:
            print(
                "thres:",
                ", ".join(["{:8.4f}".format(x) for x in b_act.export_thres[0]]),
            )
        print("input:", ", ".join(["{:8.4f}".format(x) for x in inp_tensor[0]]))
        print("prod :", ", ".join(["{:8.4f}".format(x) for x in produced[0]]))
        print("expec:", ", ".join(["{:8.4f}".format(x) for x in expected[0]]))

    assert np.isclose(produced, expected, atol=1e-3).all()
    os.remove(export_onnx_path)
