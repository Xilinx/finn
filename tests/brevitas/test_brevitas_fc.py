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
import onnx
import onnx.numpy_helper as nph
import torch
from brevitas.export.onnx.generic.manager import BrevitasONNXManager
from pkgutil import get_data
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import RemoveStaticGraphInputs
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.cleanup import cleanup as qonnx_cleanup

import finn.core.onnx_exec as oxe
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.util.basic import make_build_dir
from finn.util.test import get_test_model_trained

export_onnx_path = make_build_dir("test_brevitas_fc_")


@pytest.mark.brevitas_export
# act bits
@pytest.mark.parametrize("abits", [1, 2])
# weight bits
@pytest.mark.parametrize("wbits", [1, 2])
# network topology / size
@pytest.mark.parametrize("size", ["TFC", "SFC", "LFC"])
# QONNX export
@pytest.mark.parametrize("QONNX_export", [False, True])
def test_brevitas_fc_onnx_export_and_exec(size, wbits, abits, QONNX_export):
    if size == "LFC" and wbits == 2 and abits == 2:
        pytest.skip("No LFC-w2a2 present at the moment")
    if wbits > abits:
        pytest.skip("No wbits > abits cases at the moment")
    nname = "%s_%dW%dA_QONNX-%d" % (size, wbits, abits, QONNX_export)
    finn_onnx = export_onnx_path + "/%s.onnx" % nname
    fc = get_test_model_trained(size, wbits, abits)
    ishape = (1, 1, 28, 28)
    if QONNX_export:
        BrevitasONNXManager.export(fc, ishape, finn_onnx)
        qonnx_cleanup(finn_onnx, out_file=finn_onnx)
        model = ModelWrapper(finn_onnx)
        model = model.transform(ConvertQONNXtoFINN())
        model.save(finn_onnx)
    else:
        bo.export_finn_onnx(fc, ishape, finn_onnx)
    model = ModelWrapper(finn_onnx)
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(RemoveStaticGraphInputs())
    assert len(model.graph.input) == 1
    assert len(model.graph.output) == 1
    # load one of the test vectors
    raw_i = get_data("qonnx.data", "onnx/mnist-conv/test_data_set_0/input_0.pb")
    input_tensor = onnx.load_tensor_from_string(raw_i)
    # run using FINN-based execution
    input_dict = {model.graph.input[0].name: nph.to_array(input_tensor)}
    output_dict = oxe.execute_onnx(model, input_dict)
    produced = output_dict[list(output_dict.keys())[0]]
    # run using PyTorch/Brevitas
    input_tensor = torch.from_numpy(nph.to_array(input_tensor)).float()
    assert input_tensor.shape == (1, 1, 28, 28)
    # do forward pass in PyTorch/Brevitas
    expected = fc.forward(input_tensor).detach().numpy()
    assert np.isclose(produced, expected, atol=1e-3).all()
