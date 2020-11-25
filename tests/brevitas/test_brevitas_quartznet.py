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

import brevitas.onnx as bo
import numpy as np
import torch
import pytest

import finn.core.onnx_exec as oxe
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.general import RemoveStaticGraphInputs
from finn.transformation.infer_shapes import InferShapes
from finn.util.basic import make_build_dir, gen_finn_dt_tensor
from finn.core.datatype import DataType
import brevitas_examples.speech_to_text as stt

export_onnx_path = make_build_dir("test_brevitas_quartznet_")


@pytest.mark.slow
def test_brevitas_quartznet_onnx_export_and_exec():
    nname = "quartznet-4b"
    finn_onnx = export_onnx_path + "/%s.onnx" % nname
    quartznet_torch = stt.quant_quartznet_perchannelscaling_4b(export_mode=True)
    ishape = (1, 64, 256)
    idt = DataType.FLOAT32
    bo.export_finn_onnx(quartznet_torch, ishape, finn_onnx)
    model = ModelWrapper(finn_onnx)
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(RemoveStaticGraphInputs())
    assert len(model.graph.input) == 1
    assert len(model.graph.output) == 1
    # generate a random test vector
    iname = model.graph.input[0].name
    oname = model.graph.output[0].name
    np.random.seed(42)
    rand_inp = gen_finn_dt_tensor(idt, ishape)
    # run using FINN-based execution
    input_dict = {iname: rand_inp}
    output_dict = oxe.execute_onnx(model, input_dict)
    produced = output_dict[oname]
    # run using PyTorch/Brevitas
    rand_inp_torch = torch.from_numpy(rand_inp).float()
    # do forward pass in PyTorch/Brevitas
    expected = quartznet_torch.forward(rand_inp_torch).detach().numpy()
    assert np.isclose(produced, expected, atol=1e-3).all()
