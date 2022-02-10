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

import numpy as np
import torch
from brevitas.export import FINNManager
from torch import nn

from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.core.onnx_exec import execute_onnx
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.convert_to_hls_layers import InferLookupLayer
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.general import GiveUniqueNodeNames
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_shapes import InferShapes
from finn.util.basic import gen_finn_dt_tensor


def make_lookup_model(embeddings, ishape, idt, edt):
    num_embeddings, embedding_dim = embeddings.shape

    class LookupModel(nn.Module):
        def __init__(self, num_embeddings, embedding_dim):
            super().__init__()
            self.lookup = nn.Embedding(
                num_embeddings=num_embeddings, embedding_dim=embedding_dim
            )

        def forward(self, x):
            x = self.lookup(x)
            return x

    torch_model = LookupModel(num_embeddings, embedding_dim)
    input_t = torch.zeros(ishape, dtype=torch.int64)
    ret = FINNManager.export(torch_model, input_t=input_t, opset_version=11)
    model = ModelWrapper(ret)
    iname = model.graph.input[0].name
    ename = model.graph.node[0].input[0]
    model.set_tensor_datatype(iname, idt)
    eshape = model.get_tensor_shape(ename)
    assert tuple(eshape) == embeddings.shape
    model.set_initializer(ename, embeddings)
    model.set_tensor_datatype(ename, edt)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    return model


# embedding DataType
@pytest.mark.parametrize("edt", [DataType["FIXED<8,2>"]])
# other embedding config
@pytest.mark.parametrize(
    "embedding_cfg", [(130, DataType["UINT8"], 25), (5145, DataType["UINT16"], 20)]
)
# execution mode
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.vivado
@pytest.mark.slow
def test_fpgadataflow_lookup(edt, embedding_cfg, exec_mode):
    ishape = (1, 10)
    num_embeddings, idt, embedding_dim = embedding_cfg
    eshape = (num_embeddings, embedding_dim)
    exp_oshape = tuple(list(ishape) + [embedding_dim])
    embeddings = gen_finn_dt_tensor(edt, eshape)
    model = make_lookup_model(embeddings, ishape, idt, edt)
    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == "Gather"
    iname = model.graph.input[0].name
    ename = model.graph.node[0].input[0]
    oname = model.graph.output[0].name
    assert model.get_tensor_datatype(iname) == idt
    assert model.get_tensor_datatype(ename) == edt
    assert model.get_tensor_datatype(oname) == edt
    assert tuple(model.get_tensor_shape(ename)) == eshape
    assert tuple(model.get_tensor_shape(oname)) == exp_oshape
    assert (model.get_initializer(ename) == embeddings).all()
    itensor = gen_finn_dt_tensor(idt, ishape).astype(np.int64)
    itensor = np.clip(itensor, 0, num_embeddings - 1)
    ret = execute_onnx(model, {iname: itensor})
    exp_out = np.take(embeddings, itensor, axis=0)
    assert (exp_out == ret[oname]).all()
    # call transformation to convert to HLS and verify conversion
    model = model.transform(InferLookupLayer())
    assert model.graph.node[0].op_type == "Lookup"
    assert model.graph.node[0].input[0] == iname
    assert model.graph.node[0].input[1] == ename
    assert model.graph.node[0].output[0] == oname
    if exec_mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))
    elif exec_mode == "rtlsim":
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareIP("xczu3eg-sbva484-1-e", 10))
        model = model.transform(HLSSynthIP())
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(PrepareRTLSim())
    ret_sim = execute_onnx(model, {iname: itensor})
    assert (exp_out == ret_sim[oname]).all()
