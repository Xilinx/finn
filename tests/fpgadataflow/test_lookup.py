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

import onnx  # noqa
import os
import torch
from brevitas.export import FINNManager
from torch import nn

from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_shapes import InferShapes
from finn.util.basic import gen_finn_dt_tensor

tmpdir = os.environ["FINN_BUILD_DIR"]


def make_lookup_model(num_embeddings, embedding_dim, ishape, idt, edt):
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
    new_embs = gen_finn_dt_tensor(edt, eshape)
    model.set_initializer(ename, new_embs)
    model.set_tensor_datatype(ename, edt)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    return model


def test_lookup_export():
    export_path = tmpdir + "/test_lookup_export.onnx"
    ishape = (1, 10)
    num_embeddings = 8
    embedding_dim = 2
    exp_oshape = tuple(list(ishape) + [embedding_dim])
    idt = DataType["UINT8"]
    edt = DataType["FIXED<8,2>"]
    model = make_lookup_model(num_embeddings, embedding_dim, ishape, idt, edt)
    model.save(export_path)
    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == "Gather"
    iname = model.graph.input[0].name
    ename = model.graph.node[0].input[0]
    oname = model.graph.output[0].name
    assert model.get_tensor_datatype(iname) == idt
    assert model.get_tensor_datatype(ename) == edt
    assert model.get_tensor_datatype(oname) == edt
    assert tuple(model.get_tensor_shape(ename)) == (num_embeddings, embedding_dim)
    assert tuple(model.get_tensor_shape(oname)) == exp_oshape
