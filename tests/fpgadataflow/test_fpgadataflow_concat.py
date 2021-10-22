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

import numpy as np
import onnx
import torch
from io import BytesIO
from torch import nn

from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.core.onnx_exec import execute_onnx
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.convert_to_hls_layers import InferConcatLayer
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.general import GiveUniqueNodeNames
from finn.util.basic import gen_finn_dt_tensor


def make_concat_model(i_shapes, idt):
    class ConcatModel(nn.Module):
        def forward(self, *args):
            return torch.cat(args, -1)

    torch_model = ConcatModel()
    torch_model.eval()
    input_t = []
    for i_shape in i_shapes:
        input_t.append(torch.zeros(i_shape, dtype=torch.float32))
    input_t = tuple(input_t)
    model_bytes = BytesIO()
    torch.onnx.export(torch_model, input_t, model_bytes, opset_version=11)
    model = onnx.ModelProto.FromString(model_bytes.getvalue())
    model = ModelWrapper(model)
    for inp in model.graph.input:
        model.set_tensor_datatype(inp.name, idt)
    return model


def test_fpgadataflow_concat():
    exec_mode = "cppsim"
    i_shapes = [(1, 2, 4), (1, 2, 6)]
    idt = DataType["INT4"]
    i_data = [gen_finn_dt_tensor(idt, x) for x in i_shapes]
    model = make_concat_model(i_shapes, idt)
    assert len(i_shapes) == len(model.graph.input)
    assert len(model.graph.output) == 1
    exp_oshape = list(i_shapes[0][:-1]) + [sum(x[-1] for x in i_shapes)]
    oname = model.graph.output[0].name
    assert model.get_tensor_shape(oname) == exp_oshape
    exp_out = np.concatenate(i_data, axis=-1)
    inp_dict = {}
    for i in range(len(i_shapes)):
        print("inp %d : %s" % (i, str(i_data[i])))
        inp_dict[model.graph.input[i].name] = i_data[i]
    ret = execute_onnx(model, inp_dict)
    assert (ret[oname] == exp_out).all()
    # call transformation to convert to HLS and verify conversion
    model = model.transform(InferConcatLayer())
    assert model.graph.node[0].op_type == "Concat"
    assert model.graph.node[0].domain == "finn.custom_op.fpgadataflow"
    model.save("/tmp/finn_dev_maltanar/dbg.onnx")
    if exec_mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))
    elif exec_mode == "rtlsim":
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareIP("xc7z020clg400-1", 10))
        model = model.transform(HLSSynthIP())
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(PrepareRTLSim())
    ret_sim = execute_onnx(model, inp_dict)
    assert (exp_out == ret_sim[oname]).all()
