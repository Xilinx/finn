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
from onnx import TensorProto, helper

import finn.core.onnx_exec as oxe
from finn.analysis.fpgadataflow.hls_synth_res_estimation import hls_synth_res_estimation
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.general import GiveUniqueNodeNames
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.util.basic import gen_finn_dt_tensor
from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)


def make_modelwrapper(C, pe, idt, odt, func, vecs):
    NumChannels = C.shape[0]

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, vecs + [NumChannels])
    outp = helper.make_tensor_value_info(
        "outp", TensorProto.FLOAT, vecs + [NumChannels]
    )

    node_inp_list = ["inp", "const"]

    node = helper.make_node(
        "ChannelwiseOp_Batch",
        node_inp_list,
        ["outp"],
        domain="finn",
        backend="fpgadataflow",
        NumChannels=NumChannels,
        Func=func,
        PE=pe,
        inputDataType=idt.name,
        outputDataType=odt.name,
        numInputVectors=vecs,
    )
    graph = helper.make_graph(nodes=[node], name="graph", inputs=[inp], outputs=[outp])

    model = helper.make_model(graph, producer_name="model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)

    model.set_tensor_datatype("const", idt)
    model.set_initializer("const", C)
    return model


# activation: None or DataType
@pytest.mark.parametrize("act", [DataType.INT8])
# input datatype
@pytest.mark.parametrize("idt", [DataType.INT4])
# folding, -1 is maximum possible
@pytest.mark.parametrize("nf", [-1, 2])
# number of input features
@pytest.mark.parametrize("ich", [16])
# vecs
@pytest.mark.parametrize("vecs", [[1], [1, 7, 7]])
# function
@pytest.mark.parametrize("func", ["add", "mul"])
# execution mode
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.vivado
@pytest.mark.slow
def test_fpgadataflow_channelwise_ops(idt, act, nf, ich, func, vecs, exec_mode):
    if nf == -1:
        nf = ich
    pe = ich // nf
    assert ich % pe == 0

    # generate input data
    x = gen_finn_dt_tensor(idt, tuple(vecs + [ich]))

    odt = act
    C = np.random.randint(idt.min(), idt.max() + 1, ich).astype(np.float32)

    model = make_modelwrapper(C, pe, idt, odt, func, vecs)

    if exec_mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))
    elif exec_mode == "rtlsim":
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareIP("xc7z020clg400-1", 5))
        model = model.transform(HLSSynthIP())
        model = model.transform(ReplaceVerilogRelPaths())
        model = model.transform(PrepareRTLSim())
    else:
        raise Exception("Unknown exec_mode")

    # package input data as dictionary
    input_dict = {"inp": x}

    oshape = model.get_tensor_shape("outp")

    C_reshaped = np.broadcast_to(C.flatten(), x.shape)
    if func == "add":
        y = x + C_reshaped
    elif func == "mul":
        y = x * C_reshaped

    y_expected = y.reshape(oshape)
    # execute model
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]

    y_produced = y_produced.reshape(y_expected.shape)

    assert (y_produced == y_expected).all(), "cppsim failed"

    if exec_mode == "rtlsim":
        hls_synt_res_est = model.analysis(hls_synth_res_estimation)
        assert "ChannelwiseOp_Batch_0" in hls_synt_res_est
