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
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.fpgadataflow.codegen_ipgen import CodeGen_ipgen
from finn.transformation.fpgadataflow.codegen_npysim import CodeGen_npysim
from finn.transformation.fpgadataflow.compile import Compile
from finn.transformation.fpgadataflow.hlssynth_ipgen import HLSSynth_IPGen
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.general import GiveUniqueNodeNames
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.util.basic import gen_finn_dt_tensor
from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)


def make_labelselect_modelwrapper(labels, pe, k, idt):
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, labels])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, k])

    labelselect_node = helper.make_node(
        "LabelSelect_Batch",
        ["inp"],
        ["outp"],
        domain="finn",
        backend="fpgadataflow",
        Labels=labels,
        PE=pe,
        K=k,
        inputDataType=idt.name,
    )
    graph = helper.make_graph(
        nodes=[labelselect_node], name="graph", inputs=[inp], outputs=[outp],
    )

    model = helper.make_model(graph, producer_name="thresholding-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", DataType.UINT32)

    return model


def prepare_inputs(input_tensor, idt):
    return {"inp": input_tensor}


# TODO: folded inputs and INTx inputs fail, likely problem in hlslib
# input datatype
@pytest.mark.parametrize("idt", [DataType.UINT8, DataType.UINT16])
# labels
@pytest.mark.parametrize("labels", [10, 1000])
# folding
@pytest.mark.parametrize("fold", [-1])
# number of top labels to select
@pytest.mark.parametrize("k", [1, 5])
# execution mode
@pytest.mark.parametrize("exec_mode", ["npysim", "rtlsim"])
def test_fpgadataflow_labelselect(idt, labels, fold, k, exec_mode):
    if fold == -1:
        pe = 1
    else:
        pe = labels // fold
    assert labels % pe == 0

    if k == -1:
        k = labels

    # generate input data
    x = gen_finn_dt_tensor(idt, (1, labels))

    model = make_labelselect_modelwrapper(labels, pe, k, idt)

    if exec_mode == "npysim":
        model = model.transform(SetExecMode("npysim"))
        model = model.transform(CodeGen_npysim())
        model = model.transform(Compile())
    elif exec_mode == "rtlsim":
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(CodeGen_ipgen("xc7z020clg400-1", 5))
        model = model.transform(HLSSynth_IPGen())
        model = model.transform(ReplaceVerilogRelPaths())
        model = model.transform(PrepareRTLSim())
    else:
        raise Exception("Unknown exec_mode")

    # prepare input data
    input_dict = prepare_inputs(x, idt)

    oshape = model.get_tensor_shape("outp")
    y = np.flip(x.flatten().argsort())[:k]
    y_expected = y.reshape(oshape)
    # execute model
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]

    y_produced = y_produced.reshape(y_expected.shape)

    # we want to check that the actual values selected are the same
    # this is because sometimes the order of equal elements differs
    # in y_expected vs y_produced, but that is not relevant in practice
    selected_x_vals_expected = x.flatten()[tuple(y_expected.astype(np.int))]
    selected_x_vals_produced = x.flatten()[tuple(y_produced.astype(np.int))]

    assert (selected_x_vals_expected == selected_x_vals_produced).all(), (
        exec_mode + " failed"
    )
