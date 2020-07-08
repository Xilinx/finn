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
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.util.basic import gen_finn_dt_tensor


def make_single_vvau_modelwrapper(W, pe, dim, ch, k, wdt, idt, odt, T=None, tdt=None):

    inp = helper.make_tensor_value_info(
        "inp", TensorProto.FLOAT, [1, dim, dim, k * k * ch]
    )
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, dim, dim, ch])
    if T is not None:
        no_act = 0
        node_inp_list = ["inp", "weights", "thresh"]
        if odt == DataType.BIPOLAR:
            actval = 0
        else:
            actval = odt.min()
    else:
        # no thresholds
        node_inp_list = ["inp", "weights"]
        actval = 0
        no_act = 1

    VVAU_node = helper.make_node(
        "Vector_Vector_Activate_Batch",
        node_inp_list,
        ["outp"],
        domain="finn",
        backend="fpgadataflow",
        resType="ap_resource_lut()",
        PE=pe,
        Dim=dim,
        Channels=ch,
        Kernel=k,
        ActVal=actval,
        inputDataType=idt.name,
        weightDataType=wdt.name,
        outputDataType=odt.name,
        noActivation=no_act,
    )
    graph = helper.make_graph(
        nodes=[VVAU_node], name="vvau-graph", inputs=[inp], outputs=[outp]
    )

    model = helper.make_model(graph, producer_name="vvau-model")
    model = ModelWrapper(model)
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("weights", wdt)
    model.set_initializer("weights", W)

    if T is not None:
        model.set_tensor_datatype("thresh", tdt)
        model.set_initializer("thresh", T)
    return model


# pe
@pytest.mark.parametrize("pe", [1, 2])  # , 4, 8])
# input datatype
@pytest.mark.parametrize("idt", [DataType.INT2, DataType.INT4])
# weight datatype
@pytest.mark.parametrize("wdt", [DataType.INT2, DataType.INT4])
# kernel size
@pytest.mark.parametrize("k", [2, 4])
# dimension
@pytest.mark.parametrize("dim", [4, 6])
# channels
@pytest.mark.parametrize("ch", [2])  # 2, 4])
def test_fpgadataflow_vvau_cppsim(pe, idt, wdt, k, dim, ch):
    odt = DataType.INT32
    # generate weights
    W = gen_finn_dt_tensor(wdt, (ch, 1, k, k))
    model = make_single_vvau_modelwrapper(W, pe, dim, ch, k, wdt, idt, odt)
    model = model.transform(SetExecMode("cppsim"))
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())
    # generate inputs
    x = gen_finn_dt_tensor(idt, (1, dim, dim, k * k * ch))

    idict = {"inp": x}
    y_produced = oxe.execute_onnx(model, idict)["outp"]

    # test
    W_sparse = np.zeros((ch, ch, k, k))
    for c in range(ch):
        W_sparse[c][c] = W[c][0]

    if pe == 2:
        W_sparse = W_sparse.transpose(0, 2, 3, 1)
    W_sparse = W_sparse.reshape(ch, k * k * ch)
    y_expected = np.matmul(x, W_sparse.T)

    assert (y_produced == y_expected).all()
