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

from onnx import TensorProto, helper

import finn.core.onnx_exec as oxe
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.fpgadataflow.codegen_ipgen import CodeGen_ipgen
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile import Compile
from finn.transformation.fpgadataflow.hlssynth_ipgen import HLSSynth_IPGen
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.general import GiveUniqueNodeNames
from finn.util.basic import gen_finn_dt_tensor


def make_single_maxpoolnhwc_modelwrapper(k, ifm_ch, ifm_dim, ofm_dim, idt):
    odt = idt
    inp = helper.make_tensor_value_info(
        "inp", TensorProto.FLOAT, [1, ifm_dim, ifm_dim, ifm_ch]
    )
    outp = helper.make_tensor_value_info(
        "outp", TensorProto.FLOAT, [1, ofm_dim, ofm_dim, ifm_ch]
    )

    mp_node = helper.make_node(
        "MaxPoolNHWC",
        ["inp"],
        ["outp"],
        domain="finn",
        kernel_shape=[k, k],
        strides=[k, k],
        pads=[0, 0, 0, 0],
    )
    graph = helper.make_graph(
        nodes=[mp_node], name="mp_graph", inputs=[inp], outputs=[outp]
    )

    model = helper.make_model(graph, producer_name="mp-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)

    return model


def make_single_streamingmaxpool_modelwrapper(k, ifm_ch, ifm_dim, ofm_dim, idt):
    odt = idt
    inp = helper.make_tensor_value_info(
        "inp", TensorProto.FLOAT, [1, ifm_dim, ifm_dim, ifm_ch]
    )
    outp = helper.make_tensor_value_info(
        "outp", TensorProto.FLOAT, [1, ofm_dim, ofm_dim, ifm_ch]
    )

    smp_node = helper.make_node(
        "StreamingMaxPool_Batch",
        ["inp"],
        ["outp"],
        domain="finn",
        backend="fpgadataflow",
        PoolDim=k,
        NumChannels=ifm_ch,
        ImgDim=ifm_dim,
        dataType=idt.name,
    )
    graph = helper.make_graph(
        nodes=[smp_node], name="smp_graph", inputs=[inp], outputs=[outp]
    )

    model = helper.make_model(graph, producer_name="smp-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)

    return model


def prepare_inputs(input_tensor):
    return {"inp": input_tensor}


# input datatype
@pytest.mark.parametrize("idt", [DataType.BIPOLAR, DataType.INT2])
# kernel size
@pytest.mark.parametrize("k", [2, 4])
# input dimension
@pytest.mark.parametrize("ifm_dim", [4, 6, 8])
# input channels
@pytest.mark.parametrize("ifm_ch", [1, 2])  # , 2, 3, 4])
# execution mode
@pytest.mark.parametrize("exec_mode", ["rtlsim", "npysim"])
def test_fpgadataflow_streamingmaxpool(idt, k, ifm_dim, ifm_ch, exec_mode):
    stride = k
    ofm_dim = int(((ifm_dim - k) / stride) + 1)
    if ifm_dim % k != 0:
        pytest.skip("Skipping StreamingMaxPool test w/ ImgDim % PoolDim != 0")

    x = gen_finn_dt_tensor(idt, (1, ifm_dim, ifm_dim, ifm_ch))
    # prepare input data
    input_dict = prepare_inputs(x)

    golden = make_single_maxpoolnhwc_modelwrapper(k, ifm_ch, ifm_dim, ofm_dim, idt)
    y_expected = oxe.execute_onnx(golden, input_dict)["outp"]

    model = make_single_streamingmaxpool_modelwrapper(k, ifm_ch, ifm_dim, ofm_dim, idt)

    if exec_mode == "npysim":
        model = model.transform(SetExecMode("npysim"))
        model = model.transform(PrepareCppSim())
        model = model.transform(Compile())
    elif exec_mode == "rtlsim":
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(CodeGen_ipgen("xc7z020clg400-1", 5))
        model = model.transform(HLSSynth_IPGen())
        model = model.transform(PrepareRTLSim())
    else:
        raise Exception("Unknown exec_mode in test_fpgadataflow_slidingwindow")

    # execute model
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    assert (y_produced == y_expected).all()
