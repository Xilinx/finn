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
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.general import GiveUniqueNodeNames
from finn.util.basic import gen_finn_dt_tensor


def make_single_im2col_modelwrapper(k, ifm_ch, ifm_dim, ofm_dim, simd, stride, idt):
    odt = idt
    inp = helper.make_tensor_value_info(
        "inp", TensorProto.FLOAT, [1, ifm_dim, ifm_dim, ifm_ch]
    )
    outp = helper.make_tensor_value_info(
        "outp", TensorProto.FLOAT, [1, ofm_dim, ofm_dim, k * k * ifm_ch]
    )

    im2col_node = helper.make_node(
        "Im2Col",
        ["inp"],
        ["outp"],
        domain="finn",
        backend="fpgadataflow",
        stride=stride,
        kernel_size=k,
        input_shape=str((1, ifm_dim, ifm_dim, ifm_ch)),
        pad_amount=0,
        pad_value=0,
    )
    graph = helper.make_graph(
        nodes=[im2col_node], name="im2col_graph", inputs=[inp], outputs=[outp]
    )

    model = helper.make_model(graph, producer_name="im2col-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)

    return model


def make_single_slidingwindow_modelwrapper(
    k, ifm_ch, ifm_dim, ofm_dim, simd, stride, idt
):
    odt = idt
    inp = helper.make_tensor_value_info(
        "inp", TensorProto.FLOAT, [1, ifm_dim, ifm_dim, ifm_ch]
    )
    outp = helper.make_tensor_value_info(
        "outp", TensorProto.FLOAT, [1, ofm_dim, ofm_dim, k * k * ifm_ch]
    )

    SlidingWindow_node = helper.make_node(
        "ConvolutionInputGenerator",
        ["inp"],
        ["outp"],
        domain="finn",
        backend="fpgadataflow",
        ConvKernelDim=k,
        IFMChannels=ifm_ch,
        IFMDim=ifm_dim,
        OFMDim=ofm_dim,
        SIMD=simd,
        Stride=stride,
        inputDataType=idt.name,
        outputDataType=odt.name,
    )
    graph = helper.make_graph(
        nodes=[SlidingWindow_node],
        name="slidingwindow_graph",
        inputs=[inp],
        outputs=[outp],
    )

    model = helper.make_model(graph, producer_name="slidingwindow-model")
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
@pytest.mark.parametrize("ifm_ch", [2, 4])  # , 2, 3, 4])
# Stride
@pytest.mark.parametrize("stride", [1, 2])
# execution mode
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
# input channel parallelism ("SIMD")
@pytest.mark.parametrize("simd", [1, 2])
def test_fpgadataflow_slidingwindow(idt, k, ifm_dim, ifm_ch, stride, exec_mode, simd):
    ofm_dim = int(((ifm_dim - k) / stride) + 1)

    x = gen_finn_dt_tensor(idt, (1, ifm_dim, ifm_dim, ifm_ch))
    model = make_single_slidingwindow_modelwrapper(
        k, ifm_ch, ifm_dim, ofm_dim, simd, stride, idt
    )

    if exec_mode == "cppsim":
        model = model.transform(SetExecMode("cppsim"))
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
    elif exec_mode == "rtlsim":
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareIP("xc7z020clg400-1", 5))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())
    else:
        raise Exception("Unknown exec_mode in test_fpgadataflow_slidingwindow")

    # prepare input data
    input_dict = prepare_inputs(x)
    # execute model
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    golden = make_single_im2col_modelwrapper(
        k, ifm_ch, ifm_dim, ofm_dim, simd, stride, idt
    )
    y_expected = oxe.execute_onnx(golden, input_dict)["outp"]
    # if idt == DataType.BIPOLAR:
    #     y_expected = 2 * y_expected - 1
    assert (y_produced == y_expected).all()
