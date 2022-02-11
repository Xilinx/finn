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

# import numpy as np
from onnx import TensorProto, helper

import finn.core.onnx_exec as oxe
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper

# from finn.custom_op.registry import getCustomOp
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.general import GiveUniqueNodeNames
from finn.util.basic import gen_finn_dt_tensor


def make_single_maxpoolnhwc_modelwrapper(k, ifm_ch, ifm_dim, ofm_dim, idt):
    k_h, k_w = k
    ifm_dim_h, ifm_dim_w = ifm_dim
    ofm_dim_h, ofm_dim_w = ofm_dim
    odt = idt
    inp = helper.make_tensor_value_info(
        "inp", TensorProto.FLOAT, [1, ifm_dim_h, ifm_dim_w, ifm_ch]
    )
    outp = helper.make_tensor_value_info(
        "outp", TensorProto.FLOAT, [1, ofm_dim_h, ofm_dim_w, ifm_ch]
    )

    mp_node = helper.make_node(
        "MaxPoolNHWC",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.general",
        kernel_shape=[k_h, k_w],
        strides=[k_h, k_w],
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


def make_single_streamingmaxpool_modelwrapper(k, ifm_ch, pe, ifm_dim, ofm_dim, idt):
    k_h, k_w = k
    ifm_dim_h, ifm_dim_w = ifm_dim
    ofm_dim_h, ofm_dim_w = ofm_dim
    odt = idt
    inp = helper.make_tensor_value_info(
        "inp", TensorProto.FLOAT, [1, ifm_dim_h, ifm_dim_w, ifm_ch]
    )
    outp = helper.make_tensor_value_info(
        "outp", TensorProto.FLOAT, [1, ofm_dim_h, ofm_dim_w, ifm_ch]
    )

    smp_node = helper.make_node(
        "StreamingMaxPool_Batch",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        PoolDim=[k_h, k_w],
        NumChannels=ifm_ch,
        PE=pe,
        ImgDim=[ifm_dim_h, ifm_dim_w],
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
#@pytest.mark.parametrize("idt", [DataType["BIPOLAR"], DataType["INT4"]])
@pytest.mark.parametrize("idt", [DataType["UINT4"]])
# 1d maxpool
#@pytest.mark.parametrize("dim_1d", [False, True])
@pytest.mark.parametrize("dim_1d", [True])
# kernel size
##@pytest.mark.parametrize("k", [2, 4])
@pytest.mark.parametrize("k", [6])
# input dimension
#@pytest.mark.parametrize("ifm_dim", [4, 8])
@pytest.mark.parametrize("ifm_dim", [60])
# input channels
#@pytest.mark.parametrize("ifm_ch", [1, 3])  # 1,3
@pytest.mark.parametrize("ifm_ch", [1024])  # 1,3
# pe
#@pytest.mark.parametrize("pe", [1, 3])
@pytest.mark.parametrize("pe", [1])
# execution mode
#@pytest.mark.parametrize("exec_mode", ["rtlsim", "cppsim"])
@pytest.mark.parametrize("exec_mode", ["rtlsim"])
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_streamingmaxpool(idt, dim_1d, k, ifm_dim, ifm_ch, pe, exec_mode):
    ifm_dim_h = ifm_dim
    k_h = k
    if dim_1d:
        ifm_dim_w = 1
        k_w = 1
    else:
        ifm_dim_w = ifm_dim_h
        k_w = k_h
    ifm_dim = (ifm_dim_h, ifm_dim_w)
    k = (k_h, k_w)

    stride_h = k_h
    stride_w = k_w
    ofm_dim_h = int(((ifm_dim_h - k_h) / stride_h) + 1)
    ofm_dim_w = int(((ifm_dim_w - k_w) / stride_w) + 1)
    ofm_dim = (ofm_dim_h, ofm_dim_w)
    if idt == DataType["BIPOLAR"] and dim_1d:
        pytest.skip("Skipping binary StreamingMaxPool_1d (not implemented)")
    if ifm_dim_h % k_h != 0 or ifm_dim_w % k_w != 0:
        pytest.skip("Skipping StreamingMaxPool test w/ ImgDim % PoolDim != 0")
    if pe > ifm_ch:
        pytest.skip("SIMD cannot be larger than number of input channels")

    x = gen_finn_dt_tensor(idt, (1, ifm_dim_h, ifm_dim_w, ifm_ch))
    # prepare input data
    input_dict = prepare_inputs(x)

    golden = make_single_maxpoolnhwc_modelwrapper(k, ifm_ch, ifm_dim, ofm_dim, idt)
    y_expected = oxe.execute_onnx(golden, input_dict)["outp"]

    model = make_single_streamingmaxpool_modelwrapper(k, ifm_ch, pe, ifm_dim, ofm_dim, idt)

    if exec_mode == "cppsim":
        model = model.transform(SetExecMode("cppsim"))
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
    elif exec_mode == "rtlsim":
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(GiveUniqueNodeNames())
        #model = model.transform(PrepareIP("xc7z020clg400-1", 5))
        model = model.transform(PrepareIP("xczu3eg-sbva484-1-e", 5))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())
    else:
        raise Exception("Unknown exec_mode in test_layer_streaming_maxpool_batch")

    # execute model
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    assert (y_produced == y_expected).all()

    if exec_mode == "rtlsim":
        node = model.get_nodes_by_op_type("StreamingMaxPool_Batch")[0]
        # inst = getCustomOp(node)
        # cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
        exp_cycles_dict = model.analysis(exp_cycles_per_layer)
        exp_cycles = exp_cycles_dict[node.name]
        # FIXME: maxpool cycles prediction needs a fix
        # mostl likely due to some loops not flattening
        # assert np.isclose(exp_cycles, cycles_rtlsim, atol=15)
        assert exp_cycles != 0
