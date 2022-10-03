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
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.multithreshold import multithreshold
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import gen_finn_dt_tensor

import finn.core.onnx_exec as oxe
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode


def _infer_sparse_weight_tensor(W_conv, k_h, k_w, channels):
    W_sparse = np.zeros((channels, channels, k_h, k_w), dtype=np.float32)
    for ch in range(channels):
        W_sparse[ch][ch] = W_conv[ch][0]
    W_conv = W_sparse.astype(np.float32)
    W_matmul = W_conv.transpose(0, 2, 3, 1)
    W_matmul = W_matmul.reshape(channels, channels * k_h * k_w)
    W_matmul = W_matmul.T

    return W_matmul


def _calculate_dot_prod_range(dt_a, dt_b, len):
    """Returns the (min,max) values a dot product between two (un)signed vectors of
    types dt_a and dt_b of len elements can take."""
    min_prod = 2**30
    max_prod = -(2**30)
    for a_val in [dt_a.min(), dt_a.max()]:
        for b_val in [dt_b.min(), dt_b.max()]:
            prod = a_val * b_val * len
            if prod < min_prod:
                min_prod = prod
            if prod > max_prod:
                max_prod = prod
    return (min_prod, max_prod)


def _make_single_vvau_modelwrapper(
    W,
    pe,
    k_h,
    k_w,
    channels,
    dim_h,
    dim_w,
    wdt,
    idt,
    odt,
    T=None,
    tdt=None,
    mem_mode="const",
):
    in_shape = [1, dim_h, dim_w, k_h * k_w * channels]  # [N, H, W, K*K*CH]
    out_shape = [
        1,
        dim_h,
        dim_w,
        channels,
    ]  # [N, H, W, OFM_CH] (OFM_CH=IFM_CH because depthwise convolution)

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, in_shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, out_shape)

    if T is not None:
        no_act = 0
        node_inp_list = ["inp", "weights", "thresh"]
        actval = odt.min()
    else:
        no_act = 1
        node_inp_list = ["inp", "weights"]
        actval = 0

    VVAU_node = helper.make_node(
        "VectorVectorActivation",
        node_inp_list,
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        PE=pe,
        Dim=[dim_h, dim_w],
        Channels=channels,
        Kernel=[k_h, k_w],
        resType="lut",
        ActVal=actval,
        inputDataType=idt.name,
        weightDataType=wdt.name,
        outputDataType=odt.name,
        noActivation=no_act,
        mem_mode=mem_mode,
    )

    graph = helper.make_graph(
        nodes=[VVAU_node], name="vvau_graph", inputs=[inp], outputs=[outp]
    )

    model = helper.make_model(graph, producer_name="vvau-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("weights", wdt)

    model.set_initializer("weights", W)
    model.set_tensor_shape("weights", (channels, 1, k_h, k_w))

    if T is not None:
        model.set_tensor_datatype("thresh", tdt)
        model.set_initializer("thresh", T)

    return model


def prepare_inputs(input_tensor):
    return {"inp": input_tensor}


# input datatype
@pytest.mark.parametrize("idt", [DataType["UINT4"], DataType["UINT8"]])
# weight datatype
@pytest.mark.parametrize("wdt", [DataType["INT4"]])
# activation: None or DataType
@pytest.mark.parametrize("act", [DataType["UINT4"], None])
# PE
@pytest.mark.parametrize("pe", [1, "channels"])
# Input image shape
@pytest.mark.parametrize("dim_h", [10])
@pytest.mark.parametrize("dim_w", [10, 1])
# Kernel shape
@pytest.mark.parametrize("k_h", [3])
@pytest.mark.parametrize("k_w", [3, 1])
# Number of input and output channels
@pytest.mark.parametrize("channels", [3, 4])
# memory mode
@pytest.mark.parametrize("mem_mode", ["const", "decoupled"])
# execution mode
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_vvau(
    idt, wdt, act, pe, dim_h, dim_w, k_h, k_w, channels, mem_mode, exec_mode
):
    if pe == "channels":
        pe = channels

    if dim_w == 1 and k_w != 1:
        pytest.skip("1D image requires 1D kernel, skipping.")

    if channels % pe != 0:
        pytest.skip("Requirement Channels divisable by PE is violated.")

    # Generate weights in expected shape for ONNX and HLS node
    W = gen_finn_dt_tensor(wdt, (channels, 1, k_h, k_w))  # shape: [channels, 1, k, k]
    W_onnx = _infer_sparse_weight_tensor(
        W, k_h, k_w, channels
    )  # shape: [k*k*channels, channels]

    # Generate inputs in expected format for ONNX and HLS node
    x = gen_finn_dt_tensor(idt, (1, dim_h, dim_w, k_h * k_w * channels))
    x_vvau = x.reshape(1, dim_h, dim_w, k_h * k_w, channels // pe, pe)
    x_vvau = x_vvau.transpose(0, 1, 2, 4, 3, 5)
    x_vvau = x_vvau.reshape(1, dim_h, dim_w, channels * k_h * k_w)

    if act is None:
        T = None
        tdt = None
        odt = DataType["INT32"]
    else:
        odt = act
        (min_v, max_v) = _calculate_dot_prod_range(idt, wdt, k_h * k_w * channels)
        n_steps = act.get_num_possible_values() - 1
        T = np.random.randint(min_v, max_v - 1, (channels, n_steps)).astype(np.float32)
        T = np.sort(T, axis=1)
        tdt = DataType["INT32"]

    model = _make_single_vvau_modelwrapper(
        W, pe, k_h, k_w, channels, dim_h, dim_w, wdt, idt, odt, T, tdt, mem_mode
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
        raise Exception("Unknown exec_mode in test_fpgadataflow_vvau")

    input_dict = prepare_inputs(x_vvau)

    # Calculate output
    y_expected = np.matmul(x, W_onnx)  # Y is in [N, H, W, C] format
    if T is not None:
        # Reshape Y, as multithreshold expects Y to be in [N, C, H, W] format
        y_expected = np.transpose(y_expected, (0, 3, 1, 2))
        y_expected = multithreshold(y_expected, T)
        y_expected = np.transpose(y_expected, (0, 2, 3, 1))
        # signed offset
        y_expected += act.min()

    y_produced = oxe.execute_onnx(model, input_dict, return_full_exec_context=False)[
        "outp"
    ]

    assert (y_produced == y_expected).all(), "cppsim failed"

    if exec_mode == "rtlsim":
        node = model.get_nodes_by_op_type("VectorVectorActivation")[0]
        inst = getCustomOp(node)
        cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
        exp_cycles_dict = model.analysis(exp_cycles_per_layer)
        exp_cycles = exp_cycles_dict[node.name]
        assert np.isclose(exp_cycles, cycles_rtlsim, atol=10)
        assert exp_cycles != 0
