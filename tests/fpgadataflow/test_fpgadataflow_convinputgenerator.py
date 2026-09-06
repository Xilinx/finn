# Copyright (c) 2020-2022, Xilinx
# Copyright (C) 2023-2024, Advanced Micro Devices, Inc.
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
from qonnx.custom_op.general.im2col import compute_conv_output_dim
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers


def make_single_im2col_modelwrapper(k, ifm_ch, ifm_dim, ofm_dim, stride, dilation, idt, dw):
    k_h, k_w = k
    ifm_dim_h, ifm_dim_w = ifm_dim
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    ofm_dim_h, ofm_dim_w = ofm_dim

    odt = idt
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, ifm_dim_h, ifm_dim_w, ifm_ch])
    outp = helper.make_tensor_value_info(
        "outp", TensorProto.FLOAT, [1, ofm_dim_h, ofm_dim_w, k_h * k_w * ifm_ch]
    )

    im2col_node = helper.make_node(
        "Im2Col",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.general",
        stride=[stride_h, stride_w],
        kernel_size=[k_h, k_w],
        input_shape=str((1, ifm_dim_h, ifm_dim_w, ifm_ch)),
        dilations=[dilation_h, dilation_w],
        pad_amount=[0, 0, 0, 0],
        pad_value=0,
        depthwise=dw,
    )
    graph = helper.make_graph(
        nodes=[im2col_node], name="im2col_graph", inputs=[inp], outputs=[outp]
    )

    model = qonnx_make_model(graph, producer_name="im2col-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)

    return model


def prepare_inputs(input_tensor):
    return {"inp": input_tensor}


# input datatype
@pytest.mark.parametrize("idt", [DataType["INT2"], DataType["UINT4"]])
# kernel size
@pytest.mark.parametrize("k", [[2, 2], [3, 3], [1, 5]])
# input dimension
@pytest.mark.parametrize("ifm_dim", [[8, 8], [1, 21]])
# input channels
@pytest.mark.parametrize("ifm_ch", [2, 4])
# Stride
@pytest.mark.parametrize("stride", [[1, 1], [2, 2], [2, 1]])
# Dilation
@pytest.mark.parametrize("dilation", [[1, 1], [2, 2], [2, 1]])
# execution mode
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
# input channel parallelism ("SIMD")
@pytest.mark.parametrize("simd", [1, 2, 4])
# depthwise
@pytest.mark.parametrize("dw", [0, 1])
# parallel_window enable (MMV_out = M*K)
@pytest.mark.parametrize("parallel_window", [0, 1])
# in/out MMV ("M")
@pytest.mark.parametrize("m", [1])
# Flip dimensions
@pytest.mark.parametrize("flip", [False])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_slidingwindow(
    idt,
    k,
    ifm_dim,
    ifm_ch,
    stride,
    dilation,
    exec_mode,
    simd,
    dw,
    parallel_window,
    m,
    flip,
):
    if flip:
        if (
            ifm_dim[0] == ifm_dim[1]
            and k[0] == k[1]
            and stride[0] == stride[1]
            and dilation[0] == dilation[1]
        ):
            pytest.skip("Dimension flip would have no effect")
        k = k[::-1]
        ifm_dim = ifm_dim[::-1]
        stride = stride[::-1]
        dilation = dilation[::-1]

    k_h, k_w = k
    ifm_dim_h, ifm_dim_w = ifm_dim
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation

    kernel_width = (k_w - 1) * dilation_w + 1  # incl. dilation
    kernel_height = (k_h - 1) * dilation_h + 1  # incl. dilation

    if simd > ifm_ch:
        pytest.skip("SIMD cannot be larger than number of input channels")
    if ifm_ch % simd != 0:
        pytest.skip("SIMD must divide number of input channels")
    if kernel_height > ifm_dim_h or stride_h > ifm_dim_h:
        pytest.skip("Illegal convolution configuration: kernel or stride > FM dimension")
    if kernel_width > ifm_dim_w or stride_w > ifm_dim_w:
        pytest.skip("Illegal convolution configuration: kernel or stride > FM dimension")
    if (k_h == 1 and dilation_h != 1) or (k_w == 1 and dilation_w != 1):
        pytest.skip("Illegal convolution configuration: dilation for unitary kernel dim")
    if ((stride_h > k_h) or (stride_w > k_w)) and not (parallel_window or (k_h == 1 and k_w == 1)):
        pytest.skip("Not all combinations for stride > k edge case supported in default mode")
    if parallel_window and simd != ifm_ch and not (dw or (k_h == 1 and k_w == 1)):
        pytest.skip("Parallel window requires SIMD=C for non-depthwise case")

    ofm_dim_h = compute_conv_output_dim(ifm_dim_h, k_h, stride_h, 0, dilation_h)
    ofm_dim_w = compute_conv_output_dim(ifm_dim_w, k_w, stride_w, 0, dilation_w)
    ofm_dim = [ofm_dim_h, ofm_dim_w]

    x = gen_finn_dt_tensor(idt, (1, ifm_dim_h, ifm_dim_w, ifm_ch))
    # prepare input data
    input_dict = prepare_inputs(x)
    model = make_single_im2col_modelwrapper(k, ifm_ch, ifm_dim, ofm_dim, stride, dilation, idt, dw)
    y_expected = oxe.execute_onnx(model, input_dict)["outp"]

    model = model.transform(to_hw.InferConvInpGen())
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    assert (y_produced == y_expected).all()
    model = model.transform(SpecializeLayers("xc7z020clg400-1"))
    # set simd
    inst = getCustomOp(model.graph.node[0])
    inst.set_nodeattr("SIMD", simd)
    optype = model.graph.node[0].op_type
    if optype == "ConvolutionInputGenerator_rtl":
        inst.set_nodeattr("parallel_window", parallel_window)
        inst.set_nodeattr("M", m)

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

    # execute model
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]

    if dw == 0:
        assert (y_produced == y_expected).all()
    else:
        y_expected = y_expected.reshape(1, ofm_dim_h, ofm_dim_w, k_h * k_w, ifm_ch // simd, simd)
        y_expected = y_expected.transpose(0, 1, 2, 4, 3, 5)
        y_expected = y_expected.reshape(1, ofm_dim_h, ofm_dim_w, ifm_ch * k_h * k_w)
        assert (y_produced == y_expected).all()

    if exec_mode == "rtlsim":
        nodes = model.get_nodes_by_op_type("ConvolutionInputGenerator_rtl")
        node = nodes[0]
        inst = getCustomOp(node)
        cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
        exp_cycles_dict = model.analysis(exp_cycles_per_layer)
        exp_cycles = exp_cycles_dict[node.name]
        assert np.isclose(exp_cycles, cycles_rtlsim, atol=10, rtol=1.1)
        assert exp_cycles != 0
