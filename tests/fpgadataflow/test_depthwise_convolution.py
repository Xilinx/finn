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
import onnx.helper as oh
from onnx import TensorProto
import numpy as np

from finn.core.modelwrapper import ModelWrapper
from finn.core.datatype import DataType
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.fpgadataflow.convert_to_hls_layers import (
    InferConvInpGen,
    InferVVAU,
)
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode

import finn.core.onnx_exec as oxe
from finn.custom_op.general.im2col import compute_conv_output_dim
from finn.util.basic import calculate_signed_dot_prod_range, gen_finn_dt_tensor
from finn.custom_op.registry import getCustomOp

from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.general import GiveUniqueNodeNames
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim


def set_up_reference_model(act, idt, wdt, k, ifm_dim, ifm_ch, stride, padding):

    # set up reference model consisting of Im2Col + MatMul (+ MultiThreshold)
    ofm_ch = ifm_ch
    total_pad = 2 * padding
    ofm_dim = compute_conv_output_dim(ifm_dim, k, stride, total_pad=total_pad)

    if act is None:
        odt = DataType.INT32
    else:
        odt = act
        out_act = oh.make_tensor_value_info(
            "out_act", TensorProto.FLOAT, [1, ofm_dim, ofm_dim, ofm_ch]
        )
        T = oh.make_tensor_value_info("T", TensorProto.FLOAT, [ofm_ch, 15])
        tdt = DataType.INT32
        thresh_node = oh.make_node(
            "MultiThreshold",
            domain="finn.custom_op.general",
            inputs=["outp", "T"],
            outputs=["out_act"],
            data_layout="NHWC",
            out_dtype=odt.name,
            out_scale=1.0,
            out_bias=0.0,
        )

    # set up onnx model
    inp = oh.make_tensor_value_info(
        "inp", TensorProto.FLOAT, [1, ifm_dim, ifm_dim, ifm_ch]
    )
    outp = oh.make_tensor_value_info(
        "outp", TensorProto.FLOAT, [1, ofm_dim, ofm_dim, ofm_ch]
    )

    W_sparse = oh.make_tensor_value_info(
        "W_sparse", TensorProto.FLOAT, [ifm_ch * k * k, ofm_ch]
    )

    im2col_node = oh.make_node(
        "Im2Col",
        domain="finn.custom_op.general",
        inputs=["inp"],
        outputs=["im2col_out"],
        kernel_size=[k, k],
        stride=stride,
        pad_amount=[padding, padding, padding, padding],
        input_shape="(1, {}, {}, {})".format(ifm_dim, ifm_dim, ifm_ch),
        depthwise=1,
    )

    matmul_node = oh.make_node(
        "MatMul", inputs=["im2col_out", "W_sparse"], outputs=["outp"]
    )

    if act is None:
        node_list = [im2col_node, matmul_node]
        global_out = outp
        value_info = [W_sparse]
    else:
        node_list = [im2col_node, matmul_node, thresh_node]
        global_out = out_act
        value_info = [W_sparse, T]

    graph = oh.make_graph(
        nodes=node_list,
        name="lowered_dw_cnv_graph",
        inputs=[inp],
        outputs=[global_out],
        value_info=value_info,
    )
    model = oh.make_model(graph, producer_name="lowered_dw_cnv-model")
    model = ModelWrapper(model)

    # initialize model
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype(model.graph.output[0].name, odt)
    model.set_tensor_datatype("W_sparse", wdt)

    w_tensor = gen_finn_dt_tensor(wdt, [ofm_ch, 1, k, k])
    # create sparse matrix
    W_matrix = np.zeros((ofm_ch, ifm_ch, k, k))
    for ch in range(ifm_ch):
        W_matrix[ch][ch] = w_tensor[ch][0]
    W_matrix = W_matrix.astype(np.float32)
    W_matrix = W_matrix.transpose(0, 2, 3, 1)
    W_matrix = W_matrix.reshape(ofm_ch, ifm_ch * k * k)

    model.set_initializer("W_sparse", W_matrix.T)
    sparsity = {"dw": {"kernel_shape": k}}
    model.set_tensor_sparsity("W_sparse", sparsity)

    if act is not None:
        (min, max) = calculate_signed_dot_prod_range(idt, wdt, ifm_ch * k * k)
        n_steps = odt.get_num_possible_values() - 1
        T_values = np.random.randint(min, max - 1, (ofm_ch, n_steps)).astype(np.float32)
        # provide non-decreasing thresholds
        T_values = np.sort(T_values, axis=1)
        model.set_initializer("T", T_values)
        model.set_tensor_datatype("T", tdt)

    model = model.transform(InferShapes())

    return model


# PE
@pytest.mark.parametrize("pe", [1, 2, 4])
# Output activation
@pytest.mark.parametrize("act", [None, DataType.UINT4])
# kernel size
@pytest.mark.parametrize("k", [2, 4])
# stride
@pytest.mark.parametrize("stride", [1, 2])
# padding
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.slow
@pytest.mark.vivado
def test_depthwise_conv_hls_cppsim(act, pe, k, stride, padding):
    idt = wdt = DataType.INT4
    ifm_dim = 6
    ifm_ch = 4

    # set up reference model consisting of Im2Col + MatMul (+ MultiThreshold)
    model = set_up_reference_model(act, idt, wdt, k, ifm_dim, ifm_ch, stride, padding)

    input_tensor = gen_finn_dt_tensor(idt, [1, ifm_dim, ifm_dim, ifm_ch])
    input_dict = {"inp": input_tensor}

    new_model = model.transform(InferConvInpGen())
    new_model = new_model.transform(InferVVAU())

    # set SIMD in ConvInputGen node and PE in VVAU node

    for n in new_model.graph.node:
        if n.op_type == "ConvolutionInputGenerator":
            convinputgen_node = getCustomOp(n)
            convinputgen_node.set_nodeattr("SIMD", pe)
        elif n.op_type == "Vector_Vector_Activate_Batch":
            vvau_node = getCustomOp(n)
            vvau_node.set_nodeattr("PE", pe)
    new_model = new_model.transform(SetExecMode("cppsim"))
    new_model = new_model.transform(PrepareCppSim())
    new_model = new_model.transform(CompileCppSim())

    assert oxe.compare_execution(model, new_model, input_dict)


# PE
@pytest.mark.parametrize("pe", [1, 2, 4])
# Output activation
@pytest.mark.parametrize("act", [None, DataType.UINT4])
# kernel size
@pytest.mark.parametrize("k", [2, 4])
# stride
@pytest.mark.parametrize("stride", [1, 2])
# padding
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.slow
@pytest.mark.vivado
def test_depthwise_conv_hls_rtlsim(act, pe, k, stride, padding):
    idt = wdt = DataType.INT4
    ifm_dim = 6
    ifm_ch = 4

    # set up reference model consisting of Im2Col + MatMul (+ MultiThreshold)
    model = set_up_reference_model(act, idt, wdt, k, ifm_dim, ifm_ch, stride, padding)

    input_tensor = gen_finn_dt_tensor(idt, [1, ifm_dim, ifm_dim, ifm_ch])
    input_dict = {"inp": input_tensor}

    new_model = model.transform(InferConvInpGen())
    new_model = new_model.transform(InferVVAU())

    # set SIMD in ConvInputGen node and PE in VVAU node

    for n in new_model.graph.node:
        if n.op_type == "ConvolutionInputGenerator":
            convinputgen_node = getCustomOp(n)
            convinputgen_node.set_nodeattr("SIMD", pe)
        elif n.op_type == "Vector_Vector_Activate_Batch":
            vvau_node = getCustomOp(n)
            vvau_node.set_nodeattr("PE", pe)

    new_model = new_model.transform(SetExecMode("rtlsim"))
    new_model = new_model.transform(GiveUniqueNodeNames())
    new_model = new_model.transform(PrepareIP("xc7z020clg400-1", 5))
    new_model = new_model.transform(HLSSynthIP())
    new_model = new_model.transform(PrepareRTLSim())

    assert oxe.compare_execution(model, new_model, input_dict)
