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
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.general.im2col import compute_conv_output_dim
from finn.custom_op.registry import getCustomOp
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.general import GiveUniqueNodeNames
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from finn.util.basic import gen_finn_dt_tensor


# conv_config:
# [pad_h_begin, pad_w_begin, pad_h_end, pad_w_end]
# [kernel_size_h, kernel_size_w]
# [stride_h, stride_w]
# [dilation_h, dilation_w]
@pytest.mark.parametrize(
    "conv_config",
    [
        [[0, 0, 0, 0], [4, 1], [1, 1], [1, 1]],
        [[1, 0, 1, 0], [4, 1], [1, 1], [1, 1]],
        [[1, 0, 1, 0], [4, 1], [2, 1], [1, 1]],
        # [[1, 0, 1, 0], [4, 1], [1, 1], [2, 1]]
    ],
)
@pytest.mark.parametrize("depthwise", [False, True])
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_convert_to_hls_1d_conv_layer(conv_config, depthwise, exec_mode):
    pad, kernel_size, stride, dilation = conv_config
    np.random.seed(0)
    idt = DataType["UINT4"]

    in_feature_dim_h, in_feature_dim_w = [10, 1]
    in_chn = 16

    k_h, k_w = kernel_size
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    pad_h = pad[0] + pad[2]
    pad_w = pad[1] + pad[3]

    if depthwise is True:
        group = out_chn = in_chn
        conv_param_shape = [out_chn, 1, k_h, k_w]
    else:
        group = 1
        out_chn = 20
        conv_param_shape = [out_chn, in_chn, k_h, k_w]

    out_feature_dim_h = compute_conv_output_dim(
        in_feature_dim_h, k_h, stride_h, pad_h, dilation_h
    )
    out_feature_dim_w = compute_conv_output_dim(
        in_feature_dim_w, k_w, stride_w, pad_w, dilation_w
    )

    input_shape = [1, in_chn, in_feature_dim_h, in_feature_dim_w]
    output_shape = [1, out_chn, out_feature_dim_h, out_feature_dim_w]

    conv_weight_dt = DataType["UINT4"]

    conv_config = {}
    conv_config["dilations"] = [dilation_h, dilation_w]
    conv_config["group"] = group
    conv_config["kernel_shape"] = [k_h, k_w]
    conv_config["pads"] = pad
    conv_config["strides"] = [stride_h, stride_w]

    top_in = helper.make_tensor_value_info("top_in", TensorProto.FLOAT, input_shape)
    top_out = helper.make_tensor_value_info("top_out", TensorProto.FLOAT, output_shape)
    value_info = [
        helper.make_tensor_value_info("p1", TensorProto.FLOAT, conv_param_shape)
    ]

    modelproto = helper.make_model(
        helper.make_graph(
            name="conv_test",
            inputs=[top_in],
            outputs=[top_out],
            value_info=value_info,
            nodes=[
                helper.make_node("Conv", ["top_in", "p1"], ["top_out"], **conv_config)
            ],
        )
    )

    model = ModelWrapper(modelproto)
    model.set_tensor_datatype("top_in", idt)
    model.set_tensor_datatype("top_out", idt)
    model.set_tensor_datatype("p1", conv_weight_dt)
    model.set_initializer("p1", gen_finn_dt_tensor(conv_weight_dt, conv_param_shape))

    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    new_model = model.transform(LowerConvsToMatMul())
    new_model = new_model.transform(to_hls.InferConvInpGen())
    if depthwise is True:
        new_model = new_model.transform(to_hls.InferVVAU())
    else:
        new_model = new_model.transform(to_hls.InferQuantizedMatrixVectorActivation())
        fc_node = new_model.get_nodes_by_op_type("MatrixVectorActivation")[0]
        fc_inst = getCustomOp(fc_node)
        mw = fc_inst.get_nodeattr("MW")
        mh = fc_inst.get_nodeattr("MH")
        pe_cands = list(filter(lambda x: mh % x == 0, range(2, mh + 1)))
        simd_cands = list(filter(lambda x: mw % x == 0, range(2, mw + 1)))
        fc_inst.set_nodeattr("PE", pe_cands[0])
        fc_inst.set_nodeattr("SIMD", simd_cands[0])

    new_model = new_model.transform(GiveUniqueNodeNames())
    new_model = new_model.transform(InferShapes())
    new_model = new_model.transform(InferDataTypes())

    if exec_mode == "cppsim":
        new_model = new_model.transform(PrepareCppSim())
        new_model = new_model.transform(CompileCppSim())
        new_model = new_model.transform(SetExecMode("cppsim"))
    elif exec_mode == "rtlsim":
        new_model = new_model.transform(SetExecMode("rtlsim"))
        new_model = new_model.transform(GiveUniqueNodeNames())
        new_model = new_model.transform(PrepareIP("xc7z020clg400-1", 5))
        new_model = new_model.transform(HLSSynthIP())
        new_model = new_model.transform(PrepareRTLSim())
    else:
        raise Exception("Unknown exec_mode")

    x = gen_finn_dt_tensor(idt, input_shape)
    inp_dict = {model.graph.input[0].name: x}
    assert oxe.compare_execution(model, new_model, inp_dict)

    if pad_h == 1 and pad_w == 1:
        padding_node = new_model.get_nodes_by_op_type("FMPadding_Batch")[0]
        padding_inst = getCustomOp(padding_node)
        assert padding_inst.get_nodeattr("SIMD") == in_chn

    if depthwise is True and exec_mode == "rtlsim":
        node = new_model.get_nodes_by_op_type("Vector_Vector_Activate_Batch")[0]
        inst = getCustomOp(node)
        cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
        exp_cycles_dict = new_model.analysis(exp_cycles_per_layer)
        exp_cycles = exp_cycles_dict[node.name]
        assert np.isclose(exp_cycles, cycles_rtlsim, atol=11)
        assert exp_cycles != 0
