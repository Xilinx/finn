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

# conv_config  kernel_size,stride, pad


@pytest.mark.parametrize(
    "conv_config", [(1, 2, 0), (1, 3, 0), (3, 2, 1), (3, 1, 0), (3, 1, 1), (5, 2, 1)]
)
@pytest.mark.parametrize("depthwise", [False, True])
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.convert_to_hls
@pytest.mark.slow
@pytest.mark.vivado
def test_convert_to_hls_conv_layer(conv_config, depthwise, exec_mode):
    kernel_size, stride, pad = conv_config
    np.random.seed(0)
    idt = DataType["UINT4"]

    in_feature_dim = 7
    in_chn = 16

    if depthwise is True:
        group = out_chn = in_chn
        conv_param_shape = [out_chn, 1, kernel_size, kernel_size]
    else:
        group = 1
        out_chn = 20
        conv_param_shape = [out_chn, in_chn, kernel_size, kernel_size]

    total_pad = 2 * pad
    out_feature_dim = compute_conv_output_dim(
        in_feature_dim, kernel_size, stride, total_pad
    )

    input_shape = [1, in_chn, in_feature_dim, in_feature_dim]
    output_shape = [1, out_chn, out_feature_dim, out_feature_dim]

    conv_weight_dt = DataType["UINT4"]

    conv_config = {}
    conv_config["dilations"] = [1, 1]
    conv_config["group"] = group
    conv_config["kernel_shape"] = [kernel_size, kernel_size]
    conv_config["pads"] = [pad, pad, pad, pad]
    conv_config["strides"] = [stride, stride]

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
        new_model = new_model.transform(to_hls.InferQuantizedStreamingFCLayer())
        fc_node = new_model.get_nodes_by_op_type("StreamingFCLayer_Batch")[0]
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
    if kernel_size == 1 and stride > 1 and pad == 0:
        assert new_model.graph.node[1].op_type == "DownSampler"
        if exec_mode == "rtlsim":
            node = new_model.get_nodes_by_op_type("DownSampler")[0]
            inst = getCustomOp(node)
            cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
            exp_cycles_dict = new_model.analysis(exp_cycles_per_layer)
            exp_cycles = exp_cycles_dict[node.name]
            assert np.isclose(exp_cycles, cycles_rtlsim, atol=11)
            assert exp_cycles != 0

    if pad == 1:
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
