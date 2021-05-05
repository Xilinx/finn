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

from onnx import TensorProto, helper
import numpy as np
import pytest

from finn.core.datatype import DataType
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.general import GiveUniqueNodeNames
from finn.transformation.lower_convs_to_matmul import LowerConvsToMatMul

from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
import finn.core.onnx_exec as oxe
from finn.core.modelwrapper import ModelWrapper
from finn.util.basic import gen_finn_dt_tensor
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls

from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.custom_op.general.im2col import compute_conv_output_dim
from finn.custom_op.registry import getCustomOp
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer

import finn.transformation.streamline.absorb as absorb
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.general import (
    ApplyConfig,
    GiveReadableTensorNames,
    RemoveUnusedTensors,
    RemoveStaticGraphInputs,
)
from finn.transformation.streamline import Streamline
from finn.transformation.infer_data_layouts import InferDataLayouts
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from finn.transformation.streamline.reorder import (
    MakeMaxPoolNHWC,
    MoveScalarLinearPastInvariants,
)

import finn.core.data_layout as DataLayout


def get_multithreshold_rand_params(channels, num_of_thres, seed=None):
    if seed is not None:
        np.random.seed(seed)
    steps = np.random.rand(channels, 1) * 20
    bias = np.random.rand(channels, 1) * -10
    thres = [np.arange(num_of_thres) for chn in range(channels)]
    thres = ((thres + bias) * steps).astype(np.float32)
    thres = np.round(thres)
    return thres


# conv_config  kernel_size,stride, pad


# @pytest.mark.parametrize(
#    "conv_config", [(1, 2, 0), (1, 3, 0), (3, 2, 1), (3, 1, 0), (3, 1, 1), (5, 2, 1)]
# )
# @pytest.mark.parametrize("depthwise", [False, True])
# @pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.parametrize("conv_config", [(3, 1, 1)])
@pytest.mark.parametrize("depthwise", [False])
@pytest.mark.parametrize("exec_mode", ["cppsim"])
@pytest.mark.slow
@pytest.mark.vivado
def test_convert_to_hls_conv_fc_transition(conv_config, depthwise, exec_mode):
    kernel_size, stride, pad = conv_config
    np.random.seed(0)
    idt = DataType.UINT4
    odt = DataType.UINT4

    in_feature_dim = 2
    in_chn = 4
    fc_filters = 8

    if depthwise is True:
        group = out_chn = in_chn
        conv_param_shape = [out_chn, 1, kernel_size, kernel_size]
    else:
        group = 1
        out_chn = 4
        conv_param_shape = [out_chn, in_chn, kernel_size, kernel_size]

    total_pad = 2 * pad
    out_feature_dim = compute_conv_output_dim(
        in_feature_dim, kernel_size, stride, total_pad
    )

    input_shape = [1, in_chn, in_feature_dim, in_feature_dim]
    conv_output_shape = [1, out_chn, out_feature_dim, out_feature_dim]
    output_shape = [1, fc_filters]

    fc_param_shape = [out_chn * out_feature_dim * out_feature_dim, fc_filters]

    conv_weight_dt = DataType.INT4
    fc_weight_dt = DataType.INT4

    conv_config = {}
    conv_config["dilations"] = [1, 1]
    conv_config["group"] = group
    conv_config["kernel_shape"] = [kernel_size, kernel_size]
    conv_config["pads"] = [pad, pad, pad, pad]
    conv_config["strides"] = [stride, stride]

    global_in = helper.make_tensor_value_info(
        "global_in", TensorProto.FLOAT, input_shape
    )
    global_out = helper.make_tensor_value_info(
        "global_out", TensorProto.FLOAT, output_shape
    )
    value_info = [
        helper.make_tensor_value_info(
            "conv_param", TensorProto.FLOAT, conv_param_shape
        ),
        helper.make_tensor_value_info("thres1_param", TensorProto.FLOAT, (out_chn, 15)),
        helper.make_tensor_value_info(
            "matmul_param", TensorProto.FLOAT, fc_param_shape
        ),
        helper.make_tensor_value_info(
            "thres2_param", TensorProto.FLOAT, (fc_filters, 15)
        ),
    ]

    modelproto = helper.make_model(
        helper.make_graph(
            name="test",
            inputs=[global_in],
            outputs=[global_out],
            value_info=value_info,
            nodes=[
                helper.make_node(
                    "Conv", ["global_in", "conv_param"], ["conv_out"], **conv_config
                ),
                helper.make_node(
                    "MultiThreshold",
                    ["conv_out", "thres1_param"],
                    ["thres1_out"],
                    domain="finn.custom_op.general",
                    out_dtype="UINT4",
                    # out_bias=-7,
                    # out_scale=1.0
                ),
                helper.make_node("Flatten", ["thres1_out"], ["flatten_out"], axis=1),
                helper.make_node(
                    "MatMul", ["flatten_out", "matmul_param"], ["matmul_out"]
                ),
                helper.make_node(
                    "MultiThreshold",
                    ["matmul_out", "thres2_param"],
                    ["global_out"],
                    domain="finn.custom_op.general",
                    out_dtype="UINT4",
                    # out_bias=-7,
                    # out_scale=1.0
                ),
            ],
        )
    )

    model = ModelWrapper(modelproto)
    model.set_tensor_datatype("global_in", idt)
    model.set_tensor_layout("global_in", DataLayout.NCHW)
    model.set_tensor_datatype("global_out", odt)
    model.set_tensor_datatype("conv_param", conv_weight_dt)
    model.set_tensor_datatype("matmul_param", fc_weight_dt)
    model.set_tensor_datatype("thres1_param", DataType.INT32)
    model.set_tensor_datatype("thres2_param", DataType.INT32)
    model.set_tensor_datatype(
        "flatten_out", DataType.UINT4
    )  # TODO: not inferred automatically (FLOAT32)
    model.set_initializer(
        "conv_param", gen_finn_dt_tensor(conv_weight_dt, conv_param_shape)
    )
    model.set_initializer(
        "thres1_param", get_multithreshold_rand_params(out_chn, 15, seed=0)
    )
    model.set_initializer(
        "thres2_param", get_multithreshold_rand_params(fc_filters, 15, seed=0)
    )
    model.set_initializer(
        "matmul_param", gen_finn_dt_tensor(fc_weight_dt, fc_param_shape)
    )

    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(InferDataLayouts())

    model.save("testmodel_in.onnx")

    x = gen_finn_dt_tensor(idt, input_shape)
    inp_dict = {model.graph.input[0].name: x}
    output = oxe.execute_onnx(model, inp_dict)
    print(output)

    # streamlining step
    model = model.transform(MoveScalarLinearPastInvariants())
    model = model.transform(Streamline())
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(MakeMaxPoolNHWC())
    model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
    model = model.transform(Streamline())

    model = model.transform(InferDataLayouts())
    model = model.transform(RemoveUnusedTensors())

    model.save("testmodel_streamlined.onnx")

    output = oxe.execute_onnx(model, inp_dict)
    print(output)

    # convert_to_hls step
    model = model.transform(to_hls.InferQuantizedStreamingFCLayer())
    model = model.transform(to_hls.InferThresholdingLayer())
    model = model.transform(to_hls.InferConvInpGen())
    model = model.transform(to_hls.InferStreamingMaxPool())
    model = model.transform(RemoveCNVtoFCFlatten())
    model = model.transform(absorb.AbsorbConsecutiveTransposes())

    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(InferDataLayouts())

    if exec_mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))
    elif exec_mode == "rtlsim":
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareIP("xc7z020clg400-1", 5))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())
    else:
        raise Exception("Unknown exec_mode")

    model.save("testmodel_hls.onnx")

    output = oxe.execute_onnx(model, inp_dict)
    print(output)

    model_orig = ModelWrapper("testmodel_in.onnx")
    model_hls = ModelWrapper("testmodel_hls.onnx")

    assert oxe.compare_execution(model_orig, model_hls, inp_dict)


"""
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
 """
