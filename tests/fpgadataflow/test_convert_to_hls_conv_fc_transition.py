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

import finn.core.data_layout as DataLayout
import finn.core.onnx_exec as oxe
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
import finn.transformation.streamline.absorb as absorb
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.general.im2col import compute_conv_output_dim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.general import GiveUniqueNodeNames, RemoveUnusedTensors
from finn.transformation.infer_data_layouts import InferDataLayouts
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.reorder import MoveScalarLinearPastInvariants
from finn.util.basic import gen_finn_dt_tensor


def get_multithreshold_rand_params(channels, num_of_thres, seed=None):
    if seed is not None:
        np.random.seed(seed)
    steps = np.random.rand(channels, 1) * 30
    bias = np.random.rand(channels, 1) * -10
    thres = [np.arange(num_of_thres) for chn in range(channels)]
    thres = ((thres + bias) * steps).astype(np.float32)
    thres = np.round(thres)
    return thres


# conv_config: input_shape, kernel_shape, stride, pad
@pytest.mark.parametrize(
    "conv_config",
    [
        ((6, 6), (3, 3), (1, 1), (1, 1)),
        # TODO: enable 1d conv test cases
        # ((12, 1), (3, 1), (1, 1), (1, 0)),
        # ((1, 15), (1, 5), (1, 1), (0, 2)),
    ],
)
@pytest.mark.parametrize("depthwise", [False, True])
@pytest.mark.parametrize("use_reshape", [False, True])
@pytest.mark.vivado
@pytest.mark.slow
def test_convert_to_hls_conv_fc_transition(conv_config, depthwise, use_reshape):
    np.random.seed(0)
    idt = DataType["UINT4"]
    odt = DataType["UINT4"]
    conv_weight_dt = DataType["INT4"]
    fc_weight_dt = DataType["INT4"]

    input_shape, kernel_shape, stride, pad = conv_config
    kernel_size_h, kernel_size_w = kernel_shape
    input_size_h, input_size_w = input_shape
    stride_h, stride_w = stride
    pad_h, pad_w = pad

    in_chn = 4
    fc_filters = 16

    if depthwise is True:
        group = out_chn = in_chn
        conv_param_shape = [out_chn, 1, kernel_size_h, kernel_size_w]
    else:
        group = 1
        out_chn = 8
        conv_param_shape = [out_chn, in_chn, kernel_size_h, kernel_size_w]

    output_size_h = compute_conv_output_dim(
        input_size_h, kernel_size_h, stride_h, 2 * pad_h
    )
    output_size_w = compute_conv_output_dim(
        input_size_w, kernel_size_w, stride_w, 2 * pad_w
    )

    input_shape = [1, in_chn, input_size_h, input_size_w]
    fc_param_shape = [out_chn * output_size_h * output_size_w, fc_filters]
    output_shape = [1, fc_filters]

    conv_config = {}
    conv_config["dilations"] = [1, 1]
    conv_config["group"] = group
    conv_config["kernel_shape"] = [kernel_size_h, kernel_size_w]
    conv_config["pads"] = [pad_h, pad_w, pad_h, pad_w]
    conv_config["strides"] = [stride_h, stride_w]

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
        helper.make_tensor_value_info("reshape_shape", TensorProto.INT64, []),
    ]

    if use_reshape:
        flatten_node = helper.make_node(
            "Reshape", ["thres1_out", "reshape_shape"], ["flatten_out"]
        )
    else:
        flatten_node = helper.make_node(
            "Flatten", ["thres1_out"], ["flatten_out"], axis=1
        )

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
                ),
                flatten_node,
                helper.make_node(
                    "MatMul", ["flatten_out", "matmul_param"], ["matmul_out"]
                ),
                helper.make_node(
                    "MultiThreshold",
                    ["matmul_out", "thres2_param"],
                    ["global_out"],
                    domain="finn.custom_op.general",
                    out_dtype="UINT4",
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
    model.set_tensor_datatype("thres1_param", DataType["INT32"])
    model.set_tensor_datatype("thres2_param", DataType["INT32"])

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
    model.set_initializer("reshape_shape", np.array([1, -1], dtype=np.int64))

    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(InferDataLayouts())

    # streamlining
    new_model = model.transform(MoveScalarLinearPastInvariants())
    new_model = new_model.transform(Streamline())
    new_model = new_model.transform(LowerConvsToMatMul())
    new_model = new_model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
    new_model = new_model.transform(Streamline())
    new_model = new_model.transform(InferDataLayouts())
    new_model = new_model.transform(RemoveUnusedTensors())

    # convert_to_hls
    if depthwise is True:
        new_model = new_model.transform(to_hls.InferVVAU())
    new_model = new_model.transform(to_hls.InferQuantizedStreamingFCLayer())
    new_model = new_model.transform(to_hls.InferThresholdingLayer())
    new_model = new_model.transform(to_hls.InferConvInpGen())
    new_model = new_model.transform(to_hls.InferStreamingMaxPool())
    new_model = new_model.transform(RemoveCNVtoFCFlatten())
    new_model = new_model.transform(absorb.AbsorbConsecutiveTransposes())
    new_model = new_model.transform(GiveUniqueNodeNames())
    new_model = new_model.transform(InferDataLayouts())

    # prepare cppsim
    new_model = new_model.transform(PrepareCppSim())
    new_model = new_model.transform(CompileCppSim())
    new_model = new_model.transform(SetExecMode("cppsim"))

    # check for correct execution
    x = gen_finn_dt_tensor(idt, input_shape)
    inp_dict = {model.graph.input[0].name: x}
    assert oxe.compare_execution(model, new_model, inp_dict)

    num_transpose = len(new_model.get_nodes_by_op_type("Transpose"))
    num_flatten = len(new_model.get_nodes_by_op_type("Flatten"))
    num_reshape = len(new_model.get_nodes_by_op_type("Reshape"))

    # check if transpose->flatten was removed
    assert num_transpose == 1 and num_flatten == 0 and num_reshape == 0
