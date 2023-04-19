# Copyright (c) 2023, Xilinx
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
import os  # DEBUG
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.im2col import compute_conv_output_dim
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode


# conv_config:
# [pad_h_begin, pad_w_begin, pad_h_end, pad_w_end]
# [fm_dim_h, fm_dim_w]
# [kernel_size_h, kernel_size_w]
# [stride_h, stride_w]
# [dilation_h, dilation_w]
@pytest.mark.parametrize(
    "conv_config",
    [
        [[0, 0, 0, 0], [1, 12 + 2], [1, 3], [1, 1], [1, 1]],  # 1D
        # [[0, 1, 0, 1], [1, 12], [1, 3], [1, 1], [1, 1]], # 1D + pad
        # [[0, 0, 0, 0], [32+2, 1], [3, 1], [1, 1], [1, 1]], # 1D flipped
        # [[0, 0, 0, 0], [12+2, 12+2], [3, 3], [1, 1], [1, 1]], # 2D
    ],
)
@pytest.mark.parametrize("m", [1, 2, 4])
@pytest.mark.parametrize("depthwise", [False, True])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_convert_to_hls_conv_mmv(conv_config, m, depthwise):
    pad, in_feature_dim, kernel_size, stride, dilation = conv_config
    np.random.seed(0)
    idt = DataType["UINT4"]
    in_chn = 2
    in_feature_dim_h, in_feature_dim_w = in_feature_dim
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
        out_chn = 8
        conv_param_shape = [out_chn, in_chn, k_h, k_w]

    out_feature_dim_h = compute_conv_output_dim(
        in_feature_dim_h, k_h, stride_h, pad_h, dilation_h
    )
    out_feature_dim_w = compute_conv_output_dim(
        in_feature_dim_w, k_w, stride_w, pad_w, dilation_w
    )

    input_shape = [1, in_chn, in_feature_dim_h, in_feature_dim_w]
    output_shape = [1, out_chn, out_feature_dim_h, out_feature_dim_w]
    pool_output_shape = [1, out_chn, out_feature_dim_h, out_feature_dim_w // 2]

    conv_weight_dt = DataType["UINT4"]

    conv_config = {}
    conv_config["dilations"] = [dilation_h, dilation_w]
    conv_config["group"] = group
    conv_config["kernel_shape"] = [k_h, k_w]
    conv_config["pads"] = pad
    conv_config["strides"] = [stride_h, stride_w]

    pool_config = {}
    pool_config["strides"] = [2, 1]
    pool_config["pads"] = [0, 0, 0, 0]
    pool_config["kernel_shape"] = [2, 1]

    top_in = helper.make_tensor_value_info("top_in", TensorProto.FLOAT, input_shape)
    top_out = helper.make_tensor_value_info(
        "top_out", TensorProto.FLOAT, output_shape
    )  # pool_output_shape
    value_info = [
        helper.make_tensor_value_info("p1", TensorProto.FLOAT, conv_param_shape),
        # helper.make_tensor_value_info("conv_out", TensorProto.FLOAT, output_shape)
    ]

    modelproto = qonnx_make_model(
        helper.make_graph(
            name="conv_test",
            inputs=[top_in],
            outputs=[top_out],
            value_info=value_info,
            nodes=[
                helper.make_node(
                    "Conv", ["top_in", "p1"], ["top_out"], **conv_config
                ),  # "conv_out"
                # helper.make_node("MaxPool", ["conv_out"], ["top_out"], **pool_config)
            ],
        )
    )

    model = ModelWrapper(modelproto)
    model.set_tensor_datatype("top_in", idt)
    # model.set_tensor_datatype("conv_out", idt)
    model.set_tensor_datatype("top_out", idt)
    model.set_tensor_datatype("p1", conv_weight_dt)
    model.set_initializer("p1", gen_finn_dt_tensor(conv_weight_dt, conv_param_shape))

    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    new_model = model.transform(LowerConvsToMatMul())
    # new_model = new_model.transform(MakeMaxPoolNHWC())
    new_model = new_model.transform(to_hls.InferConvInpGen(use_rtl_variant=True))
    # new_model = new_model.transform(to_hls.InferStreamingMaxPool())
    if depthwise is True:
        new_model = new_model.transform(to_hls.InferVectorVectorActivation())
        vvau_node = new_model.get_nodes_by_op_type("VectorVectorActivation")[0]
        vvau_inst = getCustomOp(vvau_node)
        vvau_inst.set_nodeattr("PE", out_chn)  # max
        vvau_inst.set_nodeattr("SIMD", k_h * k_w)  # max
        vvau_inst.set_nodeattr("M", m)
    else:
        new_model = new_model.transform(to_hls.InferQuantizedMatrixVectorActivation())
        mvau_node = new_model.get_nodes_by_op_type("MatrixVectorActivation")[0]
        mvau_inst = getCustomOp(mvau_node)
        mvau_inst.set_nodeattr("PE", out_chn)  # max
        mvau_inst.set_nodeattr("SIMD", in_chn * k_h * k_w)  # max
        mvau_inst.set_nodeattr("M", m)
        # DEBUG
        # fc_inst.set_nodeattr("rtlsim_trace", "/workspace/finn/test_fc.vcd")

    # configure parallelism of auxiliary nodes
    # pad_node = new_model.get_nodes_by_op_type("FMPadding_rtl")[0]
    # pad_inst = getCustomOp(pad_node)
    # pad_inst.set_nodeattr("M", M)

    # pool_node = new_model.get_nodes_by_op_type("StreamingMaxPool_Batch")[0]
    # pool_inst = getCustomOp(pool_node)
    # pool_inst.set_nodeattr("M", M)

    swg_node = new_model.get_nodes_by_op_type("ConvolutionInputGenerator_rtl")[0]
    swg_inst = getCustomOp(swg_node)
    swg_inst.set_nodeattr("parallel_window", 1)
    swg_inst.set_nodeattr("M", m)

    new_model = new_model.transform(GiveUniqueNodeNames())
    new_model = new_model.transform(InferShapes())
    new_model = new_model.transform(InferDataTypes())

    # DEBUG
    model.save("MMV_test_before.onnx")
    new_model.save("MMV_test_after.onnx")

    # Stitched rtlsim
    # model = model.transform(InsertFIFO(create_shallow_fifos=True))
    new_model = new_model.transform(SetExecMode("rtlsim"))
    new_model = new_model.transform(GiveUniqueNodeNames())
    new_model = new_model.transform(PrepareIP("xc7z020clg400-1", 5))
    new_model = new_model.transform(HLSSynthIP())
    new_model = new_model.transform(PrepareRTLSim())
    # Stitched rtlsim
    # new_model = new_model.transform(CreateStitchedIP("xc7z020clg400-1", 5))
    # new_model.set_metadata_prop("exec_mode", "rtlsim")

    x = gen_finn_dt_tensor(idt, input_shape)
    inp_dict = {model.graph.input[0].name: x}
    # assert oxe.compare_execution(model, new_model, inp_dict)
    y_expected = oxe.execute_onnx(model, inp_dict)["top_out"]
    out_dict = oxe.execute_onnx(
        new_model,
        inp_dict,
        return_full_exec_context=True,
        start_node=None,
        end_node=None,
    )
    y_produced = out_dict["top_out"]

    # DEBUG
    f_debug = open(os.path.join("/home/felixj/WD/finn", "mmv_debug_result.log"), "w")
    f_debug.write("expected:\n")
    f_debug.write("%s\n" % str(y_expected))
    f_debug.write("produced:\n")
    f_debug.write("%s\n" % str(y_produced))
    f_debug.write("full out dict:\n")
    f_debug.write("%s\n" % str(out_dict))
    f_debug.close()

    assert (y_produced == y_expected).all()

    # TODO: check rtlsim cycles
