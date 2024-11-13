# Copyright (c) 2022 Xilinx, Inc.
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
# * Neither the name of Xilinx nor the names of its
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

import copy
import json
import numpy as np
import os
import shutil
import torch
from brevitas.export import export_qonnx
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.im2col import compute_conv_output_dim
from qonnx.custom_op.general.maxpoolnhwc import compute_pool_output_dim
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import (
    GiveRandomTensorNames,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    GiveUniqueParameterTensors,
)
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.builder.build_dataflow_steps import step_set_fifo_depths
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferStreamingMaxPool
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import make_build_dir
from finn.util.test import get_trained_network_and_ishape


def generate_random_threshold_values(
    data_type, num_input_channels, num_steps, narrow=False, per_tensor=False
):
    if per_tensor:
        num_input_channels = 1
    if narrow:
        num_steps -= 1

    return np.random.randint(
        data_type.min(),
        data_type.max() + 1,
        (num_input_channels, num_steps),
    ).astype(np.float32)


def sort_thresholds_increasing(thresholds):
    return np.sort(thresholds, axis=1)


def compare_two_chr_funcs(a, b, relaxation):
    # relaxation determines how much leeway we allow for the
    # analytical implementation to be off from RTL ground truth
    equal = True
    for inp in range(len(a)):
        for i in range(len(a[inp])):
            if (a[inp][i] > (b[inp][i] + relaxation)) or (a[inp][i] < (b[inp][i] - relaxation)):
                equal = False
    return equal


def make_single_fmpadding_modelwrapper(impl_style, idim, padding, num_ch, simd, idt):
    pad_h = padding[0] + padding[2]
    pad_w = padding[1] + padding[3]
    idim_h, idim_w = idim

    assert pad_h > 0 or pad_w > 0, "Output dim should be greater than input dim"
    odim_h = idim_h + pad_h
    odim_w = idim_w + pad_w

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, idim_h, idim_w, num_ch])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, odim_h, odim_w, num_ch])

    FMPadding = helper.make_node(
        "FMPadding",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        ImgDim=idim,
        Padding=padding,
        NumChannels=num_ch,
        inputDataType=str(idt.name),
        numInputVectors=1,
        SIMD=simd,
        preferred_impl_style=impl_style,
    )

    graph = helper.make_graph(
        nodes=[FMPadding], name="fmpadding_graph", inputs=[inp], outputs=[outp]
    )

    model = qonnx_make_model(graph, producer_name="fmpadding-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", idt)

    return model


def make_single_fclayer_modelwrapper(W, pe, simd, wdt, idt, odt, T=None, tdt=None):
    mw = W.shape[0]
    mh = W.shape[1]
    assert mh % pe == 0
    assert mw % simd == 0

    # there are two ways to implement bipolar weights and inputs for
    # MatrixVectorActivation:
    # - specify their datatypes as such
    # - specify their datatypes as BINARY as use binaryXnorMode
    if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
        # we'll internally convert weights/inputs to binary and specify the
        # datatypes as such, and also set the binaryXnorMode attribute to 1
        export_wdt = DataType["BINARY"]
        export_idt = DataType["BINARY"]
        binary_xnor_mode = 1
    else:
        export_wdt = wdt
        export_idt = idt
        binary_xnor_mode = 0

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, mw])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, mh])
    if T is not None:
        no_act = 0
        node_inp_list = ["inp", "weights", "thresh"]
        if odt == DataType["BIPOLAR"]:
            actval = 0
        else:
            actval = odt.min()
    else:
        # no thresholds
        node_inp_list = ["inp", "weights"]
        actval = 0
        no_act = 1
    FCLayer_node = helper.make_node(
        "MVAU",
        node_inp_list,
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        MW=mw,
        MH=mh,
        SIMD=simd,
        PE=pe,
        inputDataType=export_idt.name,
        weightDataType=export_wdt.name,
        outputDataType=odt.name,
        ActVal=actval,
        binaryXnorMode=binary_xnor_mode,
        noActivation=no_act,
    )
    graph = helper.make_graph(
        nodes=[FCLayer_node], name="fclayer_graph", inputs=[inp], outputs=[outp]
    )

    model = qonnx_make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("weights", wdt)
    if binary_xnor_mode:
        # convert bipolar to binary
        model.set_initializer("weights", (W + 1) / 2)
    else:
        model.set_initializer("weights", W)
    if T is not None:
        model.set_tensor_datatype("thresh", tdt)
        model.set_initializer("thresh", T)
    return model


def make_labelselect_modelwrapper(labels, pe, k, idt, impl_style):
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, labels])
    outp = helper.make_tensor_value_info("outp", TensorProto.INT64, [1, k])

    labelselect_node = helper.make_node(
        "LabelSelect",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        Labels=labels,
        PE=pe,
        K=k,
        inputDataType=idt.name,
        preferred_impl_style=impl_style,
    )
    graph = helper.make_graph(
        nodes=[labelselect_node],
        name="graph",
        inputs=[inp],
        outputs=[outp],
    )

    model = qonnx_make_model(graph, producer_name="thresholding-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    odt = DataType.get_smallest_possible(labels - 1)
    model.set_tensor_datatype("outp", odt)

    return model


def _make_single_vvau_modelwrapper(
    W,
    pe,
    simd,
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
    mem_mode="internal_embedded",
    impl_style="rtl",
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
        if odt == DataType["BIPOLAR"]:
            actval = 0
        else:
            actval = odt.min()
    else:
        no_act = 1
        node_inp_list = ["inp", "weights"]
        actval = 0

    VVAU_node = helper.make_node(
        "VVAU",
        node_inp_list,
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        PE=pe,
        SIMD=simd,
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
        impl_style=impl_style,
    )

    graph = helper.make_graph(nodes=[VVAU_node], name="vvau_graph", inputs=[inp], outputs=[outp])

    model = qonnx_make_model(graph, producer_name="vvau-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("weights", wdt)

    model.set_initializer("weights", W)
    model.set_tensor_shape("weights", (channels, 1, k_h, k_w))

    if T is not None:
        model.set_tensor_datatype("thresh", tdt)
        model.set_initializer("thresh", T)

    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    return model


def make_single_dw_conv_modelwrapper(conv_config, idt, wdt):
    kernel_size, in_feature_dim, in_chn = conv_config
    stride = 1
    pad = 0

    out_feature_dim = compute_conv_output_dim(in_feature_dim, kernel_size, stride, pad)
    group = out_chn = in_chn

    conv_param_shape = [out_chn, 1, kernel_size, kernel_size]
    input_shape = [1, in_chn, in_feature_dim, in_feature_dim]
    output_shape = [1, out_chn, out_feature_dim, out_feature_dim]

    conv_config = {}
    conv_config["dilations"] = [1, 1]
    conv_config["group"] = group
    conv_config["kernel_shape"] = [kernel_size, kernel_size]
    conv_config["pads"] = [pad, pad, pad, pad]
    conv_config["strides"] = [stride, stride]

    ifm = helper.make_tensor_value_info("ifm", TensorProto.FLOAT, input_shape)
    ofm = helper.make_tensor_value_info("ofm", TensorProto.FLOAT, output_shape)
    weights = [helper.make_tensor_value_info("weights", TensorProto.FLOAT, conv_param_shape)]

    modelproto = qonnx_make_model(
        helper.make_graph(
            name="conv_test",
            inputs=[ifm],
            outputs=[ofm],
            value_info=weights,
            nodes=[helper.make_node("Conv", ["ifm", "weights"], ["ofm"], **conv_config)],
        )
    )

    model = ModelWrapper(modelproto)
    model.set_tensor_datatype("ifm", idt)
    model.set_tensor_datatype("weights", wdt)
    model.set_initializer("weights", gen_finn_dt_tensor(wdt, conv_param_shape))

    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    return model


def make_conv_building_block(ifm_dim, ch, kernel_size, simd, pe, parallel_window=0):
    # hardcoded parameters
    idt = DataType["UINT4"]
    wdt = DataType["UINT4"]
    odt = DataType["UINT4"]
    tdt = DataType["UINT32"]
    stride = 1
    in_ch = out_ch = ch  # input channel = output channel for stacking
    pad = int(np.floor(kernel_size / 2))  # pad so that input dim = output dim for stacking

    total_pad = 2 * pad
    out_feature_dim = compute_conv_output_dim(ifm_dim, kernel_size, stride, total_pad)
    weights_shape = [in_ch * kernel_size * kernel_size, out_ch]
    thresholds_shape = [1, odt.get_num_possible_values() - 1]
    input_shape = [1, ifm_dim, ifm_dim, in_ch]
    padding_out_shape = [1, ifm_dim + total_pad, ifm_dim + total_pad, in_ch]
    inpgen_out_shape = [1, out_feature_dim, out_feature_dim, in_ch * kernel_size * kernel_size]
    output_shape = [1, out_feature_dim, out_feature_dim, out_ch]

    padding_config = {}
    padding_config["domain"] = "finn.custom_op.fpgadataflow.rtl"
    padding_config["backend"] = "fpgadataflow"
    padding_config["ImgDim"] = [ifm_dim, ifm_dim]
    padding_config["NumChannels"] = in_ch
    padding_config["SIMD"] = simd
    padding_config["Padding"] = [pad, pad, pad, pad]
    padding_config["inputDataType"] = idt.name

    inpgen_config = {}
    inpgen_config["domain"] = "finn.custom_op.fpgadataflow.rtl"
    inpgen_config["backend"] = "fpgadataflow"
    inpgen_config["ConvKernelDim"] = [kernel_size, kernel_size]
    inpgen_config["IFMChannels"] = in_ch
    inpgen_config["IFMDim"] = [ifm_dim + total_pad, ifm_dim + total_pad]
    inpgen_config["OFMDim"] = [ifm_dim, ifm_dim]
    inpgen_config["inputDataType"] = idt.name
    inpgen_config["outputDataType"] = idt.name
    inpgen_config["SIMD"] = simd
    inpgen_config["parallel_window"] = parallel_window
    inpgen_config["Stride"] = [stride, stride]
    inpgen_config["Dilation"] = [1, 1]

    mvau_config = {}
    mvau_config["domain"] = "finn.custom_op.fpgadataflow.hls"
    mvau_config["backend"] = "fpgadataflow"
    mvau_config["numInputVectors"] = [1, ifm_dim, ifm_dim]
    mvau_config["MW"] = in_ch * kernel_size * kernel_size
    mvau_config["MH"] = in_ch
    mvau_config["SIMD"] = simd if parallel_window == 0 else simd * kernel_size * kernel_size
    mvau_config["PE"] = pe
    mvau_config["resType"] = "lut"
    mvau_config["inputDataType"] = idt.name
    mvau_config["weightDataType"] = wdt.name
    mvau_config["outputDataType"] = odt.name

    top_in = helper.make_tensor_value_info("top_in", TensorProto.FLOAT, input_shape)
    top_out = helper.make_tensor_value_info("top_out", TensorProto.FLOAT, output_shape)
    value_info = [
        helper.make_tensor_value_info("weights", TensorProto.FLOAT, weights_shape),
        helper.make_tensor_value_info("thresholds", TensorProto.FLOAT, thresholds_shape),
        helper.make_tensor_value_info("padding_out", TensorProto.FLOAT, padding_out_shape),
        helper.make_tensor_value_info("inpgen_out", TensorProto.FLOAT, inpgen_out_shape),
    ]

    modelproto = qonnx_make_model(
        helper.make_graph(
            name="building_block",
            inputs=[top_in],
            outputs=[top_out],
            value_info=value_info,
            nodes=[
                helper.make_node("FMPadding_rtl", ["top_in"], ["padding_out"], **padding_config),
                helper.make_node(
                    "ConvolutionInputGenerator_rtl",
                    ["padding_out"],
                    ["inpgen_out"],
                    **inpgen_config,
                ),
                helper.make_node(
                    "MVAU_hls", ["inpgen_out", "weights", "thresholds"], ["top_out"], **mvau_config
                ),
            ],
        )
    )

    model = ModelWrapper(modelproto)
    model.set_tensor_datatype("top_in", idt)
    model.set_tensor_layout("top_in", ["N", "H", "W", "C"])
    model.set_tensor_datatype("top_out", odt)
    model.set_tensor_datatype("weights", wdt)
    model.set_tensor_datatype("thresholds", tdt)

    weights = gen_finn_dt_tensor(wdt, weights_shape)
    # TODO: thresholds are all the same
    thresholds = generate_random_threshold_values(
        tdt, out_ch, odt.get_num_possible_values() - 1, False, True
    )
    thresholds = sort_thresholds_increasing(thresholds)

    model.set_initializer("weights", weights)
    model.set_initializer("thresholds", thresholds)

    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    return model


def combine_blocks(lb, rb, ifm_dim, ch, pe):
    # assumes left branch (lb) and right branch (rb) each have a single (dynamic) input/output with the same shape
    # to avoid mix-ups, start by giving all tensors random names
    lb = lb.transform(GiveRandomTensorNames())
    rb = rb.transform(GiveRandomTensorNames())
    # erase all node names to avoid conflict
    for n in lb.graph.node:
        n.name = ""
    for n in rb.graph.node:
        n.name = ""

    lb_input = lb.graph.input[0]
    lb_output = lb.graph.output[0]
    rb_input = rb.graph.input[0]
    rb_output = rb.graph.output[0]

    top_in = helper.make_tensor_value_info("top_in", TensorProto.FLOAT, [1, ifm_dim, ifm_dim, ch])
    top_out = helper.make_tensor_value_info("top_out", TensorProto.FLOAT, [1, ifm_dim, ifm_dim, ch])

    dup_config = {}
    dup_config["domain"] = "finn.custom_op.fpgadataflow.hls"
    dup_config["backend"] = "fpgadataflow"
    dup_config["numInputVectors"] = [1, ifm_dim, ifm_dim]
    dup_config["NumChannels"] = ch
    dup_config["PE"] = pe
    dup_config["NumOutputStreams"] = 2
    dup_config["inputDataType"] = lb.get_tensor_datatype(lb_input.name).name

    add_config = {}
    add_config["domain"] = "finn.custom_op.fpgadataflow.hls"
    add_config["backend"] = "fpgadataflow"
    add_config["numInputVectors"] = [1, ifm_dim, ifm_dim]
    add_config["NumChannels"] = ch
    add_config["PE"] = pe
    add_config["inputDataType"] = lb.get_tensor_datatype(lb_output.name).name

    nodes_lb = [node for node in lb.graph.node]
    nodes_rb = [node for node in rb.graph.node]
    nodes_new = (
        nodes_lb
        + nodes_rb
        + [
            helper.make_node(
                "DuplicateStreams_hls", ["top_in"], [lb_input.name, rb_input.name], **dup_config
            ),
            helper.make_node(
                "AddStreams_hls", [lb_output.name, rb_output.name], ["top_out"], **add_config
            ),
        ]
    )

    value_info_lb = [x for x in lb.graph.value_info]
    value_info_rb = [x for x in rb.graph.value_info]
    value_info_new = value_info_lb + value_info_rb + [lb_input, lb_output, rb_input, rb_output]

    initializer_lb = [x for x in lb.graph.initializer]
    initializer_rb = [x for x in rb.graph.initializer]
    initializer_new = initializer_lb + initializer_rb
    modelproto = qonnx_make_model(
        helper.make_graph(
            name="branching_model",
            inputs=[top_in],
            outputs=[top_out],
            value_info=value_info_new,
            nodes=nodes_new,
        )
    )

    model = ModelWrapper(modelproto)
    model.set_tensor_datatype("top_in", lb.get_tensor_datatype(lb_input.name))
    model.set_tensor_layout("top_in", lb.get_tensor_layout(lb_input.name))
    for i in initializer_new:
        model.graph.initializer.append(i)

    # tidy-up
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(InferDataLayouts())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveUniqueParameterTensors())
    model = model.transform(GiveReadableTensorNames())
    return model


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


def make_single_maxpoolnhwc_modelwrapper(k, ifm_ch, ifm_dim, ofm_dim, idt, ceil_mode):
    k_h, k_w = k
    ifm_dim_h, ifm_dim_w = ifm_dim
    ofm_dim_h, ofm_dim_w = ofm_dim
    odt = idt
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, ifm_dim_h, ifm_dim_w, ifm_ch])
    outp = helper.make_tensor_value_info(
        "outp", TensorProto.FLOAT, [1, ofm_dim_h, ofm_dim_w, ifm_ch]
    )

    mp_node = helper.make_node(
        "MaxPoolNHWC",
        ["inp"],
        ["outp"],
        domain="qonnx.custom_op.general",
        kernel_shape=[k_h, k_w],
        strides=[k_h, k_w],
        ceil_mode=ceil_mode,
        pads=[0, 0, 0, 0],
    )
    graph = helper.make_graph(nodes=[mp_node], name="mp_graph", inputs=[inp], outputs=[outp])

    model = qonnx_make_model(graph, producer_name="mp-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)

    return model


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


def make_channelwise_modelwrapper(C, pe, idt, odt, pdt, func, vecs):
    NumChannels = C.shape[0]

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, vecs + [NumChannels])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, vecs + [NumChannels])

    node_inp_list = ["inp", "const"]

    node = helper.make_node(
        "ChannelwiseOp",
        node_inp_list,
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        NumChannels=NumChannels,
        Func=func,
        PE=pe,
        inputDataType=idt.name,
        outputDataType=odt.name,
        paramDataType=pdt.name,
        numInputVectors=vecs,
        preferred_impl_style="hls",
    )
    graph = helper.make_graph(nodes=[node], name="graph", inputs=[inp], outputs=[outp])

    model = qonnx_make_model(graph, producer_name="model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)

    model.set_tensor_datatype("const", idt)
    model.set_initializer("const", C)
    return model


def make_single_dwc_modelwrapper(in_shape, out_shape, inWidth, outWidth, finn_dtype, impl_style):
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, in_shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, out_shape)

    optype = "StreamingDataWidthConverter"

    DWC_node = helper.make_node(
        optype,
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        in_shape=in_shape,
        out_shape=out_shape,
        inWidth=inWidth,
        outWidth=outWidth,
        preferred_impl_style=impl_style,
        generalized_variant=True,
        dataType=str(finn_dtype.name),
    )

    graph = helper.make_graph(nodes=[DWC_node], name="dwc_graph", inputs=[inp], outputs=[outp])

    model = qonnx_make_model(graph, producer_name="dwc-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", finn_dtype)
    model.set_tensor_datatype("outp", finn_dtype)

    return model


def make_single_thresholding_modelwrapper(impl_style, T, idt, odt, actval, n_inp_vecs, num_ch):
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, n_inp_vecs + [num_ch])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, n_inp_vecs + [num_ch])

    node_inp_list = ["inp", "thresh"]

    Thresholding_node = helper.make_node(
        "Thresholding",
        node_inp_list,
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        NumChannels=num_ch,
        numSteps=T.shape[1],
        inputDataType=idt.name,
        weightDataType=idt.name,  # will be set by MinimizeAccumulatorWidth
        outputDataType=odt.name,
        ActVal=actval,
        numInputVectors=n_inp_vecs,
        preferred_impl_style=impl_style,
    )
    graph = helper.make_graph(
        nodes=[Thresholding_node],
        name="thresholding_graph",
        inputs=[inp],
        outputs=[outp],
    )

    model = qonnx_make_model(graph, producer_name="thresholding-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)

    model.set_tensor_datatype("thresh", idt)
    model.set_initializer("thresh", T)
    return model


def fetch_test_model(topology, wbits=2, abits=2):
    tmp_output_dir = make_build_dir("build_fifosizing_%s_" % topology)
    (model, ishape) = get_trained_network_and_ishape(topology, wbits, abits)
    chkpt_name = tmp_output_dir + "/model.onnx"
    export_qonnx(model, torch.randn(ishape), chkpt_name)
    return tmp_output_dir


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.fpgadataflow
@pytest.mark.parametrize(
    "method", ["largefifo_rtlsim_python", "largefifo_rtlsim_cpp", "characterize"]
)
@pytest.mark.parametrize("topology", ["tfc", "cnv"])
def test_fifosizing_linear(method, topology):
    force_python_rtlsim = "python" in method
    method_key = "largefifo_rtlsim" if "largefifo_rtlsim" in method else "characterize"
    tmp_output_dir = fetch_test_model(topology)
    cfg = build_cfg.DataflowBuildConfig(
        output_dir=tmp_output_dir,
        auto_fifo_depths=True,
        auto_fifo_strategy=method_key,
        target_fps=10000 if topology == "tfc" else 1000,
        force_python_rtlsim=force_python_rtlsim,
        synth_clk_period_ns=10.0,
        board="Pynq-Z1",
        rtlsim_batch_size=100 if topology == "tfc" else 2,
        shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.STITCHED_IP,
            build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
        ],
    )
    build.build_dataflow_cfg(tmp_output_dir + "/model.onnx", cfg)
    with open(tmp_output_dir + "/report/estimate_network_performance.json") as f:
        est_data = json.load(f)
    with open(tmp_output_dir + "/report/rtlsim_performance.json") as f:
        sim_data = json.load(f)
    assert (
        float(sim_data["stable_throughput[images/s]"]) / float(est_data["estimated_throughput_fps"])
        > 0.9
    )
    # now run the same build using the generated folding and FIFO config
    tmp_output_dir_cmp = fetch_test_model(topology)
    cfg_cmp = cfg
    cfg_cmp.output_dir = tmp_output_dir_cmp
    cfg_cmp.auto_fifo_depths = False
    cfg_cmp.target_fps = None
    cfg_cmp.generate_outputs = [build_cfg.DataflowOutputType.STITCHED_IP]
    cfg_cmp.folding_config_file = tmp_output_dir + "/final_hw_config.json"
    build.build_dataflow_cfg(tmp_output_dir_cmp + "/model.onnx", cfg_cmp)

    model0 = ModelWrapper(tmp_output_dir + "/intermediate_models/step_create_stitched_ip.onnx")
    model1 = ModelWrapper(tmp_output_dir_cmp + "/intermediate_models/step_create_stitched_ip.onnx")

    assert len(model0.graph.node) == len(model1.graph.node)
    for i in range(len(model0.graph.node)):
        node0 = model0.graph.node[i]
        node1 = model1.graph.node[i]
        assert node0.op_type == node1.op_type
        if node0.op_type == "StreamingFIFO":
            node0_inst = getCustomOp(node0)
            node1_inst = getCustomOp(node1)
            assert node0_inst.get_nodeattr("depth") == node1_inst.get_nodeattr("depth")

    shutil.rmtree(tmp_output_dir)
    shutil.rmtree(tmp_output_dir_cmp)


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.fpgadataflow
@pytest.mark.parametrize("strategy", ["rtlsim"])  # rtlsim #analytical
@pytest.mark.parametrize("lb_num_layers", [1])
@pytest.mark.parametrize("rb_num_layers", [4])
def test_fifosizing_nonlinear(strategy, lb_num_layers, rb_num_layers):
    np.random.seed(0)
    tmp_output_dir = make_build_dir(
        "build_fifosizing_nonlinear_%s_%s" % (lb_num_layers, rb_num_layers)
    )

    rtlsim_n = 10

    dim = 16
    ch = 4

    lb = None
    for i in range(lb_num_layers):
        new_block = make_conv_building_block(
            dim, ch, kernel_size=3, simd=4, pe=4, parallel_window=1
        )
        lb = new_block if lb is None else lb.transform(MergeONNXModels(new_block))
    lb.save(tmp_output_dir + "/lb.onnx")

    rb = None
    for i in range(rb_num_layers):
        new_block = make_conv_building_block(
            dim, ch, kernel_size=3, simd=4, pe=4, parallel_window=1
        )
        rb = new_block if rb is None else rb.transform(MergeONNXModels(new_block))
    rb.save(tmp_output_dir + "/rb.onnx")

    model = combine_blocks(lb, rb, dim, ch, pe=4)
    model.save(tmp_output_dir + "/model.onnx")

    cfg = build_cfg.DataflowBuildConfig(
        output_dir=tmp_output_dir,
        # only works with characterization-based FIFO-sizing
        auto_fifo_depths=True,
        auto_fifo_strategy="characterize",
        characteristic_function_strategy=strategy,
        split_large_fifos=False,
        # manual folding
        target_fps=None,
        # general rtlsim settings
        force_python_rtlsim=False,
        rtlsim_batch_size=rtlsim_n,
        synth_clk_period_ns=10.0,
        board="Pynq-Z1",
        shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.STITCHED_IP,
            build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
        ],
    )

    build.build_dataflow_cfg(tmp_output_dir + "/model.onnx", cfg)

    with open(tmp_output_dir + "/report/estimate_network_performance.json") as f:
        est_data = json.load(f)
    with open(tmp_output_dir + "/report/rtlsim_performance.json") as f:
        sim_data = json.load(f)

    # check for deadlock
    model_final = ModelWrapper(tmp_output_dir + "/intermediate_models/step_create_stitched_ip.onnx")
    first_node = getCustomOp(model_final.find_consumer(model_final.graph.input[0].name))
    last_node = getCustomOp(model_final.find_producer(model_final.graph.output[0].name))
    input_txns_expected = np.prod(first_node.get_folded_input_shape()[:-1]) * rtlsim_n
    output_txns_expected = np.prod(last_node.get_folded_output_shape()[:-1]) * rtlsim_n
    assert sim_data["N_IN_TXNS"] == input_txns_expected
    assert sim_data["N_OUT_TXNS"] == output_txns_expected

    # check rtlsim throughput
    # TODO: how to determine N? Take throughput or stable_throughput?
    # sim_data["stable_throughput[images/s]"]
    assert (
        float(sim_data["throughput[images/s]"]) / float(est_data["estimated_throughput_fps"]) > 0.9
    )

    # TODO:
    # reduce (individual) FIFO sizes by x % and observe throughput drop or deadlock appear
    # shutil.rmtree(tmp_output_dir)


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.fpgadataflow
# whether we are testing input or output characterization
@pytest.mark.parametrize("direction", ["input", "output"])
@pytest.mark.parametrize(
    "node",
    [
        ("LabelSelect", 10, 1, 1, DataType["UINT8"], "hls"),
        ("LabelSelect", 10, 1, 3, DataType["UINT8"], "hls"),
        ("LabelSelect", 10, 2, 3, DataType["UINT8"], "hls"),
        (
            "MVAU",
            5,
            1,
            8,
            1,
            [1, 1],
            DataType["UINT2"],
            DataType["UINT2"],
            DataType["UINT2"],
            "hls",
        ),
        (
            "MVAU",
            5,
            5,
            8,
            8,
            [1, 1],
            DataType["UINT2"],
            DataType["UINT2"],
            DataType["UINT2"],
            "hls",
        ),
        (
            "MVAU",
            10,
            5,
            20,
            4,
            [1, 1],
            DataType["UINT4"],
            DataType["UINT8"],
            DataType["UINT4"],
            "hls",
        ),
        (
            "MVAU",
            48,
            1,
            4,
            1,
            [1, 1],
            DataType["UINT2"],
            DataType["UINT2"],
            DataType["UINT2"],
            "hls",
        ),
        # generalized DWC-variant required
        # ("StreamingDataWidthConverter",[1,4,1,40],[1,4,1,40],8,2,DataType["BIPOLAR"],"hls"),
        # ("StreamingDataWidthConverter",[1,240],[1,240],12,2,DataType["BIPOLAR"],"hls"),
        # ("StreamingDataWidthConverter",[1,36],[1,36],12,12,DataType["BIPOLAR"],"hls"),
        # ("StreamingDataWidthConverter",[1,4,1,9],[1,4,1,18],3,9,DataType["UINT4"],"hls"),
        # ("StreamingDataWidthConverter",[1,1,1,18],[1,1,1,30],9,3,DataType["BIPOLAR"],"hls"),
        # ("StreamingDataWidthConverter",[1,90],[1,90],3,10,DataType["BIPOLAR"],"hls"),
        # ("StreamingDataWidthConverter",[1,40],[1,30],10,3,DataType["BIPOLAR"],"hls"),
        ("FMPadding", [8, 8], [1, 1, 1, 1], 2, 1, DataType["INT2"], "hls"),
        ("FMPadding", [8, 8], [1, 1, 1, 1], 4, 1, DataType["INT2"], "hls"),
        ("FMPadding", [8, 8], [1, 1, 1, 1], 12, 1, DataType["INT2"], "hls"),
        ("FMPadding", [8, 8], [4, 0, 4, 0], 12, 1, DataType["INT2"], "hls"),
        ("FMPadding", [8, 8], [0, 4, 0, 4], 5, 1, DataType["INT2"], "hls"),
        ("FMPadding", [2, 3], [0, 3, 0, 4], 5, 5, DataType["INT2"], "hls"),
        ("FMPadding", [4, 8], [0, 4, 0, 2], 5, 5, DataType["INT2"], "hls"),
        ("FMPadding", [2, 3], [0, 3, 0, 4], 5, 5, DataType["INT2"], "hls"),
        # idim, pad, num_ch,simd,idt
        (
            "ChannelwiseOp",
            DataType["INT8"],
            DataType["INT4"],
            DataType["INT4"],
            4,
            16,
            "add",
            [1, 4, 4],
            "hls",
        ),
        (
            "ChannelwiseOp",
            DataType["INT8"],
            DataType["INT4"],
            DataType["INT4"],
            2,
            16,
            "add",
            [1],
            "hls",
        ),
        # ,idt, act, pdt, nf, ich, func, vecs, impl_style
        # (Pdb) (ifm_dim,output_size,is1d, NumChannels,PoolDim,ImgDim,PE)
        ("StreamingMaxPool", DataType["INT4"], True, 1, 4, 1, 1, 0, "hls"),
        ("StreamingMaxPool", DataType["BIPOLAR"], False, 1, 10, 1, 1, 1, "hls"),
        # ("StreamingMaxPool", DataType["INT4"], True, 2, 32, 4, 1, 0, "hls"),
        # ("StreamingMaxPool", DataType["BIPOLAR"], False, 2, 28, 64, 1, 0, "hls"),
        # ("StreamingMaxPool", DataType["BIPOLAR"], False, 2, 10, 64, 1, 1, "hls"),
        # ("StreamingMaxPool", DataType["INT4"], True, 4, 10, 3, 3, 1, "hls"),
        # idt, dim_1d, k, ifm_dim, ifm_ch, pe, ceil_mode,impl_style
        (
            "ConvolutionInputGenerator",
            DataType["INT2"],
            [6, 1],
            [12, 1],
            16,
            [1, 1],
            [1, 1],
            2,
            0,
            0,
            1,
            False,
            1,
            "hls",
        ),
        # idt,k, ifm_dim, ifm_ch,stride, dilation,
        # simd, dw, parallel_window, m,  flip,   is1d
        (
            "ConvolutionInputGenerator",
            DataType["INT2"],
            [4, 4],
            [8, 8],
            6,
            [4, 4],
            [1, 1],
            2,
            1,
            0,
            1,
            False,
            0,
            "hls",
        ),
        (
            "ConvolutionInputGenerator",
            DataType["INT2"],
            [6, 6],
            [10, 10],
            8,
            [2, 2],
            [1, 1],
            2,
            1,
            0,
            1,
            False,
            0,
            "hls",
        ),
        (
            "ConvolutionInputGenerator",
            DataType["INT2"],
            [4, 4],
            [10, 10],
            16,
            [2, 2],
            [1, 1],
            2,
            1,
            0,
            1,
            False,
            0,
            "hls",
        ),
        (
            "ConvolutionInputGenerator",
            DataType["INT2"],
            [6, 1],
            [8, 1],
            8,
            [3, 1],
            [1, 1],
            1,
            0,
            0,
            1,
            False,
            1,
            "hls",
        ),
        # ("ConvolutionInputGenerator", DataType["INT2"],
        # [6, 6],[12, 12],8,[4, 4],[1, 1],2,0,0,1,False,0,"hls",),
        # ("ConvolutionInputGenerator",DataType["INT2"],
        # [6, 1],[12, 1],16,[2, 1],[1, 1],2,1,0,1,False,1,"hls",),
        # ("ConvolutionInputGenerator",DataType["INT2"],
        # [6, 1],[12, 1],16,[1, 1],[1, 1],2,1,0,1,False,1,"hls",),
        # idt,k, ifm_dim, ifm_ch,stride, dilation, simd,
        # dw, parallel_window, m,  flip,   is1d
        (
            "VVAU",
            DataType["INT4"],
            DataType["INT4"],
            DataType["INT4"],
            3,
            1,
            10,
            10,
            3,
            3,
            3,
            "internal_embedded",
            0,
            "hls",
        ),
        (
            "ChannelwiseOp",
            DataType["INT8"],
            DataType["INT4"],
            DataType["INT4"],
            1,
            16,
            "add",
            [1, 3, 3],
            "hls",
        ),
        (
            "ConvolutionInputGenerator",
            DataType["INT2"],
            [6, 1],
            [12, 1],
            16,
            [3, 1],
            [1, 1],
            2,
            0,
            0,
            1,
            False,
            1,
            "hls",
            # ("Thresholding", [15, 3], True, True, "hls"),
        ),
    ],
)
def test_fifosizing_analytical_characterization(direction, node):
    test_rtl = True

    test_fpga_part = "xc7z020clg400-1"
    target_clk_ns = 4

    # attempt to cache a pre-existing variant of the model
    # this is to avoid generating RTL multiple times during
    # test debugging
    build_dir = os.environ["FINN_BUILD_DIR"]
    model_cache = None
    for x in os.listdir(build_dir):
        if x.startswith(str(node)):
            model_cache = f"{build_dir}/{x}/model.onnx"

    tmp_output_dir = make_build_dir("build_fifosizing")

    if node[0] == "LabelSelect":
        labels, pe, k, idt, impl_style = node[1:]
        model0 = make_labelselect_modelwrapper(labels, pe, k, idt, impl_style)

    elif node[0] == "Thresholding":
        cfg, narrow, per_tensor, impl_style = node[1:]
        ch = cfg[0]
        pe = cfg[1]
        n_inp_vecs = [1, 1, 1]
        hls_mem_mode = "internal_decoupled"
        act = DataType["INT4"]
        idt = DataType["INT16"]
        odt = act
        n_steps = act.get_num_possible_values() - 1
        # Generate random thresholds and sort in ascending order
        T = generate_random_threshold_values(idt, ch, n_steps, narrow, per_tensor)

        # provide non-decreasing/ascending thresholds
        T = sort_thresholds_increasing(T)

        actval = act.min()
        if narrow:
            actval += 1

        model = make_single_thresholding_modelwrapper(
            impl_style, T, idt, odt, actval, n_inp_vecs, ch
        )
        model = model.transform(SpecializeLayers(test_fpga_part))

        # Make sure that specialize layer did not default to HLS implementation
        assert model.graph.node[0].op_type == "Thresholding_" + str(impl_style)

        node_inst = model.get_nodes_by_op_type(f"Thresholding_{impl_style}")[0]
        op_inst = getCustomOp(node_inst)
        op_inst.set_nodeattr("PE", pe)
        if impl_style == "hls":
            op_inst.set_nodeattr("mem_mode", hls_mem_mode)
        op_inst.set_nodeattr("runtime_writeable_weights", 1)
        model0 = model

    elif node[0] == "MVAU":
        mw, simd, mh, pe, numVectors, wdt, idt, odt, impl_style = node[1:]
        W = gen_finn_dt_tensor(wdt, (mw, mh))
        model0 = make_single_fclayer_modelwrapper(W, pe, simd, wdt, idt, odt, T=None, tdt=None)

        getCustomOp(model0.graph.node[0]).set_nodeattr("numInputVectors", numVectors)
    # model0 = make_labelselect_modelwrapper(labels, pe, k, idt, impl_style)

    elif node[0] == "ChannelwiseOp":
        idt, act, pdt, nf, ich, func, vecs, impl_style = node[1:]
        if nf == -1:
            nf = ich
        odt = act
        pe = ich // nf
        C = gen_finn_dt_tensor(pdt, (ich))

        model0 = make_channelwise_modelwrapper(C, pe, idt, odt, pdt, func, vecs)

    elif node[0] == "FMPadding":
        idim, pad, num_ch, simd, idt, impl_style = node[1:]
        model0 = make_single_fmpadding_modelwrapper(impl_style, idim, pad, num_ch, simd, idt)

    elif node[0] == "StreamingDataWidthConverter":
        in_shape, out_shape, in_width, out_width, dtype, impl_style = node[1:]
        model0 = make_single_dwc_modelwrapper(
            in_shape, out_shape, in_width, out_width, dtype, impl_style
        )
    # model0 = make_labelselect_modelwrapper(labels, pe, k, idt, impl_style)

    elif node[0] == "StreamingMaxPool":
        idt, dim_1d, k, ifm_dim, ifm_ch, pe, ceil_mode, impl_style = node[1:]
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
        ofm_dim_h = compute_pool_output_dim(ifm_dim_h, k_h, stride_h, 0, ceil_mode)
        ofm_dim_w = compute_pool_output_dim(ifm_dim_w, k_w, stride_w, 0, ceil_mode)
        ofm_dim = (ofm_dim_h, ofm_dim_w)
        # if idt == DataType["BIPOLAR"] and dim_1d:
        #    pytest.skip("Skipping binary StreamingMaxPool_1d (not implemented)")
        if (ifm_dim_h % k_h != 0 or ifm_dim_w % k_w != 0) and (not dim_1d):
            pytest.skip("StreamingMaxPool_2d test w/ ImgDim % PoolDim != 0 not implemented")
        if pe > ifm_ch:
            pytest.skip("PE cannot be larger than number of input channels")
        # if pe > 1 and (not dim_1d):
        #     pytest.skip("PE>1 only supported for StreamingMaxPool_1d")

        golden = make_single_maxpoolnhwc_modelwrapper(k, ifm_ch, ifm_dim, ofm_dim, idt, ceil_mode)

        model = golden.transform(InferStreamingMaxPool())
        model = model.transform(InferShapes())

        model0 = model.transform(SpecializeLayers(test_fpga_part))

        # Ensure PE value is set
        streamingmaxpool_node = model0.get_nodes_by_op_type("StreamingMaxPool_hls")[0]
        # assert True == False
        if pe > 1 and (not dim_1d):
            getCustomOp(streamingmaxpool_node).set_nodeattr("PE", 1)
        else:
            getCustomOp(streamingmaxpool_node).set_nodeattr("PE", pe)

    elif node[0] == "ConvolutionInputGenerator":
        (
            idt,
            k,
            ifm_dim,
            ifm_ch,
            stride,
            dilation,
            simd,
            dw,
            parallel_window,
            m,
            flip,
            is1d,
            impl_style,
        ) = node[1:]
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
        if ((stride_h > k_h) or (stride_w > k_w)) and not (
            parallel_window or (k_h == 1 and k_w == 1)
        ):
            pytest.skip("Not all combinations for stride > k edge case supported in default mode")
        if parallel_window and simd != ifm_ch and not (dw or (k_h == 1 and k_w == 1)):
            pytest.skip("Parallel window requires SIMD=C for non-depthwise case")

        ofm_dim_h = compute_conv_output_dim(ifm_dim_h, k_h, stride_h, 0, dilation_h)
        ofm_dim_w = compute_conv_output_dim(ifm_dim_w, k_w, stride_w, 0, dilation_w)
        ofm_dim = [ofm_dim_h, ofm_dim_w]

        model = make_single_im2col_modelwrapper(
            k, ifm_ch, ifm_dim, ofm_dim, stride, dilation, idt, dw
        )
        model = model.transform(to_hw.InferConvInpGen())

        # set impl_style
        inst = getCustomOp(model.get_nodes_by_op_type("ConvolutionInputGenerator")[0])
        inst.set_nodeattr("is1D", is1d)
        inst.set_nodeattr("preferred_impl_style", impl_style)
        model = model.transform(SpecializeLayers(test_fpga_part))
        # set simd
        inst = getCustomOp(model.graph.node[0])
        inst.set_nodeattr("SIMD", simd)
        optype = model.graph.node[0].op_type
        if optype == "ConvolutionInputGenerator_rtl":
            inst.set_nodeattr("parallel_window", parallel_window)
            inst.set_nodeattr("M", m)
        if optype == "ConvolutionInputGenerator_hls":
            if inst.get_nodeattr("is1D"):
                inst.set_nodeattr("parallel_window", parallel_window)
        model0 = model

    elif node[0] == "VVAU":
        (
            idt,
            wdt,
            act,
            pe,
            simd,
            dim_h,
            dim_w,
            k_h,
            k_w,
            channels,
            mem_mode,
            no_act,
            impl_style,
        ) = node[1:]

        if dim_w == 1 and k_w != 1:
            pytest.skip("1D image requires 1D kernel, skipping.")

        if channels % pe != 0:
            pytest.skip("Requirement Channels divisable by PE is violated.")

        if (k_h * k_w) % simd != 0:
            pytest.skip("Requirement kernel (k_h * k_w) divisable by SIMD is violated.")

        # Generate weights in expected shape for ONNX and HLS node
        W = gen_finn_dt_tensor(wdt, (channels, 1, k_h, k_w))  # shape: [channels, 1, k, k]
        # Generate inputs in expected format for ONNX and HLS node
        x = gen_finn_dt_tensor(idt, (1, dim_h, dim_w, k_h * k_w * channels))
        x_vvau = x.reshape(1, dim_h, dim_w, k_h * k_w, channels // pe, pe)
        x_vvau = x_vvau.transpose(0, 1, 2, 4, 3, 5)
        x_vvau = x_vvau.reshape(1, dim_h, dim_w, channels * k_h * k_w)

        if act is None:
            T = None
            tdt = None
            if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
                odt = DataType["UINT32"]
            else:
                odt = DataType["INT32"]
        else:
            odt = act
            (min_v, max_v) = _calculate_dot_prod_range(idt, wdt, k_h * k_w)
            n_steps = act.get_num_possible_values() - 1
            T = np.random.randint(min_v, max_v - 1, (channels, n_steps)).astype(np.float32)
            T = np.sort(T, axis=1)
            if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
                tdt = DataType["UINT32"]
                # bias thresholds to be positive
                T = np.ceil((T + (k_h * k_w)) / 2)
                assert (T >= 0).all()
            else:
                tdt = DataType["INT32"]

        model = _make_single_vvau_modelwrapper(
            W,
            pe,
            simd,
            k_h,
            k_w,
            channels,
            dim_h,
            dim_w,
            wdt,
            idt,
            odt,
            T,
            tdt,
            mem_mode,
            impl_style,
        )
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())

        inst = getCustomOp(model.graph.node[0])
        inst.set_nodeattr("noActivation", no_act)
        if impl_style == "rtl":
            inst.set_nodeattr("resType", "dsp")
        inst.set_nodeattr("preferred_impl_style", impl_style)

        model0 = model.transform(SpecializeLayers(test_fpga_part))
        # test_fpga_part = test_fpga_part

    outputs = [build_cfg.DataflowOutputType.ESTIMATE_REPORTS]
    model1 = copy.deepcopy(model0)

    if model_cache is not None:
        model0 = ModelWrapper(model_cache)

    node_inst0 = getCustomOp(model0.graph.node[0])
    node_inst1 = getCustomOp(model1.graph.node[0])

    cfg = build_cfg.DataflowBuildConfig(
        output_dir=tmp_output_dir,
        synth_clk_period_ns=target_clk_ns,
        generate_outputs=outputs,
        fpga_part=test_fpga_part,
        auto_fifo_strategy="characterize",
        characteristic_function_strategy="analytical",
        auto_fifo_depths=True,
        split_large_fifos=False,
    )

    # analytical
    inst = getCustomOp(model1.graph.node[0])
    inst.set_nodeattr("preferred_impl_style", impl_style)
    model1 = model1.transform(SpecializeLayers(test_fpga_part))
    model1 = model1.transform(GiveUniqueNodeNames())
    model1 = model1.transform(PrepareIP(test_fpga_part, target_clk_ns))
    model1 = step_set_fifo_depths(model1, cfg)

    # rtlsim-based
    if test_rtl:
        cfg.characteristic_function_strategy = "rtlsim"
        if model_cache is None:
            inst = getCustomOp(model0.graph.node[0])
            model0 = model0.transform(SpecializeLayers(test_fpga_part))
            model0 = model0.transform(GiveUniqueNodeNames())
            model0 = model0.transform(PrepareIP(test_fpga_part, target_clk_ns))
            model0 = step_set_fifo_depths(model0, cfg)

            tmp_caching_output_dir = make_build_dir(str(node))
            model0.save(tmp_caching_output_dir + "/model.onnx")

    # grab the last nodes of the model
    if test_rtl:
        for n in model0.graph.node:
            if n.op_type.startswith(node[0]):
                node_inst0 = getCustomOp(n)
                continue

    for n in model1.graph.node:
        if n.op_type.startswith(node[0]):
            node_inst1 = getCustomOp(n)
            continue

    if test_rtl:
        test_relaxation = 5
        if direction == "input":
            assert compare_two_chr_funcs(
                node_inst0.get_nodeattr("io_chrc_in"),
                node_inst1.get_nodeattr("io_chrc_in"),
                test_relaxation,
            )
        elif direction == "output":
            assert compare_two_chr_funcs(
                node_inst0.get_nodeattr("io_chrc_out"),
                node_inst1.get_nodeattr("io_chrc_out"),
                test_relaxation,
            )
