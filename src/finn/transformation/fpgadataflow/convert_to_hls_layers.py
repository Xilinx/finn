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


import numpy as np
import warnings
from onnx import TensorProto, helper

import finn.core.data_layout as DataLayout
from finn.core.datatype import DataType
from finn.custom_op.registry import getCustomOp
from finn.transformation.base import Transformation
from finn.transformation.fpgadataflow.minimize_accumulator_width import (
    MinimizeAccumulatorWidth,
)
from finn.transformation.general import SortGraph
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_shapes import InferShapes
from finn.util.basic import get_by_name
from finn.util.onnx import nchw_to_nhwc


class InferConvInpGen(Transformation):
    """Convert Im2Col layers to ConvolutionInputGenerator layers."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Im2Col":
                i2c_input = n.input[0]
                i2c_output = n.output[0]
                i2c_in_shape = model.get_tensor_shape(i2c_input)
                i2c_out_shape = model.get_tensor_shape(i2c_output)
                dt = model.get_tensor_datatype(i2c_input)
                if not dt.is_integer():
                    warnings.warn(
                        "%s : Input is not int. Can't infer ConvInpGen." % n.name
                    )
                    continue
                i2c_inst = getCustomOp(n)
                stride_h, stride_w = i2c_inst.get_nodeattr("stride")
                k_h, k_w = i2c_inst.get_nodeattr("kernel_size")
                pad_attr = i2c_inst.get_nodeattr("pad_amount")
                pad_h = pad_attr[0] + pad_attr[2]
                pad_w = pad_attr[1] + pad_attr[3]
                dilation_h, dilation_w = i2c_inst.get_nodeattr("dilations")
                # temporary checks until non-square conv support is finalized
                pad_val = i2c_inst.get_nodeattr("pad_value")
                depthwise = i2c_inst.get_nodeattr("depthwise")
                ifm_ch = i2c_in_shape[-1]
                ifm_dim_h = i2c_in_shape[1]
                ifm_dim_w = i2c_in_shape[2]
                ofm_dim_h = i2c_out_shape[1]
                ofm_dim_w = i2c_out_shape[2]

                # default params for ConvolutionInputGenerator
                ConvInpGen_node_idx = node_ind
                ConvInpGen_input = i2c_input
                ConvInpGen_idim_h = ifm_dim_h
                ConvInpGen_idim_w = ifm_dim_w

                if pad_h > 0 or pad_w > 0:
                    # if padding enabled, ensure pad_val supported by DataType
                    # assert dt.allowed(pad_val),"""FMPadding_Batch DataType
                    # must support pad_val"""
                    assert pad_val == 0, (
                        "%s : FMPadding_Batch doesn't currently support pad_val!= 0"
                        % n.name
                    )

                    odim_padding_h = ifm_dim_h + pad_h
                    odim_padding_w = ifm_dim_w + pad_w

                    padding_out = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        (1, odim_padding_h, odim_padding_w, ifm_ch),
                    )
                    graph.value_info.append(padding_out)
                    padding_out = padding_out.name
                    model.set_tensor_datatype(padding_out, dt)

                    ConvInpGen_node_idx += 1
                    ConvInpGen_input = padding_out
                    ConvInpGen_idim_h = odim_padding_h
                    ConvInpGen_idim_w = odim_padding_w

                    padding_node = helper.make_node(
                        "FMPadding_Batch",
                        [i2c_input],
                        [padding_out],
                        domain="finn.custom_op.fpgadataflow",
                        backend="fpgadataflow",
                        ImgDim=[ifm_dim_h, ifm_dim_w],
                        Padding=pad_attr,
                        NumChannels=ifm_ch,
                        inputDataType=dt.name,
                        SIMD=ifm_ch,
                        name="FMPadding_Batch_" + n.name,
                    )
                    graph.node.insert(node_ind, padding_node)

                # Ensure that only supported HLS nodes are inserted
                is_square_image = ConvInpGen_idim_h == ConvInpGen_idim_w
                is_square_kernel = k_h == k_w
                is_kernel_pointwise = k_h == 1 and k_w == 1
                is_equal_stride = stride_h == stride_w
                is_1d_convolution = (k_h == 1 and k_w > 1 and ifm_dim_h == 1) or (
                    k_h > 1 and k_w == 1 and ifm_dim_w == 1
                )

                if (stride_h > 1 or stride_w > 1) and is_kernel_pointwise:
                    assert is_square_image, (
                        "%s : DownSampler currently only supports square input images."
                        % n.name
                    )
                    assert is_equal_stride, (
                        """%s : DownSampler currently only supports equal stride value
                        along different axes."""
                        % n.name
                    )
                    ConvInpGen_idim = ConvInpGen_idim_h
                    stride = stride_h
                    # create DownSampler node
                    ConvInpGen_node = helper.make_node(
                        "DownSampler",
                        [ConvInpGen_input],
                        [i2c_output],
                        domain="finn.custom_op.fpgadataflow",
                        backend="fpgadataflow",
                        ImgDim=ConvInpGen_idim,
                        NumChannels=ifm_ch,
                        SIMD=ifm_ch,
                        Stride=stride,
                        inputDataType=dt.name,
                        name="DownSampler_" + n.name,
                    )
                    graph.node.insert(ConvInpGen_node_idx, ConvInpGen_node)
                else:
                    # create equivalent ConvolutionInputGenerator node
                    if (
                        is_square_image and is_square_kernel
                    ):  # square images and square kernels
                        assert is_equal_stride, (
                            """%s: Non-equal strides along different axes is not supported
                            for (non-)square convolutions"""
                            % n.name
                        )
                        assert dilation_h == 1 and dilation_w == 1, (
                            """%s: Dilation value != 1 is not supported
                            for square convolutions"""
                            % n.name
                        )
                        ConvInpGen_node = helper.make_node(
                            "ConvolutionInputGenerator",
                            [ConvInpGen_input],
                            [i2c_output],
                            domain="finn.custom_op.fpgadataflow",
                            backend="fpgadataflow",
                            ConvKernelDim=[k_h, k_w],
                            IFMChannels=ifm_ch,
                            IFMDim=[ConvInpGen_idim_h, ConvInpGen_idim_w],
                            OFMDim=[ofm_dim_h, ofm_dim_w],
                            SIMD=ifm_ch,
                            Stride=[stride_h, stride_w],
                            Dilation=[dilation_h, dilation_w],
                            inputDataType=dt.name,
                            outputDataType=dt.name,
                            depthwise=depthwise,
                            name="ConvolutionInputGenerator_" + n.name,
                        )
                    else:  # non-square images and/or kernels
                        assert is_1d_convolution, (
                            "%s: ConvolutionInputGenerator1D works only for 1D convs"
                            % n.name
                        )
                        if dilation_h > 1 or dilation_w > 1:
                            assert stride_h == 1 and stride_w == 1, (
                                """%s: Stride value of greater than 1 is not supported for convolutions
                                with dilation value greater than 1"""
                                % n.name
                            )
                        ConvInpGen_node = helper.make_node(
                            "ConvolutionInputGenerator1D",
                            [ConvInpGen_input],
                            [i2c_output],
                            domain="finn.custom_op.fpgadataflow",
                            backend="fpgadataflow",
                            ConvKernelDim=[k_h, k_w],
                            IFMChannels=ifm_ch,
                            IFMDim=[ConvInpGen_idim_h, ConvInpGen_idim_w],
                            OFMDim=[ofm_dim_h, ofm_dim_w],
                            SIMD=ifm_ch,
                            Stride=[stride_h, stride_w],
                            Dilation=[dilation_h, dilation_w],
                            inputDataType=dt.name,
                            outputDataType=dt.name,
                            depthwise=depthwise,
                            name="ConvolutionInputGenerator1D_" + n.name,
                        )
                    graph.node.insert(ConvInpGen_node_idx, ConvInpGen_node)
                # remove old nodes
                graph.node.remove(n)
                graph_modified = True
        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class InferUpsample(Transformation):
    """
    Convert Upsample and Resize nodes to layers to UpsampleNearestNeighbour_Batch nodes.
    """

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Upsample" or n.op_type == "Resize":
                # Extract mode and scales and input shape
                mode = get_by_name(n.attribute, "mode").s.decode("ascii")
                if n.op_type == "Upsample":
                    scales = model.get_initializer(n.input[1])
                else:
                    scales = model.get_initializer(n.input[2])
                in_shape = model.get_tensor_shape(n.input[0])

                dt = model.get_tensor_datatype(n.input[0])
                if not dt.is_integer():
                    warnings.warn(
                        "%s: Input not int. Can't infer UpsampleNearestNeighbour."
                        % n.name
                    )
                    continue

                if model.get_tensor_layout(n.input[0]) != DataLayout.NHWC:
                    warnings.warn(
                        "%s: Input not NHWC. Can't infer UpsampleNearestNeighbour."
                        % n.name
                    )
                    continue

                # Check that the parameters are okay
                assert mode == "nearest", (
                    "%s: Upsampling is only supported for the mode nearest." % n.name
                )
                assert len(in_shape) == 4, "Upsampling is only supported for 4D inputs."
                assert scales.shape == (4,), (
                    "%s: Upsampling is only supported for 4D scales." % n.name
                )
                assert (scales >= 1).all(), (
                    n.name + ": Upsampling is only supported for scales "
                    "which are larger or equal 1 in all dimensions."
                )

                # Assumes nhwc layout for scales and input
                assert scales[1] == scales[2], (
                    "%s: Upsampling is only supported for quadratic scales." % n.name
                )
                assert scales[0] == scales[3] == 1, (
                    n.name + ": Upsampling is only supported for scales with "
                    "the first and last dimensions being 1."
                )
                spatial_scale = scales[1]
                assert spatial_scale == int(spatial_scale), (
                    "%s: Upsampling is only supported for integer scales." % n.name
                )

                assert in_shape[1] == in_shape[2], (
                    "%s: Upsampling is only supported for quadratic input shapes."
                    % n.name
                )

                # Extract information for HLS node
                IFMDim = in_shape[1]
                OFMDim = int(round(in_shape[1] * spatial_scale))
                NumChannels = in_shape[-1]
                numInputVectors = in_shape[0]
                inputDataType = dt.name

                # Insert the HLSCustomOp node
                Upsample_HLS_node = helper.make_node(
                    "UpsampleNearestNeighbour_Batch",
                    [n.input[0]],
                    [n.output[0]],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    OFMDim=OFMDim,
                    IFMDim=IFMDim,
                    NumChannels=NumChannels,
                    inputDataType=inputDataType,
                    numInputVectors=numInputVectors,
                    name="UpsampleNearestNeighbour_Batch_" + n.name,
                )

                # Remove the old node
                graph.node.insert(node_ind, Upsample_HLS_node)
                # remove old nodes
                graph.node.remove(n)
                graph_modified = True
        return (model, graph_modified)


class InferStreamingMaxPool(Transformation):
    """Convert MaxPoolNHWC layers to StreamingMaxPool layers."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "MaxPoolNHWC":
                mp_input = n.input[0]
                mp_output = n.output[0]
                mp_in_shape = model.get_tensor_shape(mp_input)
                # mp_out_shape = model.get_tensor_shape(mp_output)
                dt = model.get_tensor_datatype(mp_input)
                mp_inst = getCustomOp(n)
                k_h, k_w = mp_inst.get_nodeattr("kernel_shape")
                ifm_ch = mp_in_shape[-1]
                ifm_dim_h = mp_in_shape[1]
                ifm_dim_w = mp_in_shape[2]
                if ifm_dim_h % k_h == 0 and ifm_dim_w % k_w == 0:
                    # create equivalent StreamingMaxPool_Batch node
                    new_node = helper.make_node(
                        "StreamingMaxPool_Batch",
                        [mp_input],
                        [mp_output],
                        domain="finn.custom_op.fpgadataflow",
                        backend="fpgadataflow",
                        PoolDim=(k_h, k_w),
                        NumChannels=ifm_ch,
                        ImgDim=(ifm_dim_h, ifm_dim_w),
                        dataType=dt.name,
                        name="StreamingMaxPool_Batch_" + n.name,
                    )
                    graph.node.insert(node_ind, new_node)
                    # remove old nodes
                    graph.node.remove(n)
                    graph_modified = True
        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class InferPool_Batch(Transformation):
    """If kernel_shape > strides, replace Pool layer with  with of Im2col
    + pool(with kernel_shape == strides), plus Transpose layers to keep the original
    data layout."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type in ["MaxPool", "QuantAvgPool2d", "MaxPoolNHWC"]:
                # extract pool parameters

                if n.op_type == "MaxPool":
                    k = get_by_name(n.attribute, "kernel_shape").ints[-1]
                    stride = get_by_name(n.attribute, "strides").ints[-1]
                    # assumed datalayout
                    dlayout = "NCHW"
                elif n.op_type == "QuantAvgPool2d":
                    inst = getCustomOp(n)
                    k = inst.get_nodeattr("kernel")
                    stride = inst.get_nodeattr("stride")
                    dlayout = inst.get_nodeattr("data_layout")
                elif n.op_type == "MaxPoolNHWC":
                    inst = getCustomOp(n)
                    k_shape = inst.get_nodeattr("kernel_shape")
                    strides = inst.get_nodeattr("strides")
                    assert k_shape[0] == k_shape[1]
                    assert strides[0] == strides[1]
                    k = k_shape[0]
                    stride = strides[0]
                    dlayout = "NHWC"
                try:
                    pad = get_by_name(n.attribute, "pads").ints[-1]
                except AttributeError:
                    pad = 0

                node_input = n.input[0]
                node_output = n.output[0]
                idt = model.get_tensor_datatype(node_input)

                if not idt.is_integer():
                    continue

                if k < stride:
                    continue
                elif k == stride:
                    warnings.warn(
                        n.name
                        + """: Inferring Pool_Batch node for k == stride.
                        This case can be optimized.
                        For example, for MaxPool run InferStreamingMaxPool before
                        InferPool_Batch """
                    )

                odt = model.get_tensor_datatype(node_output)

                if dlayout == "NCHW":
                    ifm_ch = model.get_tensor_shape(n.input[0])[1]
                else:
                    ifm_ch = model.get_tensor_shape(n.input[0])[-1]
                ofm_ch = ifm_ch
                ifm_dim = model.get_tensor_shape(n.input[0])[-2]
                ofm_dim = model.get_tensor_shape(n.output[0])[-2]

                # if data layout NCHW, we need transpose nodes surrounding
                # the hls layer
                if dlayout == "NCHW":
                    # create new intermediate values
                    inp_trans_out = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        (1, ifm_dim, ifm_dim, ifm_ch),  # NHWC
                    )
                    graph.value_info.append(inp_trans_out)
                    inp_trans_out = inp_trans_out.name
                    model.set_tensor_datatype(inp_trans_out, idt)

                    pool_output = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        (1, ofm_dim, ofm_dim, ofm_ch),
                    )
                    graph.value_info.append(pool_output)
                    pool_output = pool_output.name
                    # model.set_tensor_datatype(pool_output, odt)

                im2col_out = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    TensorProto.FLOAT,
                    (1, ofm_dim, ofm_dim, ifm_ch * k * k),
                )
                graph.value_info.append(im2col_out)
                im2col_out = im2col_out.name
                model.set_tensor_datatype(im2col_out, idt)

                # create new nodes
                if dlayout == "NCHW":
                    # NCHW -> NHWC
                    inp_trans_node = helper.make_node(
                        "Transpose", [node_input], [inp_trans_out], perm=[0, 2, 3, 1]
                    )
                    im2col_in = inp_trans_out
                else:
                    im2col_in = node_input
                    pool_output = node_output

                accum_bits = 0
                pool_size_param = k
                pad_value = 0
                if n.op_type in ["MaxPool", "MaxPoolNHWC"]:
                    pool_fxn = "MaxPool"
                    odt = idt
                    pad_value = idt.min()
                elif n.op_type == "QuantAvgPool2d":
                    assert odt.is_integer(), """Output data type for QuantAvgPool2d
                    needs to be integer"""
                    assert pad == 0, "Padding is not supported for QuantAvgPool2d"
                    inst = getCustomOp(n)
                    pool_fxn = "QuantAvgPool"
                    pool_size_param = inst.get_shifts()
                    accum_bits = inst.get_accum_size()

                else:
                    raise Exception(
                        "pad_value and pool_fxn not configured for {}".format(n.op_type)
                    )

                # format input tensor
                im2col_node = helper.make_node(
                    "Im2Col",
                    [im2col_in],
                    [im2col_out],
                    domain="finn.custom_op.general",
                    stride=[stride, stride],
                    kernel_size=[k, k],
                    pad_amount=[pad, pad, pad, pad],
                    pad_value=pad_value,
                    depthwise=1,
                    input_shape="(1,{},{},{})".format(ifm_dim, ifm_dim, ifm_ch),
                    name="Im2Col_" + n.name,
                )

                # Warning PE has to be equal to ifm_ch until Im2Col is replaced by
                # ConvolutionInputGenerator with depthwise=1.
                # For other settings the output will be incorrect due to incorrect input
                # data layout
                pool_node = helper.make_node(
                    "Pool_Batch",
                    [im2col_out],
                    [pool_output],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    InputDataType=idt.name,
                    OutputDataType=odt.name,
                    Channels=ifm_ch,
                    PE=ifm_ch,
                    KernelSize=k,
                    Function=pool_fxn,
                    OutImgDim=ofm_dim,
                    AccumBits=accum_bits,
                    Size=pool_size_param,
                    BatchSize=1,
                    name="Pool_Batch_" + n.name,
                )

                if dlayout == "NCHW":
                    # NHWC -> NCHW
                    out_trans_node = helper.make_node(
                        "Transpose", [pool_output], [node_output], perm=[0, 3, 1, 2]
                    )

                # insert nodes where the conv is to preserve topological ordering
                if dlayout == "NCHW":
                    graph.node.insert(node_ind, inp_trans_node)
                    graph.node.insert(node_ind + 1, im2col_node)
                    graph.node.insert(node_ind + 2, pool_node)
                    graph.node.insert(node_ind + 3, out_trans_node)
                else:
                    graph.node.insert(node_ind, im2col_node)
                    graph.node.insert(node_ind + 1, pool_node)
                # remove old node
                graph.node.remove(n)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class InferBinaryStreamingFCLayer(Transformation):
    """Convert XnorPopcountMatMul layers to
    StreamingFCLayer_Batch layers. Any immediately following MultiThreshold
    layers will also be absorbed into the MVTU."""

    def __init__(self, mem_mode="const"):
        super().__init__()
        self.mem_mode = mem_mode

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "XnorPopcountMatMul":
                mm_input = n.input[0]
                mm_weight = n.input[1]
                mm_output = n.output[0]
                mm_in_shape = model.get_tensor_shape(mm_input)
                mm_out_shape = model.get_tensor_shape(mm_output)
                assert model.get_tensor_datatype(mm_input) == DataType["BINARY"], (
                    n.name
                    + """: First
                input for xnorpopcount is not set to FINN DataType BINARY."""
                )
                assert model.get_tensor_datatype(mm_weight) == DataType["BINARY"], (
                    n.name
                    + """: Second
                input (weights) for xnorpopcount is not set to FINN DataType BINARY."""
                )
                idt = DataType["BINARY"]
                wdt = DataType["BINARY"]
                mm_output = n.output[0]
                W = model.get_initializer(mm_weight)
                # extract weight shape, note that ONNX and finn-hlslib
                # make different assumptions about dim order here
                # ONNX assumes W has (in, out) shape
                # finn-hlslib assumes W has (out, in) shape
                mh = int(W.shape[1])
                mw = int(W.shape[0])
                # create node with no parallelization first
                pe = 1
                simd = 1
                wmem = mw * mh // (pe * simd)
                assert mw * mh == wmem * pe * simd, (
                    n.name
                    + """: Requirement (MW * MH) divisiable by
                (WMEM * PE * SIMD) is violated."""
                )
                # see if we have any following thresholds
                consumer = model.find_consumer(mm_output)
                if consumer is not None and consumer.op_type == "MultiThreshold":
                    # TODO ensure integer thresholds?
                    # create MVTU (i.e. including activation)
                    mt_output = consumer.output[0]
                    mt_out_shape = model.get_tensor_shape(mt_output)
                    mt_thres = consumer.input[1]
                    T = model.get_initializer(mt_thres)
                    assert T.shape[0] == 1 or T.shape[0] == mh, (
                        consumer.name
                        + """: First dimension of
                    thresholds neither 1 nor MH."""
                    )
                    odt = model.get_tensor_datatype(mt_output)
                    if odt.bitwidth() == 1:
                        # covers both bipolar and binary
                        actval = 0
                    else:
                        actval = odt.min()
                    model.set_tensor_shape(mm_input, mm_in_shape)
                    model.set_tensor_shape(mt_output, mt_out_shape)
                    # create and insert new StreamingFCLayer node
                    new_node = helper.make_node(
                        "StreamingFCLayer_Batch",
                        [mm_input, mm_weight, mt_thres],
                        [mt_output],
                        domain="finn.custom_op.fpgadataflow",
                        backend="fpgadataflow",
                        MW=mw,
                        MH=mh,
                        SIMD=simd,
                        PE=pe,
                        inputDataType=idt.name,
                        weightDataType=wdt.name,
                        outputDataType=odt.name,
                        ActVal=actval,
                        binaryXnorMode=1,
                        noActivation=0,
                        numInputVectors=list(mm_in_shape[:-1]),
                        mem_mode=self.mem_mode,
                        name=n.name,
                    )
                    graph.node.insert(node_ind, new_node)
                    # remove old nodes
                    graph.node.remove(n)
                    graph.node.remove(consumer)
                    graph_modified = True
                else:
                    # no activation, matmul only
                    odt = model.get_tensor_datatype(mm_output)
                    model.set_tensor_shape(mm_input, mm_in_shape)
                    model.set_tensor_shape(mm_output, mm_out_shape)
                    # create and insert new StreamingFCLayer node
                    new_node = helper.make_node(
                        "StreamingFCLayer_Batch",
                        [mm_input, mm_weight],
                        [mm_output],
                        domain="finn.custom_op.fpgadataflow",
                        backend="fpgadataflow",
                        MW=mw,
                        MH=mh,
                        SIMD=simd,
                        PE=pe,
                        inputDataType=idt.name,
                        weightDataType=wdt.name,
                        outputDataType=odt.name,
                        ActVal=0,
                        binaryXnorMode=1,
                        noActivation=1,
                        numInputVectors=list(mm_in_shape[:-1]),
                        mem_mode=self.mem_mode,
                        name=n.name,
                    )
                    graph.node.insert(node_ind, new_node)
                    # remove old node
                    graph.node.remove(n)
                    graph_modified = True
        if graph_modified:
            model = model.transform(MinimizeAccumulatorWidth())
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class InferQuantizedStreamingFCLayer(Transformation):
    """Convert MatMul layers with quantized inputs and weights to
    StreamingFCLayer_Batch layers. Any immediately following MultiThreshold
    layers will also be absorbed into the MVTU."""

    def __init__(self, mem_mode="const"):
        super().__init__()
        self.mem_mode = mem_mode

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "MatMul" and model.get_tensor_sparsity(n.input[1]) is None:
                mm_input = n.input[0]
                mm_weight = n.input[1]
                mm_output = n.output[0]
                mm_in_shape = model.get_tensor_shape(mm_input)
                mm_out_shape = model.get_tensor_shape(mm_output)
                idt = model.get_tensor_datatype(mm_input)
                wdt = model.get_tensor_datatype(mm_weight)
                if idt.is_integer() and wdt.is_integer():
                    mm_output = n.output[0]
                    W = model.get_initializer(mm_weight)
                    # extract weight shape, note that ONNX and finn-hlslib
                    # make different assumptions about dim order here
                    # ONNX assumes W has (in, out) shape
                    # finn-hlslib assumes W has (out, in) shape
                    mh = int(W.shape[1])
                    mw = int(W.shape[0])
                    # create node with no parallelization first
                    pe = 1
                    simd = 1
                    wmem = mw * mh // (pe * simd)
                    assert mw * mh == wmem * pe * simd, (
                        n.name
                        + """: Requirement (MW * MH) divisible by
                    (WMEM * PE * SIMD) is violated."""
                    )
                    # see if we have any following thresholds
                    consumer = model.find_consumer(mm_output)
                    if consumer is not None and consumer.op_type == "MultiThreshold":
                        # TODO ensure integer thresholds?
                        # create MVTU (i.e. including activation)
                        mt_output = consumer.output[0]
                        mt_out_shape = model.get_tensor_shape(mt_output)
                        mt_thres = consumer.input[1]
                        T = model.get_initializer(mt_thres)
                        assert T.shape[0] == 1 or T.shape[0] == mh, (
                            consumer.name
                            + """: First dimension of
                        thresholds neither 1 nor MH."""
                        )
                        odt = model.get_tensor_datatype(mt_output)
                        scale = getCustomOp(consumer).get_nodeattr("out_scale")
                        actval = getCustomOp(consumer).get_nodeattr("out_bias")
                        assert int(actval) == actval, (
                            consumer.name
                            + ": out_bias must be integer for HLS conversion."
                        )
                        actval = int(actval)
                        odt_is_bipolar = odt == DataType["BIPOLAR"]
                        bipolar_ok = (
                            odt_is_bipolar and (scale == 2.0) and (actval == -1)
                        )
                        assert scale == 1.0 or bipolar_ok, (
                            consumer.name
                            + ": out_scale=1 or bipolar output needed for conversion."
                        )
                        assert (not odt.signed()) or (actval < 0), (
                            consumer.name + ": Signed output requres actval < 0"
                        )
                        model.set_tensor_shape(mm_input, mm_in_shape)
                        model.set_tensor_shape(mt_output, mt_out_shape)
                        if bipolar_ok:
                            # remove bias for bipolar, since
                            # binary->bipolar is achieved by reinterpretation
                            actval = 0
                        # create and insert new StreamingFCLayer node
                        new_node = helper.make_node(
                            "StreamingFCLayer_Batch",
                            [mm_input, mm_weight, mt_thres],
                            [mt_output],
                            domain="finn.custom_op.fpgadataflow",
                            backend="fpgadataflow",
                            MW=mw,
                            MH=mh,
                            SIMD=simd,
                            PE=pe,
                            inputDataType=idt.name,
                            weightDataType=wdt.name,
                            outputDataType=odt.name,
                            ActVal=actval,
                            binaryXnorMode=0,
                            noActivation=0,
                            numInputVectors=list(mm_in_shape[:-1]),
                            mem_mode=self.mem_mode,
                            name="StreamingFCLayer_Batch_" + n.name,
                        )
                        graph.node.insert(node_ind, new_node)
                        # remove old nodes
                        graph.node.remove(n)
                        graph.node.remove(consumer)
                        graph_modified = True
                    else:
                        # no activation, matmul only
                        odt = model.get_tensor_datatype(mm_output)
                        model.set_tensor_shape(mm_input, mm_in_shape)
                        model.set_tensor_shape(mm_output, mm_out_shape)
                        # create and insert new StreamingFCLayer node
                        new_node = helper.make_node(
                            "StreamingFCLayer_Batch",
                            [mm_input, mm_weight],
                            [mm_output],
                            domain="finn.custom_op.fpgadataflow",
                            backend="fpgadataflow",
                            MW=mw,
                            MH=mh,
                            SIMD=simd,
                            PE=pe,
                            inputDataType=idt.name,
                            weightDataType=wdt.name,
                            outputDataType=odt.name,
                            ActVal=0,
                            binaryXnorMode=0,
                            noActivation=1,
                            numInputVectors=list(mm_in_shape[:-1]),
                            mem_mode=self.mem_mode,
                            name="StreamingFCLayer_Batch_" + n.name,
                        )
                        graph.node.insert(node_ind, new_node)
                        # remove old node
                        graph.node.remove(n)
                        graph_modified = True
        if graph_modified:
            model = model.transform(MinimizeAccumulatorWidth())
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class InferVVAU(Transformation):
    """Convert MatMul layers with quantized inputs and weights to
    Vector_Vector_Activate_Batch layers, if the sparsity annotation
    of the weight matrix indicates that the MatMul layer belongs to
    a depthwise convolution. Any immediately following MultiThreshold
    layers will also be absorbed into the VVAU."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if (
                n.op_type == "MatMul"
                and model.get_tensor_sparsity(n.input[1]) is not None
            ):
                sparsity = model.get_tensor_sparsity(n.input[1])
                try:
                    k_h, k_w = sparsity["dw"]["kernel_shape"]
                except KeyError:
                    raise Exception(
                        n.name
                        + """: sparsity annotation doesn't indicate that MatMul
                        belongs to a depthwise convolution."""
                    )

                mm_input = n.input[0]
                mm_weight = n.input[1]
                mm_output = n.output[0]
                mm_in_shape = model.get_tensor_shape(mm_input)
                mm_out_shape = model.get_tensor_shape(mm_output)
                idt = model.get_tensor_datatype(mm_input)
                wdt = model.get_tensor_datatype(mm_weight)
                if idt.is_integer() and wdt.is_integer():
                    mm_output = n.output[0]
                    W = model.get_initializer(mm_weight)
                    # infer dense weight tensor from sparse weight matrix
                    # kernel size (k_h, k_w) which was extracted above and the value of
                    # the channels is used.
                    # the weight matrix has a shape of (k_h * k_w * Channels, Channels)
                    # we need to reverse the creation of the sparse weight matrix
                    # to achieve a weight tensor of shape (Channels, 1, k_h, k_w)
                    channels = int(W.shape[1])
                    # transpose to achieve a shape of (k_h * k_w * Channels, Channels)
                    W = W.T
                    # reshape to (Channels, k_h, k_w, Channels) to transpose afterwards
                    # to (Channels, Channels, k_h, k_w)
                    W = W.reshape(channels, k_h, k_w, channels)
                    W = W.transpose(0, 3, 1, 2)
                    # now we can extract the values using a for loop over the channels
                    # and fill a zero numpy array in the correct shape
                    w_tensor = np.zeros((channels, 1, k_h, k_w))
                    for ch in range(channels):
                        w_tensor[ch][0] = W[ch][ch]
                    model.set_initializer(mm_weight, w_tensor)
                    model.set_tensor_shape(mm_weight, (channels, 1, k_h, k_w))
                    # create node with pe=channels as default
                    pe = channels
                    # see if we have any following thresholds
                    consumer = model.find_consumer(mm_output)
                    if consumer is not None and consumer.op_type == "MultiThreshold":
                        # create VVAU (i.e. including activation)
                        mt_output = consumer.output[0]
                        mt_out_shape = model.get_tensor_shape(mt_output)
                        mt_thres = consumer.input[1]
                        T = model.get_initializer(mt_thres)
                        assert T.shape[0] == 1 or T.shape[0] == channels, (
                            consumer.name
                            + """: First dimension of
                        thresholds neither 1 nor Channels."""
                        )
                        odt = model.get_tensor_datatype(mt_output)
                        scale = getCustomOp(consumer).get_nodeattr("out_scale")
                        assert scale == 1.0, (
                            consumer.name
                            + ": out_scale must be equal to 1.0 for HLS conversion."
                        )
                        actval = getCustomOp(consumer).get_nodeattr("out_bias")
                        assert int(actval) == actval, (
                            consumer.name
                            + ": out_bias must be integer for HLS conversion."
                        )
                        actval = int(actval)
                        assert (not odt.signed()) or (actval < 0), (
                            consumer.name + ": Signed output requres actval < 0"
                        )
                        model.set_tensor_shape(mm_input, mm_in_shape)
                        model.set_tensor_shape(mt_output, mt_out_shape)
                        # create and insert new Vector_Vector_Activate_Batch node
                        new_node = helper.make_node(
                            "Vector_Vector_Activate_Batch",
                            [mm_input, mm_weight, mt_thres],
                            [mt_output],
                            domain="finn.custom_op.fpgadataflow",
                            backend="fpgadataflow",
                            resType="lut",
                            PE=pe,
                            Dim=[mm_in_shape[1], mm_in_shape[2]],
                            Channels=channels,
                            Kernel=[k_h, k_w],
                            inputDataType=idt.name,
                            weightDataType=wdt.name,
                            outputDataType=odt.name,
                            ActVal=actval,
                            noActivation=0,
                            name="Vector_Vector_Activate_Batch_" + n.name,
                        )
                        graph.node.insert(node_ind, new_node)
                        # remove old nodes
                        graph.node.remove(n)
                        graph.node.remove(consumer)
                        graph_modified = True
                    else:
                        # no activation, matmul only
                        odt = model.get_tensor_datatype(mm_output)
                        model.set_tensor_shape(mm_input, mm_in_shape)
                        model.set_tensor_shape(mm_output, mm_out_shape)
                        # create and insert new VVAU node
                        new_node = helper.make_node(
                            "Vector_Vector_Activate_Batch",
                            [mm_input, mm_weight],
                            [mm_output],
                            domain="finn.custom_op.fpgadataflow",
                            backend="fpgadataflow",
                            resType="lut",
                            PE=pe,
                            Dim=[mm_in_shape[1], mm_in_shape[2]],
                            Channels=channels,
                            Kernel=[k_h, k_w],
                            inputDataType=idt.name,
                            weightDataType=wdt.name,
                            outputDataType=odt.name,
                            ActVal=0,
                            noActivation=1,
                            name="Vector_Vector_Activate_Batch_" + n.name,
                        )
                        graph.node.insert(node_ind, new_node)
                        # remove old node
                        graph.node.remove(n)
                        graph_modified = True
        if graph_modified:
            model = model.transform(MinimizeAccumulatorWidth())
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class InferThresholdingLayer(Transformation):
    """Convert any MultiThreshold into a standalone thresholding HLS layer."""

    def __init__(self, mem_mode="const"):
        super().__init__()
        self.mem_mode = mem_mode

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            if node.op_type == "MultiThreshold":
                thl_input = node.input[0]
                thl_threshold = node.input[1]
                thl_output = node.output[0]
                thl_in_shape = model.get_tensor_shape(thl_input)
                thl_thres_shape = model.get_tensor_shape(thl_threshold)
                idt = model.get_tensor_datatype(thl_input)

                # skip conversion for layers with float input
                if not idt.is_integer():
                    continue

                # check layout of inputs/outputs, and convert if needed
                # check layout and convert if necessary
                thl_in_layout = model.get_tensor_layout(thl_input)
                if thl_in_layout == DataLayout.NCHW:
                    thl_input = nchw_to_nhwc(thl_input, model, node_ind)
                    node_ind += 1
                    thl_in_shape = model.get_tensor_shape(thl_input)

                # keep track of where we need to insert the HLS Op
                # it has to be ahead of the output transform
                insert_point = node_ind
                thl_output_layout = model.get_tensor_layout(thl_output)
                if thl_output_layout == DataLayout.NCHW:
                    thl_output = nchw_to_nhwc(thl_output, model, node_ind, reverse=True)
                    node_ind += 1

                # now safe to assume number of channels is in last dimension
                ifc = int(thl_in_shape[-1])
                # create node with no parallelization first
                pe = 1

                odt = model.get_tensor_datatype(thl_output)
                scale = getCustomOp(node).get_nodeattr("out_scale")
                assert scale == 1.0, (
                    node.name
                    + ": MultiThreshold out_scale must be 1 for HLS conversion."
                )
                actval = getCustomOp(node).get_nodeattr("out_bias")
                assert int(actval) == actval, (
                    node.name
                    + ": MultiThreshold out_bias must be integer for HLS conversion."
                )
                actval = int(actval)
                assert (not odt.signed()) or (actval < 0), (
                    node.name + ": Signed output requres actval < 0"
                )
                # create and insert new Thresholding_Batch node
                new_node = helper.make_node(
                    "Thresholding_Batch",
                    [thl_input, thl_threshold],
                    [thl_output],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    NumChannels=ifc,
                    PE=pe,
                    numSteps=thl_thres_shape[1],
                    inputDataType=idt.name,
                    weightDataType=idt.name,  # will be set by MinimizeAccumulatorWidth
                    outputDataType=odt.name,
                    numInputVectors=list(thl_in_shape[:-1]),
                    ActVal=actval,
                    mem_mode=self.mem_mode,
                    name="Thresholding_Batch_" + node.name,
                )
                graph.node.insert(insert_point, new_node)
                # remove old node
                graph.node.remove(node)
                graph_modified = True

        if graph_modified:
            model = model.transform(MinimizeAccumulatorWidth())
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class InferAddStreamsLayer(Transformation):
    """Convert any Add into a AddStreams HLS layer."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            if node.op_type == "Add":
                in0 = node.input[0]
                in1 = node.input[1]
                result = node.output[0]
                in0_shape = model.get_tensor_shape(in0)
                in1_shape = model.get_tensor_shape(in1)

                # skip if different shapes on inputs
                if in0_shape != in1_shape:
                    continue

                idt0 = model.get_tensor_datatype(in0)
                idt1 = model.get_tensor_datatype(in1)

                # skip if different data types on inputs
                if idt0 != idt1:
                    continue

                idt = idt0

                # skip conversion for layers with float input
                if not idt.is_integer():
                    continue

                # check layout and convert if necessary
                in0_layout = model.get_tensor_layout(in0)
                in1_layout = model.get_tensor_layout(in1)
                result_layout = model.get_tensor_layout(result)

                if in0_layout == DataLayout.NCHW:
                    in0 = nchw_to_nhwc(in0, model, node_ind)
                    node_ind += 1
                    in0_shape = model.get_tensor_shape(in0)

                if in1_layout == DataLayout.NCHW:
                    in1 = nchw_to_nhwc(in1, model, node_ind)
                    node_ind += 1
                    in1_shape = model.get_tensor_shape(in1)

                # keep track of where we need to insert the HLS Op
                # it has to be ahead of the output transform
                insert_point = node_ind

                if result_layout == DataLayout.NCHW:
                    result = nchw_to_nhwc(result, model, node_ind, reverse=True)
                    node_ind += 1

                # now safe to assume num_channels is size of last dimension
                num_channels = int(in0_shape[-1])
                # create node with no parallelization first
                pe = 1

                # create and insert new StreamingFCLayer node
                new_node = helper.make_node(
                    "AddStreams_Batch",
                    [in0, in1],
                    [result],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    NumChannels=num_channels,
                    PE=pe,
                    inputDataType=idt.name,
                    numInputVectors=in0_shape[:-1],
                    name="AddStreams_Batch_" + node.name,
                )
                graph.node.insert(insert_point, new_node)
                # remove old node
                graph.node.remove(node)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class InferDuplicateStreamsLayer(Transformation):
    """Insert a DuplicateStreams HLS layer for any tensor with fanout == 2"""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            successors = model.find_consumers(node.output[0])
            if successors is not None and len(successors) == 2:
                output_tensor = node.output[0]

                dt = model.get_tensor_datatype(output_tensor)

                # skip conversion for layers with float input
                if not dt.is_integer():
                    continue

                # create clone tensors
                out_shape = model.get_tensor_shape(output_tensor)
                out_tensor_clones = []
                for i in range(2):
                    clone = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(), TensorProto.FLOAT, out_shape
                    )
                    model.graph.value_info.append(clone)
                    out_tensor_clones += [clone.name]

                num_ch = int(out_shape[-1])
                vecs = out_shape[:-1]

                # create node with no parallelization first
                pe = 1

                dup_node = helper.make_node(
                    "DuplicateStreams_Batch",
                    [output_tensor],
                    out_tensor_clones,
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    NumChannels=num_ch,
                    PE=pe,
                    inputDataType=dt.name,
                    numInputVectors=vecs,
                    name="DuplicateStreams_Batch_" + node.name,
                )

                graph.node.insert(node_ind, dup_node)

                # connect successors to out tensor clone
                clone_idx = 0
                for successor in successors:
                    for i, succ_input in enumerate(successor.input):
                        if succ_input == output_tensor:
                            successor.input[i] = out_tensor_clones[clone_idx]
                            clone_idx += 1
                            # if one node has multiple connections to the same output
                            # find_direct_successors will return one node per input
                            # so break the inner loop will result in correct behaviour
                            break

                graph_modified = True

        if graph_modified:
            model = model.transform(SortGraph())
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class InferChannelwiseLinearLayer(Transformation):
    """Convert any channel-wise Add/Mul into a HLS layer."""

    def get_smallest_possible(self, vals):
        """Returns smallest (fewest bits) possible DataType that can represent
        value. Prefers unsigned integers where possible."""
        vals = np.array(vals)
        for v in vals:
            assert int(v) == v, "Error float value"

        for k in DataType.get_accumulator_dt_cands():
            dt = DataType[k]

            if dt in [DataType["BIPOLAR"], DataType["TERNARY"], DataType["FLOAT32"]]:
                # not currently supported
                continue

            if (dt.min() <= vals).all() and (vals <= dt.max()).all():
                return dt

        warnings.warn(
            """InferChannelwiseLinearLayer: Output values may not be
        representable with supported data types.
        Setting maximum width data type available.
        This will lead to errors if there are no constrains on the input
        """
        )

        if (0 <= vals).all():
            return DataType["UINT64"]
        else:
            return DataType["INT64"]

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            if node.op_type == "Add" or node.op_type == "Mul":
                # assuming input[0] is dynamic
                ll_input = node.input[0]
                ll_output = node.output[0]
                ll_in_shape = model.get_tensor_shape(ll_input)

                # check if input 1 has an initializer
                ll_const = node.input[1]
                if ll_const is not None:
                    ll_cinit = model.get_initializer(ll_const)
                    if ll_cinit is None:
                        # input 1 is also dynamic
                        continue
                else:
                    continue

                # get number of channels and channel index from input
                ll_in_layout = model.get_tensor_layout(ll_input)
                if ll_in_layout == DataLayout.NHWC or ll_in_layout == DataLayout.NC:
                    ch_index = -1
                    ch = ll_in_shape[-1]
                elif ll_in_layout == DataLayout.NCHW:
                    ch_index = 1
                    ch = ll_in_shape[1]
                else:
                    continue

                # check if the shape of initializer is compatible
                ll_cinit_shape = list(ll_cinit.shape)
                if np.prod(ll_cinit_shape) == 1:
                    warnings.warn(
                        "Broadcasting " + str(node.op_type) + "(" + node.name + ")"
                    )
                    ll_cinit = np.full((ch), ll_cinit.flatten()[0])
                elif np.prod(ll_cinit_shape) != ch or ll_cinit_shape[ch_index] != ch:
                    # parameter shape not compatible with Channelwise_batch
                    continue

                # check initializer contains integers as floats
                if not (ll_cinit.astype(np.int32) == ll_cinit).all():
                    continue
                # all initializer conditions are met

                # check inputs
                idt = model.get_tensor_datatype(ll_input)
                if not idt.is_integer():
                    # skip conversion for layers with float input
                    continue

                # check layout of inputs/outputs, and convert if needed
                # check layout and convert if necessary
                if ll_in_layout == DataLayout.NCHW:
                    ll_input = nchw_to_nhwc(ll_input, model, node_ind)
                    node_ind += 1
                    ll_in_shape = model.get_tensor_shape(ll_input)

                # keep track of where we need to insert the HLS Op
                # it has to be ahead of the output transform
                insert_point = node_ind
                ll_output_layout = model.get_tensor_layout(ll_output)
                if ll_output_layout == DataLayout.NCHW:
                    ll_output = nchw_to_nhwc(ll_output, model, node_ind, reverse=True)
                    node_ind += 1

                # get parameter data type
                param_min = min(ll_cinit.flatten())
                param_max = max(ll_cinit.flatten())
                pdt = self.get_smallest_possible([param_min, param_max])

                # set function and determine output data type
                if node.op_type == "Add":
                    func = "add"
                    out_min = idt.min() + param_min
                    out_max = idt.max() + param_max
                    odt = self.get_smallest_possible([out_min, out_max])
                elif node.op_type == "Mul":
                    func = "mul"
                    possible_limits = []
                    possible_limits += [idt.min() * param_min]
                    possible_limits += [idt.min() * param_max]
                    possible_limits += [idt.max() * param_min]
                    possible_limits += [idt.max() * param_max]
                    odt = self.get_smallest_possible(possible_limits)

                model.set_initializer(ll_const, ll_cinit.reshape(ch))
                model.set_tensor_datatype(ll_output, odt)

                # create node with no parallelization first
                pe = 1
                assert ch % pe == 0, "Requirement IFC divisable by PE is violated."
                # create and insert node
                new_node = helper.make_node(
                    "ChannelwiseOp_Batch",
                    [ll_input, ll_const],
                    [ll_output],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    Func=func,
                    NumChannels=ch,
                    PE=pe,
                    inputDataType=idt.name,
                    paramDataType=pdt.name,
                    outputDataType=odt.name,
                    numInputVectors=list(ll_in_shape[:-1]),
                    name="ChannelwiseOp_Batch_" + node.name,
                )
                graph.node.insert(insert_point, new_node)
                # remove old node
                graph.node.remove(node)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class InferLabelSelectLayer(Transformation):
    """Convert any TopK into a LabelSelect HLS layer."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            if node.op_type == "TopK":
                fc_input = node.input[0]
                k_input = node.input[1]
                val_output = node.output[0]
                idx_output = node.output[1]
                fc_in_shape = model.get_tensor_shape(fc_input)

                idt = model.get_tensor_datatype(fc_input)

                # skip conversion for layers with float input
                if not idt.is_integer():
                    continue

                # skip conversion for if value output is connected (not supported)
                if model.find_consumer(val_output) is not None:
                    continue

                num_labels = int(fc_in_shape[-1])
                num_inp_vecs = list(fc_in_shape[:-1])
                # create node with no parallelization first
                pe = 1

                k = model.get_initializer(k_input)[0]

                # create and insert new StreamingFCLayer node
                new_node = helper.make_node(
                    "LabelSelect_Batch",
                    [fc_input],
                    [idx_output],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    Labels=num_labels,
                    PE=pe,
                    K=k,
                    inputDataType=idt.name,
                    numInputVectors=num_inp_vecs,
                    name="LabelSelect_Batch_" + node.name,
                )
                graph.node.insert(node_ind, new_node)
                # remove old node
                graph.node.remove(node)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class InferGlobalAccPoolLayer(Transformation):
    """Convert any GlobalAveragePool into a GlobalAccPool HLS layer and a scalar Mul."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            if node.op_type == "GlobalAveragePool":
                in0 = node.input[0]
                result = node.output[0]
                in0_shape = model.get_tensor_shape(in0)

                idt = model.get_tensor_datatype(in0)

                # skip conversion for layers with float input
                if not idt.is_integer():
                    continue

                # check layout and convert if necessary
                in0_layout = model.get_tensor_layout(in0)
                result_layout = model.get_tensor_layout(result)

                if in0_layout == DataLayout.NCHW:
                    in0 = nchw_to_nhwc(in0, model, node_ind)
                    node_ind += 1
                    in0_shape = model.get_tensor_shape(in0)

                # keep track of where we need to insert the HLS Op
                # it has to be ahead of the output transform
                insert_point = node_ind

                if result_layout == DataLayout.NCHW:
                    result = nchw_to_nhwc(result, model, node_ind, reverse=True)
                    node_ind += 1

                num_ch = int(in0_shape[-1])
                vecs = in0_shape[:-1]
                # create node with no parallelization first
                pe = 1

                # create an additional tensor of the same shape and layout as result
                out_shape = model.get_tensor_shape(result)
                pool_out = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(), TensorProto.FLOAT, out_shape
                )
                model.graph.value_info.append(pool_out)
                pool_out = pool_out.name
                model.set_tensor_layout(pool_out, model.get_tensor_layout(result))

                new_pool = helper.make_node(
                    "GlobalAccPool_Batch",
                    [in0],
                    [pool_out],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    NumChannels=num_ch,
                    PE=pe,
                    inputDataType=idt.name,
                    numInputVectors=vecs,
                    name="GlobalAccPool_Batch_" + node.name,
                )

                mul_value = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(), TensorProto.FLOAT, [1]
                )
                model.graph.value_info.append(mul_value)
                model.set_initializer(mul_value.name, np.array(1 / (vecs[1] * vecs[2])))
                new_mul = helper.make_node(
                    "Mul",
                    [pool_out, mul_value.name],
                    [result],
                )
                graph.node.insert(insert_point, new_pool)
                graph.node.insert(insert_point + 1, new_mul)
                node_ind += 1
                # remove old node
                graph.node.remove(node)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class InferLookupLayer(Transformation):
    """Convert Gather nodes with constant op0 into Lookup HLS layers."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            if node.op_type == "Gather":
                emb_name = node.input[0]
                embs = model.get_initializer(emb_name)
                axis = get_by_name(node.attribute, "axis")
                # skip conversion if input0 is not constant
                if embs is None:
                    continue
                # skip conversion if axis != 0
                if axis is not None and axis.i != 0:
                    continue
                ind_name = node.input[1]
                ind_dtype = model.get_tensor_datatype(ind_name)
                emb_dtype = model.get_tensor_datatype(emb_name)
                # skip conversion if inputs are not unsigned integers
                if (not ind_dtype.is_integer()) or ind_dtype.signed():
                    continue
                num_embs, emb_dim = embs.shape
                out_name = node.output[0]
                ishape = model.get_tensor_shape(node.input[1])
                # create and insert new Lookup node
                new_node = helper.make_node(
                    "Lookup",
                    [ind_name, emb_name],
                    [out_name],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    name="Lookup_" + node.name,
                    NumEmbeddings=num_embs,
                    EmbeddingDim=emb_dim,
                    EmbeddingType=emb_dtype.name,
                    InputType=ind_dtype.name,
                    InputShape=list(ishape),
                )
                graph.node.insert(node_ind, new_node)
                # remove old node
                graph.node.remove(node)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)
