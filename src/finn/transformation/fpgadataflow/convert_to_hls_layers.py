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

from onnx import helper, TensorProto
import numpy as np

from finn.core.datatype import DataType
from finn.transformation import Transformation
from finn.custom_op.registry import getCustomOp
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.infer_datatypes import InferDataTypes
import finn.core.data_layout as DataLayout
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
                i2c_inst = getCustomOp(n)
                stride = i2c_inst.get_nodeattr("stride")
                k = i2c_inst.get_nodeattr("kernel_size")
                pad = i2c_inst.get_nodeattr("pad_amount")
                pad_val = i2c_inst.get_nodeattr("pad_value")
                ifm_ch = i2c_in_shape[-1]
                ifm_dim = i2c_in_shape[1]
                ofm_dim = i2c_out_shape[1]

                # default params for ConvolutionInputGenerator
                ConvInpGen_node_idx = node_ind
                ConvInpGen_input = i2c_input
                ConvInpGen_idim = ifm_dim

                if pad > 0:
                    # if padding enabled, ensure pad_val supported by DataType
                    assert dt.allowed(pad_val), "Im2Col DataType must support pad_val"

                    odim_padding = ifm_dim + 2 * pad

                    padding_out = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        (1, odim_padding, odim_padding, ifm_ch),
                    )
                    graph.value_info.append(padding_out)
                    padding_out = padding_out.name
                    model.set_tensor_datatype(padding_out, dt)

                    ConvInpGen_node_idx += 1
                    ConvInpGen_input = padding_out
                    ConvInpGen_idim = odim_padding

                    padding_node = helper.make_node(
                        "FMPadding_Batch",
                        [i2c_input],
                        [padding_out],
                        domain="finn",
                        backend="fpgadataflow",
                        ImgDim=ifm_dim,
                        Padding=2 * pad,
                        NumChannels=ifm_ch,
                        inputDataType=dt.name,
                    )
                    graph.node.insert(node_ind, padding_node)

                # create equivalent ConvolutionInputGenerator node
                ConvInpGen_node = helper.make_node(
                    "ConvolutionInputGenerator",
                    [ConvInpGen_input],
                    [i2c_output],
                    domain="finn",
                    backend="fpgadataflow",
                    ConvKernelDim=k,
                    IFMChannels=ifm_ch,
                    IFMDim=ConvInpGen_idim,
                    OFMDim=ofm_dim,
                    SIMD=ifm_ch,
                    Stride=stride,
                    inputDataType=dt.name,
                    outputDataType=dt.name,
                )
                graph.node.insert(ConvInpGen_node_idx, ConvInpGen_node)
                # remove old nodes
                graph.node.remove(n)
                graph_modified = True
        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
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
                # stride = mp_inst.get_nodeattr("strides")[0]
                k = mp_inst.get_nodeattr("kernel_shape")[0]
                # pad = mp_inst.get_nodeattr("pads")[0]
                ifm_ch = mp_in_shape[-1]
                ifm_dim = mp_in_shape[1]
                # ofm_dim = mp_out_shape[1]
                if ifm_dim % k == 0:
                    # create equivalent StreamingMaxPool_Batch node
                    # TODO support non-k strides
                    new_node = helper.make_node(
                        "StreamingMaxPool_Batch",
                        [mp_input],
                        [mp_output],
                        domain="finn",
                        backend="fpgadataflow",
                        PoolDim=k,
                        NumChannels=ifm_ch,
                        ImgDim=ifm_dim,
                        dataType=dt.name,
                    )
                    graph.node.insert(node_ind, new_node)
                    # remove old nodes
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
                assert (
                    model.get_tensor_datatype(mm_input) == DataType.BINARY
                ), """First
                input for xnorpopcount is not set to FINN DataType BINARY."""
                assert (
                    model.get_tensor_datatype(mm_weight) == DataType.BINARY
                ), """Second
                input (weights) for xnorpopcount is not set to FINN DataType BINARY."""
                idt = DataType.BINARY
                wdt = DataType.BINARY
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
                assert mh % pe == 0, "Requirement MH divisable by PE is violated."
                assert mw % simd == 0, "Requirement MW divisable by SIMD is violated."
                wmem = mw * mh // (pe * simd)
                assert (
                    mw * mh == wmem * pe * simd
                ), """Requirement (MW * MH) divisiable by
                (WMEM * PE * SIMD) is violated."""
                # see if we have any following thresholds
                consumer = model.find_consumer(mm_output)
                if consumer is not None and consumer.op_type == "MultiThreshold":
                    # TODO ensure integer thresholds?
                    # create MVTU (i.e. including activation)
                    mt_output = consumer.output[0]
                    mt_out_shape = model.get_tensor_shape(mt_output)
                    mt_thres = consumer.input[1]
                    T = model.get_initializer(mt_thres)
                    assert (
                        T.shape[0] == 1 or T.shape[0] == mh
                    ), """First dimension of
                    thresholds neither 1 nor MH."""
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
                        domain="finn",
                        backend="fpgadataflow",
                        resType="ap_resource_lut()",
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
                        domain="finn",
                        backend="fpgadataflow",
                        resType="ap_resource_lut()",
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
                    )
                    graph.node.insert(node_ind, new_node)
                    # remove old node
                    graph.node.remove(n)
                    graph_modified = True
        if graph_modified:
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
            if n.op_type == "MatMul":
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
                    assert mh % pe == 0, "Requirement MH divisable by PE is violated."
                    assert (
                        mw % simd == 0
                    ), "Requirement MW divisable by SIMD is violated."
                    wmem = mw * mh // (pe * simd)
                    assert (
                        mw * mh == wmem * pe * simd
                    ), """Requirement (MW * MH) divisiable by
                    (WMEM * PE * SIMD) is violated."""
                    # see if we have any following thresholds
                    consumer = model.find_consumer(mm_output)
                    if consumer is not None and consumer.op_type == "MultiThreshold":
                        # TODO ensure integer thresholds?
                        # create MVTU (i.e. including activation)
                        mt_output = consumer.output[0]
                        mt_out_shape = model.get_tensor_shape(mt_output)
                        mt_thres = consumer.input[1]
                        T = model.get_initializer(mt_thres)
                        assert (
                            T.shape[0] == 1 or T.shape[0] == mh
                        ), """First dimension of
                        thresholds neither 1 nor MH."""
                        odt = model.get_tensor_datatype(mt_output)
                        scale = getCustomOp(consumer).get_nodeattr("out_scale")
                        assert (
                            scale == 1.0
                        ), "out_scale must be equal to 1.0 for HLS conversion."
                        actval = getCustomOp(consumer).get_nodeattr("out_bias")
                        assert (
                            int(actval) == actval
                        ), "out_bias must be integer for HLS conversion."
                        actval = int(actval)
                        assert (not odt.signed()) or (
                            actval < 0
                        ), "Signed output requres actval < 0"
                        model.set_tensor_shape(mm_input, mm_in_shape)
                        model.set_tensor_shape(mt_output, mt_out_shape)
                        # create and insert new StreamingFCLayer node
                        new_node = helper.make_node(
                            "StreamingFCLayer_Batch",
                            [mm_input, mm_weight, mt_thres],
                            [mt_output],
                            domain="finn",
                            backend="fpgadataflow",
                            resType="ap_resource_lut()",
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
                            domain="finn",
                            backend="fpgadataflow",
                            resType="ap_resource_lut()",
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
                        )
                        graph.node.insert(node_ind, new_node)
                        # remove old node
                        graph.node.remove(n)
                        graph_modified = True
        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class InferThresholdingLayer(Transformation):
    """Convert any MultiThreshold into a standalone thresholding HLS layer."""

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
                idt = model.get_tensor_datatype(thl_input)

                # skip conversion for layers with float input
                if not idt.is_integer():
                    continue

                # skip conversion if input is not NHWC or NC
                thl_in_layout = model.get_tensor_layout(thl_input)
                if thl_in_layout != DataLayout.NHWC and thl_in_layout != DataLayout.NC:
                    continue

                # now safe to assume number of channels is in last dimension
                ifc = int(thl_in_shape[-1])
                # create node with no parallelization first
                pe = 1
                assert ifc % pe == 0, "Requirement IFC divisable by PE is violated."

                odt = model.get_tensor_datatype(thl_output)
                # create and insert new StreamingFCLayer node
                new_node = helper.make_node(
                    "Thresholding_Batch",
                    [thl_input, thl_threshold],
                    [thl_output],
                    domain="finn",
                    backend="fpgadataflow",
                    NumChannels=ifc,
                    PE=pe,
                    inputDataType=idt.name,
                    outputDataType=odt.name,
                    numInputVectors=list(thl_in_shape[:-1]),
                )
                graph.node.insert(node_ind, new_node)
                # remove old node
                graph.node.remove(node)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class InferChannelwiseLinearLayer(Transformation):
    """Convert any channel-wise Add/Mul into a HLS layer."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            if node.op_type == "Add" or node.op_type == "Mul":
                ll_input = node.input[0]
                ll_output = node.output[0]
                ll_in_shape = model.get_tensor_shape(ll_input)
                # check that:
                # input 1 is an initializer
                # with shape (ch, 1)
                # and initializer is integers
                ll_const = node.input[1]
                if ll_const is not None:
                    ll_cinit = model.get_initializer(ll_const)
                else:
                    continue

                ll_cinit_shape = list(ll_cinit.shape)
                # get number of channels from input
                ll_in_layout = model.get_tensor_layout(ll_input)
                if ll_in_layout == DataLayout.NHWC or ll_in_layout == DataLayout.NC:
                    ch = ll_in_shape[-1]
                elif ll_in_layout == DataLayout.NCHW:
                    ch = ll_in_shape[1]
                else:
                    continue

                # check if the shape of initializer is compatible with
                # number of channels, e.g. (ch,1) or (ch)
                # TODO: verify plausible shapes
                if np.prod(ll_cinit_shape) != ch:
                    continue

                # check initializer contains integers as floats
                if not (ll_cinit.astype(np.int32) == ll_cinit).all():
                    continue

                # all initializer conditions are met
                # check inputs
                idt = model.get_tensor_datatype(ll_input)

                # skip conversion for layers with float input
                if not idt.is_integer():
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

                # create node with no parallelization first
                pe = 1
                assert ch % pe == 0, "Requirement IFC divisable by PE is violated."

                # set function and determine output data type
                if node.op_type == "Add":
                    func = "add"
                    if idt.signed():
                        odt = DataType.get_smallest_possible(2 * idt.min())
                    else:
                        odt = DataType.get_smallest_possible(2 * idt.max())
                elif node.op_type == "Mul":
                    func = "mul"
                    if idt.signed():
                        odt = DataType.get_smallest_possible(abs(idt.min()) * idt.min())
                    else:
                        odt = DataType.get_smallest_possible(idt.max() * idt.max())
                model.set_initializer(ll_const, ll_cinit.reshape(ch))
                model.set_tensor_datatype(ll_output, odt)
                # create and insert node
                new_node = helper.make_node(
                    "ChannelwiseOp_Batch",
                    [ll_input, ll_const],
                    [ll_output],
                    domain="finn",
                    backend="fpgadataflow",
                    Func=func,
                    NumChannels=ch,
                    PE=pe,
                    inputDataType=idt.name,
                    outputDataType=odt.name,
                    numInputVectors=list(ll_in_shape[:-1]),
                )
                graph.node.insert(insert_point, new_node)
                # remove old node
                graph.node.remove(node)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)
