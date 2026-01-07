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


import numpy as np
import qonnx.core.data_layout as DataLayout
import warnings
from onnx import NodeProto, TensorProto, helper
from qonnx.core.datatype import DataType

# QONNX wrapper to ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import SortGraph
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name
from qonnx.util.onnx import nchw_to_nhwc

# Module containing specializations of elementwise binary operations
import finn.custom_op.fpgadataflow.elementwise_binary as elementwise_binary


class InferConvInpGen(Transformation):
    """Convert Im2Col layers to ConvolutionInputGenerator layers."""

    def __init__(self):
        super().__init__()

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
                    warnings.warn("%s : Input is not int. Can't infer ConvInpGen." % n.name)
                    continue
                i2c_inst = getCustomOp(n)
                stride_h, stride_w = i2c_inst.get_nodeattr("stride")
                k_h, k_w = i2c_inst.get_nodeattr("kernel_size")
                pad_attr = i2c_inst.get_nodeattr("pad_amount")
                pad_h = pad_attr[0] + pad_attr[2]
                pad_w = pad_attr[1] + pad_attr[3]
                dilation_h, dilation_w = i2c_inst.get_nodeattr("dilations")
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
                    assert pad_val == 0, (
                        "%s : FMPadding_Batch doesn't currently support pad_val!= 0" % n.name
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
                        "FMPadding",
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

                is_1D = (ifm_dim_h == 1) or (ifm_dim_w == 1)
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
                    is1D=is_1D,
                    name="ConvolutionInputGenerator_" + n.name,
                )
                graph.node.insert(ConvInpGen_node_idx, ConvInpGen_node)
                # remove old nodes
                graph.node.remove(n)
                graph_modified = True
        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class InferThresholdingLayer(Transformation):
    """Convert any MultiThreshold into a standalone thresholding HLS layer."""

    def __init__(self):
        super().__init__()

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
                tdt = model.get_tensor_datatype(thl_threshold)

                # only infer layers where input and thresholds are integers, floats, or fixed-point
                idt_int = idt.is_integer()
                tdt_int = tdt.is_integer()
                idt_fp = idt in ["FLOAT32", "FLOAT16"]
                tdt_fp = tdt in ["FLOAT32", "FLOAT16"]
                idt_fxp = idt.is_fixed_point()
                tdt_fxp = tdt.is_fixed_point()
                if not (idt_int or idt_fp or idt_fxp):
                    continue
                if not (tdt_int or tdt_fp or tdt_fxp):
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
                    node.name + ": MultiThreshold out_scale must be 1 for HLS conversion."
                )
                actval = getCustomOp(node).get_nodeattr("out_bias")
                assert int(actval) == actval, (
                    node.name + ": MultiThreshold out_bias must be integer for HLS conversion."
                )
                actval = int(actval)

                # a signed activation should always have a negative bias,
                # but BIPOLAR uses the -1 as 0 encoding so the assert does not apply
                if odt != DataType["BIPOLAR"]:
                    assert (not odt.signed()) or (actval < 0), (
                        node.name + ": Signed output requires actval < 0"
                    )

                new_node = helper.make_node(
                    "Thresholding",
                    [thl_input, thl_threshold],
                    [thl_output],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    NumChannels=ifc,
                    PE=pe,
                    numSteps=thl_thres_shape[1],
                    inputDataType=idt.name,
                    weightDataType=tdt.name,
                    outputDataType=odt.name,
                    numInputVectors=list(thl_in_shape[:-1]),
                    ActVal=actval,
                    name="Thresholding_" + node.name,
                )

                graph.node.insert(insert_point, new_node)
                # remove old node
                graph.node.remove(node)
                graph_modified = True

        return (model, graph_modified)


class InferUpsample(Transformation):
    """Convert Upsample and Resize nodes to layers to UpsampleNearestNeighbour nodes."""

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
                        "%s: Input not int. Can't infer UpsampleNearestNeighbour." % n.name
                    )
                    continue

                if model.get_tensor_layout(n.input[0]) != DataLayout.NHWC:
                    warnings.warn(
                        "%s: Input not NHWC. Can't infer UpsampleNearestNeighbour." % n.name
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
                assert scales[0] == scales[3] == 1, (
                    n.name + ": Upsampling is only supported for scales with "
                    "the first and last dimensions being 1 in NHWC."
                )

                # Extract information for HW node
                HI = in_shape[1]
                WI = in_shape[2]
                HO = int(round(HI * scales[1]))
                WO = int(round(WI * scales[2]))
                NumChannels = in_shape[-1]
                batchSize = in_shape[0]
                inputDataType = dt.name

                # Insert the HWCustomOp node
                Upsample_HW_node = helper.make_node(
                    "UpsampleNearestNeighbour",
                    [n.input[0]],
                    [n.output[0]],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    SIMD=1,
                    HO=HO,
                    WO=WO,
                    HI=HI,
                    WI=WI,
                    NumChannels=NumChannels,
                    inputDataType=inputDataType,
                    batchSize=batchSize,
                    name="UpsampleNearestNeighbour_" + n.name,
                    cpp_interface="hls_vector",
                    hls_style="freerunning",
                )

                # Remove the old node
                graph.node.insert(node_ind, Upsample_HW_node)
                # remove old nodes
                graph.node.remove(n)
                graph_modified = True
        return (model, graph_modified)


class InferAddStreamsLayer(Transformation):
    """Convert any Add into a AddStreams HW layer."""

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
                in0_static = not (model.get_initializer(in0) is None)
                in1_static = not (model.get_initializer(in1) is None)

                # skip if different shapes on inputs
                if in0_shape != in1_shape:
                    continue
                # skip if any of inputs have initializers
                # (this node is meant for adding two dynamic streams)
                if in0_static or in1_static:
                    continue

                idt0 = model.get_tensor_datatype(in0)
                idt1 = model.get_tensor_datatype(in1)

                # skip conversion for layers with float input
                if not idt0.is_integer() or not idt1.is_integer():
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

                # keep track of where we need to insert the HW Op
                # it has to be ahead of the output transform
                insert_point = node_ind

                if result_layout == DataLayout.NCHW:
                    result = nchw_to_nhwc(result, model, node_ind, reverse=True)
                    node_ind += 1

                # now safe to assume num_channels is size of last dimension
                num_channels = int(in0_shape[-1])
                # create node with no parallelization first
                pe = 1

                # create and insert new AddStreams node
                new_node = helper.make_node(
                    "AddStreams",
                    [in0, in1],
                    [result],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    NumChannels=num_channels,
                    PE=pe,
                    inputDataTypes=[idt0.name, idt1.name],
                    numInputVectors=in0_shape[:-1],
                    name="AddStreams_" + node.name,
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
    """Insert a DuplicateStreams HW layer for any tensor with fanout == 2"""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        # check first if global input is split
        successors = model.find_consumers(graph.input[0].name)
        dt = model.get_tensor_datatype(graph.input[0].name)
        if successors is not None and len(successors) >= 2 and dt.is_integer():
            output_tensor = graph.input[0].name
            n_outputs = len(successors)
            dt = model.get_tensor_datatype(output_tensor)

            # create clone tensors
            out_shape = model.get_tensor_shape(output_tensor)
            out_tensor_clones = []
            for i in range(n_outputs):
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
                "DuplicateStreams",
                [output_tensor],
                out_tensor_clones,
                domain="finn.custom_op.fpgadataflow",
                backend="fpgadataflow",
                NumChannels=num_ch,
                PE=pe,
                inputDataType=dt.name,
                numInputVectors=vecs,
                NumOutputStreams=n_outputs,
                outFIFODepths=[2] * n_outputs,
                name="DuplicateStreams_" + output_tensor,
            )

            graph.node.insert(0, dup_node)

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

        for node in graph.node:
            node_ind += 1
            for output_tensor in node.output:
                successors = model.find_consumers(output_tensor)
                if successors is not None and len(successors) >= 2:
                    n_outputs = len(successors)

                    dt = model.get_tensor_datatype(output_tensor)

                    # skip conversion for layers with float input
                    if not dt.is_integer():
                        continue

                    # create clone tensors
                    out_shape = model.get_tensor_shape(output_tensor)
                    out_tensor_clones = []
                    for i in range(n_outputs):
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
                        "DuplicateStreams",
                        [output_tensor],
                        out_tensor_clones,
                        domain="finn.custom_op.fpgadataflow",
                        backend="fpgadataflow",
                        NumChannels=num_ch,
                        PE=pe,
                        inputDataType=dt.name,
                        numInputVectors=vecs,
                        NumOutputStreams=n_outputs,
                        outFIFODepths=[2] * n_outputs,
                        name="DuplicateStreams_" + node.name,
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
    """Convert any channel-wise Add/Mul into a HW layer."""

    def get_smallest_possible(self, vals):
        """Returns smallest (fewest bits) possible DataType that can represent
        value. Prefers unsigned integers where possible."""
        vals = np.array(vals, dtype=np.float64)
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
                    warnings.warn("Broadcasting " + str(node.op_type) + "(" + node.name + ")")
                    ll_cinit = np.full((ch), ll_cinit.flatten()[0])
                elif np.prod(ll_cinit_shape) != ch or ll_cinit_shape[ch_index] != ch:
                    # parameter shape not compatible with Channelwise
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

                # keep track of where we need to insert the HW Op
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
                    "ChannelwiseOp",
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
                    name="ChannelwiseOp_" + node.name,
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
    """Convert any TopK into a LabelSelect HW layer."""

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

                # create and insert new LabelSelect node
                new_node = helper.make_node(
                    "LabelSelect",
                    [fc_input],
                    [idx_output],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    Labels=num_labels,
                    PE=pe,
                    K=k,
                    inputDataType=idt.name,
                    numInputVectors=num_inp_vecs,
                    name="LabelSelect_" + node.name,
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
    """Convert any GlobalAveragePool into a GlobalAccPool HW layer and a scalar Mul."""

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

                # keep track of where we need to insert the HW Op
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
                    "GlobalAccPool",
                    [in0],
                    [pool_out],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    NumChannels=num_ch,
                    PE=pe,
                    inputDataType=idt.name,
                    numInputVectors=vecs,
                    name="GlobalAccPool_" + node.name,
                )

                mul_value = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(), TensorProto.FLOAT, [1]
                )
                model.graph.value_info.append(mul_value)
                model.set_initializer(
                    mul_value.name, np.array(1 / (vecs[1] * vecs[2]), dtype=np.float32)
                )
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


class InferPool(Transformation):
    """If kernel_shape > strides, replace Pool layer with  with of Im2col
    + pool(with kernel_shape == strides), plus Transpose layers to keep the original
    data layout."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            if node.op_type in ["MaxPool", "QuantAvgPool2d", "MaxPoolNHWC"]:
                node_input = node.input[0]
                ishape = model.get_tensor_shape(node_input)
                node_output = node.output[0]
                idt = model.get_tensor_datatype(node_input)
                oshape = model.get_tensor_shape(node_output)
                # only support 4D input tensors (1D convs need extra dummy dim)
                if len(ishape) != 4:
                    continue

                # extract pool parameters
                if node.op_type == "MaxPool":
                    kh, kw = list(get_by_name(node.attribute, "kernel_shape").ints)
                    sh, sw = list(get_by_name(node.attribute, "strides").ints)
                    dlayout = "NCHW"
                elif node.op_type == "QuantAvgPool2d":
                    inst = getCustomOp(node)
                    # QuantAvgPool2d has a single scalar attribute
                    # for kernel size and stride (implicit square)
                    kh = kw = inst.get_nodeattr("kernel")
                    sh = sw = inst.get_nodeattr("stride")
                    dlayout = inst.get_nodeattr("data_layout")
                elif node.op_type == "MaxPoolNHWC":
                    inst = getCustomOp(node)
                    kh, kw = inst.get_nodeattr("kernel_shape")
                    sh, sw = inst.get_nodeattr("strides")
                    dlayout = "NHWC"
                try:
                    pad = list(get_by_name(node.attribute, "pads").ints)
                except AttributeError:
                    pad = [0, 0, 0, 0]

                if not idt.is_integer():
                    continue

                if (kh < sh) or (kw < sw):
                    # TODO check/implement swg support
                    continue

                odt = model.get_tensor_datatype(node_output)

                if dlayout == "NCHW":
                    _, ifm_ch, ifm_h, ifm_w = ishape
                    _, ofm_ch, ofm_h, ofm_w = oshape
                elif dlayout == "NHWC":
                    _, ifm_h, ifm_w, ifm_ch = ishape
                    _, ofm_h, ofm_w, ofm_ch = oshape
                else:
                    raise Exception("Unknown dlayout: " + str(dlayout))

                # if data layout NCHW, we need transpose nodes surrounding
                # the hw layer
                if dlayout == "NCHW":
                    # create new intermediate values
                    inp_trans_out = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        (1, ifm_h, ifm_w, ifm_ch),  # NHWC
                    )
                    graph.value_info.append(inp_trans_out)
                    inp_trans_out = inp_trans_out.name
                    model.set_tensor_datatype(inp_trans_out, idt)

                    pool_output = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        (1, ofm_h, ofm_w, ofm_ch),
                    )
                    graph.value_info.append(pool_output)
                    pool_output = pool_output.name

                im2col_out = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    TensorProto.FLOAT,
                    (1, ofm_h, ofm_w, ifm_ch * kh * kw),
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
                pool_size_param = 0  # will be overridden if neededs
                pad_value = 0
                if node.op_type in ["MaxPool", "MaxPoolNHWC"]:
                    pool_fxn = "MaxPool"
                    odt = idt
                    pad_value = idt.min()
                elif node.op_type == "QuantAvgPool2d":
                    assert odt.is_integer(), """Output data type for QuantAvgPool2d
                    needs to be integer"""
                    assert all(x == 0 for x in pad), "Padding is not supported for QuantAvgPool2d"
                    inst = getCustomOp(node)
                    pool_fxn = "QuantAvgPool"
                    pool_size_param = inst.get_shifts()
                    accum_bits = inst.get_accum_size()

                else:
                    raise Exception(
                        "pad_value and pool_fxn not configured for {}".format(node.op_type)
                    )

                # format input tensor
                im2col_node = helper.make_node(
                    "Im2Col",
                    [im2col_in],
                    [im2col_out],
                    domain="qonnx.custom_op.general",
                    stride=[sh, sw],
                    kernel_size=[kh, kw],
                    pad_amount=pad,
                    pad_value=pad_value,
                    depthwise=1,
                    input_shape="(1,{},{},{})".format(ifm_h, ifm_w, ifm_ch),
                    name="Im2Col_" + node.name,
                )

                # Warning PE has to be equal to ifm_ch until Im2Col is replaced by
                # ConvolutionInputGenerator with depthwise=1.
                # For other settings the output will be incorrect due to incorrect input
                # data layout
                pool_node = helper.make_node(
                    "Pool",
                    [im2col_out],
                    [pool_output],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    InputDataType=idt.name,
                    OutputDataType=odt.name,
                    Channels=ifm_ch,
                    PE=ifm_ch,
                    KernelSize=[kh, kw],
                    Function=pool_fxn,
                    OutImgDims=[ofm_h, ofm_w],
                    AccumBits=accum_bits,
                    Size=pool_size_param,
                    BatchSize=1,
                    cpp_interface="hls_vector",
                    name="Pool_" + node.name,
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
                graph.node.remove(node)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class InferLookupLayer(Transformation):
    """Convert Gather nodes with constant op0 into Lookup HW layers."""

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


class InferConcatLayer(Transformation):
    """Convert suitable Concat nodes (operating on last/-1 axis)
    into StreamingConcat HW layers."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            if node.op_type == "Concat":
                ishape = model.get_tensor_shape(node.input[0])
                axis = get_by_name(node.attribute, "axis")
                if (axis is None) or (ishape is None):
                    continue
                axis = axis.i
                last_axis = len(ishape) - 1
                # skip conversion if not using last axis
                if (axis != -1) and (axis != last_axis):
                    continue
                # check datatype coherence
                if any([model.get_tensor_datatype(x) is None for x in node.input]):
                    warnings.warn(
                        "Inputs with undefined datatype detected, skipping InferConcatLayer()"
                    )
                    continue
                # skip conversion if any inputs are static
                any_static = any([model.get_initializer(x) is not None for x in node.input])
                if any_static:
                    continue
                # skip conversion if inputs are not integers
                all_integer = all([model.get_tensor_datatype(x).is_integer() for x in node.input])
                if not all_integer:
                    warnings.warn(
                        "Inputs with non-integer datatype detected, skipping InferConcatLayer()"
                    )
                    continue
                # ready for conversion
                channels_per_stream = [model.get_tensor_shape(x)[-1] for x in node.input]
                inp_vec = list(model.get_tensor_shape(node.input[0])[:-1])
                new_node = helper.make_node(
                    "StreamingConcat",
                    node.input,
                    node.output,
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    name="StreamingConcat_" + node.name,
                    SIMD=1,
                    ChannelsPerStream=channels_per_stream,
                    inputDataTypes=[model.get_tensor_datatype(x).name for x in node.input],
                    numInputVectors=inp_vec,
                    inFIFODepths=[2] * len(node.input),
                    cpp_interface="hls_vector",
                    hls_style="freerunning",
                )
                graph.node.insert(node_ind, new_node)
                # remove old node
                graph.node.remove(node)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class InferSplitLayer(Transformation):
    """Convert suitable Split nodes (operating on last/-1 axis)
    into StreamingConcat HW layers."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            if node.op_type == "Split":
                split_param = node.input[1]
                if model.get_initializer(split_param) is None:
                    warnings.warn("Split param not constant, skipping InferSplitLayer()")
                    continue
                ishape = model.get_tensor_shape(node.input[0])
                axis = get_by_name(node.attribute, "axis")
                if (axis is None) or (ishape is None):
                    continue
                axis = axis.i
                last_axis = len(ishape) - 1
                # skip conversion if not using last axis
                if (axis != -1) and (axis != last_axis):
                    warnings.warn(
                        "StreamingSplit supports only last axis, skipping InferSplitLayer()"
                    )
                    continue
                # only one input allowed (two including split_param)
                if len(node.input) != 2:
                    warnings.warn("Only one input allowed, skipping InferSplitLayer()")
                    continue
                # skip conversion if the input is static
                if model.get_initializer(node.input[0]) is not None:
                    warnings.warn("Static input detected, skipping InferSplitLayer()")
                    continue
                # skip conversion if inputs are not integers
                if not model.get_tensor_datatype(node.input[0]).is_integer():
                    warnings.warn("Non-integer input detected, skipping InferSplitLayer()")
                    continue
                # ready for conversion
                channels_per_stream = [model.get_tensor_shape(x)[-1] for x in node.output]
                inp_vec = list(model.get_tensor_shape(node.input[0])[:-1])
                # when creating the fpgadataflow node we remove the second parameter input
                new_node = helper.make_node(
                    "StreamingSplit",
                    [node.input[0]],
                    node.output,
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    name="StreamingSplit_" + node.name,
                    SIMD=1,
                    cpp_interface="hls_vector",
                    hls_style="freerunning",
                    ChannelsPerStream=channels_per_stream,
                    inputDataType=model.get_tensor_datatype(node.input[0]).name,
                    numInputVectors=inp_vec,
                    outFIFODepths=[2] * len(node.output),
                )
                graph.node.insert(node_ind, new_node)
                # remove old node
                graph.node.remove(node)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class InferStreamingEltwise(Transformation):
    """Convert eltwise Add, Sub or Sub -> Abs to StreamingEltwise layer
    with AddEltwise, SubEltwise or AbsDiffEltwise op."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            if node.op_type in ["Sub", "Add"]:
                in0 = node.input[0]
                in1 = node.input[1]
                result = node.output[0]
                in0_shape = model.get_tensor_shape(in0)
                in1_shape = model.get_tensor_shape(in1)
                in0_static = not (model.get_initializer(in0) is None)
                in1_static = not (model.get_initializer(in1) is None)

                # skip if different shapes on inputs
                if in0_shape != in1_shape:
                    continue
                # skip if any of inputs have initializers
                # (this node is meant for two dynamic streams)
                if in0_static or in1_static:
                    continue

                idt0 = model.get_tensor_datatype(in0)
                idt1 = model.get_tensor_datatype(in1)

                # skip conversion for layers with float input
                if not (idt0.is_integer() and idt1.is_integer()):
                    continue

                eltwiseOp = node.op_type
                nodes_to_remove = [node]
                if node.op_type == "Sub":
                    # look for a downstream Abs node
                    res_consumer = model.find_consumer(result)
                    if (res_consumer is not None) and (res_consumer.op_type == "Abs"):
                        eltwiseOp = "AbsDiff"
                        result = res_consumer.output[0]
                        nodes_to_remove.append(res_consumer)

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

                # keep track of where we need to insert the HW Op
                # it has to be ahead of the output transform
                insert_point = node_ind

                if result_layout == DataLayout.NCHW:
                    result = nchw_to_nhwc(result, model, node_ind, reverse=True)
                    node_ind += 1

                # now safe to assume num_channels is size of last dimension
                num_channels = int(in0_shape[-1])
                # create node with no parallelization first
                pe = 1

                # create and insert new Eltwise node
                new_node = helper.make_node(
                    "StreamingEltwise",
                    [in0, in1],
                    [result],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    NumChannels=num_channels,
                    PE=pe,
                    inputDataType0=idt0.name,
                    inputDataType1=idt1.name,
                    eltwiseOp=eltwiseOp,
                    numInputVectors=in0_shape[:-1],
                    name="StreamingEltwise_" + node.name,
                )
                graph.node.insert(insert_point, new_node)
                # remove old nodes
                for nd in nodes_to_remove:
                    graph.node.remove(nd)
                graph_modified = True

        return (model, graph_modified)


class InferBinaryMatrixVectorActivation(Transformation):
    """Convert XnorPopcountMatMul layers to
    MatrixVectorActivation layers. Any immediately following MultiThreshold
    layers will also be absorbed into the MVTU."""

    def __init__(self):
        super().__init__()

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
                input for xnorpopcount is not Wset to FINN DataType BINARY."""
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
                    # create and insert new MatrixVectorActivation node
                    new_node = helper.make_node(
                        "MVAU",
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
                    # create and insert new MatrixVectorActivation node
                    new_node = helper.make_node(
                        "MVAU",
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
                        name=n.name,
                    )
                    graph.node.insert(node_ind, new_node)
                    # remove old node
                    graph.node.remove(n)
                    graph_modified = True
        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class InferQuantizedMatrixVectorActivation(Transformation):
    """Convert MatMul layers with quantized inputs and weights to
    MatrixVectorActivation layers."""

    def __init__(self):
        super().__init__()

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
                W = model.get_initializer(mm_weight)
                if idt.is_integer() and wdt.is_integer():
                    # extract weight shape, note that ONNX and finn-hlslib
                    # make different assumptions about dim order here
                    # ONNX assumes W has (in, out) shape
                    # finn-hlslib assumes W has (out, in) shape
                    if W is None:
                        # dynamic
                        mm_dyn_shape = model.get_tensor_shape(mm_weight)
                        mh = int(mm_dyn_shape[-1])
                        mw = int(mm_dyn_shape[-2])
                    else:
                        # static
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
                            consumer.name + ": out_bias must be integer for HLS conversion."
                        )
                        actval = int(actval)
                        odt_is_bipolar = odt == DataType["BIPOLAR"]
                        bipolar_ok = odt_is_bipolar and (scale == 2.0) and (actval == -1)
                        assert scale == 1.0 or bipolar_ok, (
                            consumer.name + ": out_scale=1 or bipolar output needed for conversion."
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
                        # create and insert new MatrixVectorActivation node
                        new_node = helper.make_node(
                            "MVAU",
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
                            name="MVAU_" + n.name,
                            dynamic_input=W is None,
                            inFIFODepths=[2, 2] if W is None else [2],
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
                        # create and insert new MatrixVectorActivation node
                        new_node = helper.make_node(
                            "MVAU",
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
                            name="MVAU_" + n.name,
                            dynamic_input=W is None,
                            inFIFODepths=[2, 2] if W is None else [2],
                        )
                        graph.node.insert(node_ind, new_node)
                        # remove old node
                        graph.node.remove(n)
                        graph_modified = True
        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class InferVectorVectorActivation(Transformation):
    """Convert MatMul layers with quantized inputs and weights to
    VectorVectorActivation layers, if the sparsity annotation
    of the weight matrix indicates that the MatMul layer belongs to
    a depthwise convolution. Any immediately following MultiThreshold
    layers will also be absorbed into the VVAU."""

    def __init__(self):
        super().__init__()

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "MatMul" and model.get_tensor_sparsity(n.input[1]) is not None:
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
                    w_tensor = np.zeros((channels, 1, k_h, k_w), dtype=np.float32)
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
                            consumer.name + ": out_scale must be equal to 1.0 for HLS conversion."
                        )
                        actval = getCustomOp(consumer).get_nodeattr("out_bias")
                        assert int(actval) == actval, (
                            consumer.name + ": out_bias must be integer for HLS conversion."
                        )
                        actval = int(actval)
                        assert (not odt.signed()) or (actval < 0), (
                            consumer.name + ": Signed output requres actval < 0"
                        )
                        model.set_tensor_shape(mm_input, mm_in_shape)
                        model.set_tensor_shape(mt_output, mt_out_shape)
                        # create and insert new VectorVectorActivation node
                        new_node = helper.make_node(
                            "VVAU",
                            [mm_input, mm_weight, mt_thres],
                            [mt_output],
                            domain="finn.custom_op.fpgadataflow",
                            backend="fpgadataflow",
                            PE=pe,
                            Dim=[mm_in_shape[1], mm_in_shape[2]],
                            Channels=channels,
                            Kernel=[k_h, k_w],
                            inputDataType=idt.name,
                            weightDataType=wdt.name,
                            outputDataType=odt.name,
                            ActVal=actval,
                            noActivation=0,
                            name="VVAU_" + n.name,
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
                            "VVAU",
                            [mm_input, mm_weight],
                            [mm_output],
                            domain="finn.custom_op.fpgadataflow",
                            backend="fpgadataflow",
                            PE=pe,
                            Dim=[mm_in_shape[1], mm_in_shape[2]],
                            Channels=channels,
                            Kernel=[k_h, k_w],
                            inputDataType=idt.name,
                            weightDataType=wdt.name,
                            outputDataType=odt.name,
                            ActVal=0,
                            noActivation=1,
                            name="VVAU_" + n.name,
                        )
                        graph.node.insert(node_ind, new_node)
                        # remove old node
                        graph.node.remove(n)
                        graph_modified = True
        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class InferHWSoftmax(Transformation):
    """
    Infers a regular softmax node without merging the multithreshold
    and setting the softmax to perform the quantisation.
    """

    def __init__(self):
        super().__init__()

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            if n.op_type == "Softmax":
                input_shape = model.get_tensor_shape(n.input[0])
                idt0 = model.get_tensor_datatype(n.input[0])
                odt0 = model.get_tensor_datatype(n.output[0])
                new_node = helper.make_node(
                    "HWSoftmax",
                    [n.input[0]],  # input tensor(s)
                    [n.output[0]],  # output tensor(s)
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    ifm_dim=input_shape,
                    input_data_type=idt0.name,
                    output_data_type=odt0.name,
                    name=n.name,
                    SIMD=1,
                    NumChannels=input_shape[-1],
                    cpp_interface="hls_vector",
                    hls_style="freerunning",
                )
                graph.node.insert(node_ind, new_node)
                graph.node.remove(n)
        return (model, graph_modified)


class InferShuffle(Transformation):
    """
    Find transpose layers with (optionally) reshape layers around them
    and convert them into a shuffle operator
    """

    def __init__(self):
        super().__init__()

    def _is_streaming_ptranspose(self, perm, shape):
        """
        Check if the permutation represents a streaming InnerShuffle case.
        A streaming InnerShuffle works when the last two dimensions are swapped,
        regardless of how many outer dimensions there are.
        """
        if len(perm) < 2 or len(shape) < 2:
            return False

        # Check if last two dimensions are swapped while others stay in order
        expected_perm = list(range(len(perm) - 2)) + [len(perm) - 1, len(perm) - 2]
        return perm == expected_perm

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        for node_ind, n in enumerate(graph.node, start=1):
            if n.op_type == "Transpose":
                to_remove = [n]

                new_in_tensor = None
                new_out_tensor = None

                perm = n.attribute[0]

                new_in_tensor = n.input[0]
                in_shape = model.get_tensor_shape(n.input[0])
                in_reshaped = in_shape

                # Detect a reshape at the input and capture it
                producer = model.find_producer(n.input[0])
                if producer is not None:
                    if producer.op_type == "Reshape":
                        new_in_tensor = producer.input[0]
                        in_shape = model.get_tensor_shape(new_in_tensor)
                        in_reshaped = model.get_tensor_shape(n.input[0])
                        to_remove.append(producer)

                new_out_tensor = n.output[0]
                out_shape = model.get_tensor_shape(new_out_tensor)
                out_reshaped = out_shape

                # Detect a reshape at the output and capture it
                consumer = model.find_consumer(n.output[0])
                if consumer is not None:
                    if consumer.op_type == "Reshape":
                        new_out_tensor = consumer.output[0]
                        out_shape = model.get_tensor_shape(n.output[0])
                        out_reshaped = model.get_tensor_shape(new_out_tensor)
                        to_remove.append(consumer)

                # Handle None shapes (shape inference might have failed)
                assert (
                    in_reshaped is not None
                ), f"""Could not infer shape for tensor {n.input[0]}.
                    Please run InferShapes first"""
                assert (
                    out_reshaped is not None
                ), f"""Could not infer shape for tensor {new_out_tensor}.
                    Please run InferShapes first"""

                idt = model.get_tensor_datatype(new_in_tensor)
                odt = model.get_tensor_datatype(new_out_tensor)

                # Some sanity checks for the transformation
                if idt != odt:
                    raise RuntimeError(
                        """
                    Input datatype and output datatype of the shuffle must be the same,
                    did something go wrong during transformation?
                    """
                    )

                if len(perm.ints) != len(in_reshaped):
                    raise RuntimeError(
                        f"""
                    Permutation list {perm.ints=} does not match the reshaped input dimension
                    {in_reshaped=}
                    """
                    )

                if len(perm.ints) != len(out_shape):
                    raise RuntimeError(
                        f"""
                    Permutation list {perm.ints=} does not match the reshaped out dimension
                    {out_reshaped=}
                    """
                    )

                simd = 1

                new_node = helper.make_node(
                    "Shuffle",
                    [new_in_tensor],
                    [new_out_tensor],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    in_shape=in_shape,
                    transpose_in_shape=in_reshaped,
                    out_shape=out_reshaped,
                    transpose_out_shape=out_shape,
                    data_type=idt.name,
                    name=f"Shuffle_{n.name}",
                    SIMD=simd,
                    NumChannels=in_reshaped[-1],
                )
                new_node.attribute.extend([perm])
                graph.node.insert(node_ind, new_node)

                for i in to_remove:
                    graph.node.remove(i)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())

        return (model, graph_modified)


# Lifts scalar to rank-1 tensor
def lift_to_rank1(name: str, model: ModelWrapper):
    # Scalars have a shape of lengths zero
    if len(model.get_tensor_shape(name)) == 0:
        # Lift shape to rank-1 tensor with single element
        model.set_tensor_shape(name, [1])
        # Check whether this tensor has an initializer
        if (tensor := model.get_initializer(name)) is not None:
            # Set new initializer tensor of shape [1]
            model.set_initializer(name, tensor.reshape(1))


# Converts supported elementwise binary operations to their FINN custom
# operation
class InferElementwiseBinaryOperation(Transformation):
    # Filter function to filter out the last elementwise Mul operation,
    # typically corresponding to output de-quantization, which should happen
    # off-chip
    @staticmethod
    def reject_output_dequant(model: ModelWrapper, node: NodeProto):
        # The operator must be a Mul and have no successor nodes
        if node.op_type == "Mul" and not model.find_direct_successors(node):
            # If the output is a floating-point tensors, reject this
            if model.get_tensor_datatype(node.output[0]) == "FLOAT32":
                # Filter False rejects this node
                return False
        # Filter True accepts this node
        return True

    # Initializes the transformation method with an optional filter function
    def __init__(self, _filter=None):
        # Initialize the base class Transformation object
        super().__init__()
        # Register the filter function as attribute
        self._filter = _filter if _filter is not None else lambda *_: True

    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Skip transforming nodes rejected by the filter
            if not self._filter(model, node):
                continue
            # If a custom operation with corresponding name is implemented in
            # the module, this operator is supported for conversion
            if f"Elementwise{node.op_type}" in dir(elementwise_binary):
                in0 = node.input[0]
                in1 = node.input[1]
                # if both inputs are constant, throw an error and
                # ask user to run FoldConstants transform first
                assert (
                    model.get_initializer(in0) is None or model.get_initializer(in1) is None
                ), """Both inputs are constant,
                    please run FoldConstants from qonnx.transformation.fold_constants first."""
                result = node.output[0]

                # Need to "lift" potential scalar inputs to rank-1 tensors
                lift_to_rank1(in0, model)
                lift_to_rank1(in1, model)

                in0_shape = model.get_tensor_shape(in0)
                in1_shape = model.get_tensor_shape(in1)
                out_shape = model.get_tensor_shape(result)

                idt0 = model.get_tensor_datatype(in0)
                idt1 = model.get_tensor_datatype(in1)
                odt0 = model.get_tensor_datatype(result)

                new_node = helper.make_node(
                    f"Elementwise{node.op_type}",
                    [in0, in1],
                    [result],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    lhs_shape=in0_shape,
                    rhs_shape=in1_shape,
                    out_shape=out_shape,
                    lhs_dtype=str(idt0),
                    rhs_dtype=str(idt1),
                    out_dtype=str(odt0),
                )
                graph.node.insert(index + 1, new_node)
                graph.node.remove(node)

                # Consider the graph to be modified, triggering exhaustive
                # re-application of this transformation
                graph_modified = True
                # Exiting here triggers type and shape inference and cleanup
                # after each transformed node. This helps QONNX to behave
                # better / more consistent in certain cases...
                break
        # Re-do shape and data type annotations after potential changes to the
        # model graph
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified


class InferLayerNorm(Transformation):
    """Convert LayerNorm into HW, only norming over channel dim.
    This transform is adapted from Brainsmith InferLayerNorm."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            if node.op_type == "LayerNormalization":
                scale = model.get_initializer(node.input[1])
                if len(node.input) > 2:
                    bias = model.get_initializer(node.input[2])
                else:
                    bias = None
                scale_is_one = (scale == 1).all()
                bias_is_zero = not np.any(bias)
                if not (scale_is_one and (bias_is_zero or bias is not None)):
                    warnings.warn(
                        f"""{node.name}: Scale is not one or bias is not zero.
                        Can't be converted to HWCustomOp. Please run ExtractNormScaleBias first."""
                    )
                    continue
                act_in = node.input[0]
                act_out = node.output[0]
                # Get any shape info that needs reuse
                shape_in = model.get_tensor_shape(act_in)
                # Get datatypes
                idt = model.get_tensor_datatype(act_in)
                odt = model.get_tensor_datatype(act_out)
                if not idt == odt == DataType["FLOAT32"]:
                    warnings.warn(
                        f"""{node.name}: Datatype is not float32,
                        currently only fp32 layernorm is supported."""
                    )
                    continue

                norm_axis = helper.get_node_attr_value(node, "axis")
                if model.get_tensor_layout(act_in) == DataLayout.NCHW:
                    act_in = nchw_to_nhwc(act_in, model, node_ind)
                    node_ind += 1
                    shape_in = model.get_tensor_shape(act_in)
                    # shift axis for norm appropriately
                    norm_axis = (norm_axis + 2) % 4
                ch = shape_in[-1]

                # keep track of where we need to insert the HLS Op
                # it has to be ahead of the output transform
                insert_point = node_ind
                if model.get_tensor_layout(act_out) == DataLayout.NCHW:
                    act_out = nchw_to_nhwc(act_out, model, node_ind, reverse=True)
                    node_ind += 1

                # Check if 1D, norming on channel axis
                if not (norm_axis == -1 or norm_axis == len(shape_in) - 1):
                    continue

                # create node with no parallelization first
                simd = 1
                assert ch % simd == 0, "Requirement IFC divisable by PE is violated."
                # create and insert nodes
                new_node = helper.make_node(
                    "LayerNorm",
                    [act_in],
                    [act_out],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    SIMD=simd,
                    ifm_dim=shape_in,
                    epsilon=helper.get_node_attr_value(node, "epsilon"),
                    inputDataType=idt.name,
                    outputDataType=odt.name,
                    name="LayerNorm_" + node.name,
                )
                graph.node.insert(insert_point, new_node)
                # remove old node
                graph.node.remove(node)

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


def elements_are_consecutive(indices):
    if indices.size == 1:
        return True
    else:
        indices.sort()
        return np.all(np.diff(indices) == 1)


class InferCrop(Transformation):
    """
    Find gather layers that can be converted into a Crop layer
    and replace them with a Crop layer
    """

    def __init__(self):
        super().__init__()

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Gather":
                # ensure that the indices input is an initializer
                if model.get_initializer(n.input[1]) is None:
                    continue

                # ensure that the axis is among the two innermost dimensions
                input_shape = model.get_tensor_shape(n.input[0])
                assert (
                    len(input_shape) > 1
                ), "Input shape needs to be at least 2D to be converted to Crop."

                max_index = len(input_shape) - 1
                axis = get_by_name(n.attribute, "axis").i
                if len(input_shape) >= 3:
                    assert axis in [
                        max_index - 1,
                        max_index - 2,
                    ], "Crop Operates on height and width of the input, assuming (N)HWC layout."
                else:
                    assert (
                        axis == max_index - 1
                    ), "Crop Operates on width of the input, for 2D input assuming WC layout."
                is_vertical = axis == max_index  # otherwise horizontal
                assert is_vertical is False, "This operator does not current support vertical crops"

                # assume that the indices input is an int64 scalar or array
                indices = model.get_initializer(n.input[1])
                assert indices.dtype == np.int64, "Indices must be int64"
                # Handle both scalar (0-d) and array cases
                if indices.ndim == 0:
                    # Single scalar index - always consecutive
                    indices_to_check = np.array([indices.item()])
                else:
                    indices_to_check = indices
                assert elements_are_consecutive(indices_to_check), "Indices must be consecutive"

                idt0 = model.get_tensor_datatype(n.input[0])

                crop_north = 0
                crop_east = 0
                crop_west = 0
                crop_south = 0
                num_inp_vec = [0]

                if len(input_shape) >= 3:
                    height_ind = len(input_shape) - 3
                    width_ind = len(input_shape) - 2
                    channels_ind = len(input_shape) - 1

                    height = input_shape[height_ind]
                    width = input_shape[width_ind]
                    channels = input_shape[channels_ind]
                    # save other dimensions in numInpVectors
                    if len(input_shape) > 3:
                        num_inp_vec = list(input_shape[:height_ind])

                    crop_min = int(np.min(indices_to_check))
                    crop_max = input_shape[axis] - int(np.max(indices_to_check)) - 1

                    if axis == height_ind:
                        crop_north = crop_min
                        crop_south = crop_max
                    elif axis == width_ind:
                        crop_west = crop_min
                        crop_east = crop_max

                elif len(input_shape) == 2:
                    # if there are only two dimensions, assume
                    height = 0
                    width_ind = len(input_shape) - 2
                    channels_ind = len(input_shape) - 1
                    width = input_shape[width_ind]
                    channels = input_shape[channels_ind]

                    # axis is on width dimension
                    crop_west = int(np.min(indices_to_check))
                    crop_east = input_shape[axis] - int(np.max(indices_to_check)) - 1

                # create and insert new node
                new_node = helper.make_node(
                    "Crop",
                    [n.input[0]],  # input tensor(s)
                    [n.output[0]],  # output tensor(s)
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    DataType=idt0.name,
                    name="Crop" + n.name,
                    SIMD=1,
                    ImgDim=[height, width],
                    NumChannels=channels,
                    CropNorth=crop_north,
                    CropEast=crop_east,
                    CropWest=crop_west,
                    CropSouth=crop_south,
                    numInputVectors=num_inp_vec,
                    cpp_interface="hls_vector",
                    hls_style="freerunning",
                )
                graph.node.insert(node_ind, new_node)
                graph.node.remove(n)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)
