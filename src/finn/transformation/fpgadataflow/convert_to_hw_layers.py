# Copyright (C) 2023, Advanced Micro Devices, Inc.
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
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import SortGraph
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.onnx import nchw_to_nhwc


class InferStreamingMaxPool(Transformation):
    """Convert MaxPoolNHWC layers to StreamingMaxPool HW layers."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            if node.op_type == "MaxPoolNHWC":
                mp_input = node.input[0]
                mp_output = node.output[0]
                mp_in_shape = model.get_tensor_shape(mp_input)
                # mp_out_shape = model.get_tensor_shape(mp_output)
                dt = model.get_tensor_datatype(mp_input)
                mp_inst = getCustomOp(node)
                k_h, k_w = mp_inst.get_nodeattr("kernel_shape")
                ifm_ch = mp_in_shape[-1]
                ifm_dim_h = mp_in_shape[1]
                ifm_dim_w = mp_in_shape[2]
                pe = 1
                ceil_mode = mp_inst.get_nodeattr("ceil_mode")
                is_1d = (ifm_dim_h == 1 and k_h == 1) or (ifm_dim_w == 1 and k_w == 1)
                is_divisable = (ifm_dim_h % k_h == 0) or (ifm_dim_w % k_w == 0)
                is_bipolar = dt == DataType["BIPOLAR"]
                pass_1d = is_1d and (not is_bipolar)
                pass_2d = (not is_1d) and is_divisable
                if pass_1d or pass_2d:
                    # create equivalent StreamingMaxPool_Batch node
                    new_node = helper.make_node(
                        "StreamingMaxPool",
                        [mp_input],
                        [mp_output],
                        domain="finn.custom_op.fpgadataflow",
                        backend="fpgadataflow",
                        PoolDim=(k_h, k_w),
                        NumChannels=ifm_ch,
                        ImgDim=(ifm_dim_h, ifm_dim_w),
                        dataType=dt.name,
                        PE=pe,
                        CeilMode=ceil_mode,
                        name="StreamingMaxPool_" + node.name,
                    )
                    graph.node.insert(node_ind, new_node)
                    # remove old nodes
                    graph.node.remove(node)
                    graph_modified = True
                else:
                    warnings.warn(node.name + ": could not convert to HW")
        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
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
                    inputDataType=idt.name,
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
        for node in graph.node:
            node_ind += 1
            successors = model.find_consumers(node.output[0])
            if successors is not None and len(successors) >= 2:
                output_tensor = node.output[0]
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
