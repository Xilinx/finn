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
import qonnx.core.data_layout as DataLayout
import warnings
from onnx import TensorProto
from onnx import helper as oh
from qonnx.core.datatype import DataType
from qonnx.core.onnx_exec import execute_node
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import SortGraph
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name


class MoveAddPastMul(Transformation):
    """Move add operations past multiply operations on linear segments of the graph.
    The aim is to have them next to each other such that they can be collapsed into
    a single add."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if (
                n.op_type == "Add"
                and not model.is_fork_node(n)
                and not model.is_join_node(n)
            ):
                consumer = model.find_consumer(n.output[0])
                if (
                    consumer is not None
                    and consumer.op_type == "Mul"
                    and not model.is_join_node(consumer)
                ):
                    # have: (x) -> add(,B) -> (x+B) -> mul(,A) -> (xA+BA)
                    # want: (x) -> mul(,A) -> (xA) -> add(,BA) -> (xA+BA)
                    # assume input 0 is from the previous layer, input 1 is the
                    # trained (constant) parameter
                    mul_weight_name = consumer.input[1]
                    add_weight_name = n.input[1]
                    A = model.get_initializer(mul_weight_name)
                    B = model.get_initializer(add_weight_name)
                    if (A is None) or (B is None):
                        warnings.warn(
                            "Mul or add does not have constant params, skipping"
                        )
                        continue
                    start_name = n.input[0]
                    middle_name = n.output[0]
                    end_name = consumer.output[0]
                    # compute new param value for add
                    BA = B * A

                    # make and insert new nodes
                    new_mul = oh.make_node(
                        "Mul",
                        [start_name, mul_weight_name],
                        [middle_name],
                        name=consumer.name,
                    )
                    new_add = oh.make_node(
                        "Add", [middle_name, add_weight_name], [end_name], name=n.name
                    )
                    graph.node.insert(node_ind, new_mul)
                    graph.node.insert(node_ind + 1, new_add)
                    # replace add value
                    model.set_initializer(add_weight_name, BA)
                    # remove old nodes
                    graph.node.remove(n)
                    graph.node.remove(consumer)
                    graph_modified = True

        model = model.transform(InferShapes())
        return (model, graph_modified)


class MoveScalarMulPastMatMul(Transformation):
    """Move scalar mul operations past matmul operations. We want to have muls
    next to each other such that they can be collapsed into a single mul."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if (
                n.op_type == "Mul"
                and not model.is_fork_node(n)
                and not model.is_join_node(n)
            ):
                consumer = model.find_consumer(n.output[0])
                if (
                    consumer is not None
                    and consumer.op_type == "MatMul"
                    and not model.is_join_node(consumer)
                ):
                    mul_weight_name = n.input[1]
                    matmul_weight_name = consumer.input[1]
                    A = model.get_initializer(mul_weight_name)
                    W = model.get_initializer(matmul_weight_name)
                    if (A is None) or (W is None):
                        warnings.warn("MatMul or Mul params are not constant, skipping")
                        continue
                    start_name = n.input[0]
                    middle_name = n.output[0]
                    end_name = consumer.output[0]
                    mm_out_shape = model.get_tensor_shape(end_name)
                    if all(x == 1 for x in A.shape):
                        # if the mul is scalar, we can simply swap the order of ops
                        # make and insert new nodes
                        new_matmul = oh.make_node(
                            "MatMul",
                            [start_name, matmul_weight_name],
                            [middle_name],
                            name=consumer.name,
                        )
                        new_mul = oh.make_node(
                            "Mul",
                            [middle_name, mul_weight_name],
                            [end_name],
                            name=n.name,
                        )
                        graph.node.insert(node_ind, new_matmul)
                        graph.node.insert(node_ind + 1, new_mul)
                        model.set_tensor_shape(middle_name, mm_out_shape)
                        # remove old nodes
                        graph.node.remove(n)
                        graph.node.remove(consumer)
                        graph_modified = True
        model = model.transform(InferShapes())
        return (model, graph_modified)


class MoveScalarAddPastMatMul(Transformation):
    """Move scalar add operations past matmul operations. We want to have adds
    next to each other such that they can be collapsed into a single add."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if (
                n.op_type == "Add"
                and not model.is_fork_node(n)
                and not model.is_join_node(n)
            ):
                consumer = model.find_consumer(n.output[0])
                if (
                    consumer is not None
                    and consumer.op_type == "MatMul"
                    and not model.is_join_node(consumer)
                ):
                    add_weight_name = n.input[1]
                    matmul_weight_name = consumer.input[1]
                    A = model.get_initializer(add_weight_name)
                    W = model.get_initializer(matmul_weight_name)
                    if (A is None) or (W is None):
                        warnings.warn("MatMul or Add params are not constant, skipping")
                        continue
                    start_name = n.input[0]
                    middle_name = n.output[0]
                    end_name = consumer.output[0]
                    mm_out_shape = model.get_tensor_shape(end_name)
                    if all(x == 1 for x in A.shape):
                        # if the add is scalar, we can move it past the matmul
                        # by taking it past the matmul with a dot product
                        Anew = np.dot(A * np.ones(W.shape[0], dtype=np.float32), W)
                        # update the add weight
                        model.set_initializer(add_weight_name, Anew)
                        new_matmul = oh.make_node(
                            "MatMul",
                            [start_name, matmul_weight_name],
                            [middle_name],
                            name=consumer.name,
                        )
                        new_add = oh.make_node(
                            "Add",
                            [middle_name, add_weight_name],
                            [end_name],
                            name=n.name,
                        )
                        graph.node.insert(node_ind, new_matmul)
                        graph.node.insert(node_ind + 1, new_add)
                        model.set_tensor_shape(middle_name, mm_out_shape)
                        # remove old nodes
                        graph.node.remove(n)
                        graph.node.remove(consumer)
                        graph_modified = True
        model = model.transform(InferShapes())
        return (model, graph_modified)


class MoveAddPastConv(Transformation):
    """Move scalar and channelwise add operations past conv operations. We want to have adds
    next to each other such that they can be collapsed into a single add."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if (
                n.op_type == "Add"
                and not model.is_fork_node(n)
                and not model.is_join_node(n)
            ):
                consumer = model.find_consumer(n.output[0])
                if (
                    consumer is not None
                    and consumer.op_type == "Conv"
                    and not model.is_join_node(consumer)
                ):
                    conv_node = consumer
                    add_node = n
                    add_weight_name = n.input[1]
                    conv_in_name = consumer.input[0]
                    conv_in_shape = model.get_tensor_shape(conv_in_name)
                    # assume datalayout to be NCHW
                    channels = conv_in_shape[1]
                    A = model.get_initializer(add_weight_name)
                    if A is None:
                        warnings.warn("Add param is not constant, skipping")
                        continue
                    start_name = n.input[0]
                    end_name = consumer.output[0]
                    conv_out_shape = model.get_tensor_shape(end_name)

                    using_padding = True
                    pads = list(get_by_name(consumer.attribute, "pads").ints)
                    if sum(pads) == 0:
                        using_padding = False
                    if (
                        all(x == 1 for x in A.shape) or A.shape == (1, channels, 1, 1)
                    ) and not using_padding:
                        # create a tensor filled with the add constant, in
                        # the shape expected by the convolution
                        conv_in_const = np.zeros(conv_in_shape, dtype=np.float32)
                        if A.shape == (1, channels, 1, 1):
                            for ch in range(channels):
                                conv_in_const[0][ch].fill(A[0][ch].item())
                        else:
                            conv_in_const.fill(A.item())
                        # create an execution context and put in const input
                        exec_ctx = model.make_empty_exec_context()
                        exec_ctx[conv_in_name] = conv_in_const
                        # execute the conv node only
                        execute_node(conv_node, exec_ctx, model.graph)
                        # retrieve the conv output
                        Anew = exec_ctx[end_name]

                        # strip out repetition if no padding
                        Anew = Anew[0, :, 0, 0].reshape(1, -1, 1, 1)
                        # update the add weight
                        model.set_initializer(add_weight_name, Anew)
                        # rewire add input to be conv input
                        conv_node.input[0] = start_name
                        model.set_tensor_shape(start_name, conv_in_shape)
                        # use old conv input tensor as conv output
                        conv_node.output[0] = conv_in_name
                        model.set_tensor_shape(conv_in_name, conv_out_shape)
                        # use new conv output as new add node input
                        add_node.input[0] = conv_in_name
                        # use old conv output as new add node output
                        add_node.output[0] = end_name
                        # move add node past conv node
                        graph.node.remove(add_node)
                        graph.node.insert(node_ind, add_node)
                        graph_modified = True

        model = model.transform(InferShapes())
        return (model, graph_modified)


class MoveScalarMulPastConv(Transformation):
    """Move scalar mul operations past conv operations. We want to have muls
    next to each other such that they can be collapsed into a single mul."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if (
                n.op_type == "Mul"
                and not model.is_fork_node(n)
                and not model.is_join_node(n)
            ):
                consumer = model.find_consumer(n.output[0])
                if (
                    consumer is not None
                    and consumer.op_type == "Conv"
                    and not model.is_join_node(consumer)
                ):
                    mul_weight_name = n.input[1]
                    A = model.get_initializer(mul_weight_name)
                    if A is None:
                        warnings.warn("Mul param is not constant, skipping")
                        continue
                    conv_node = consumer
                    mul_node = n
                    start_name = mul_node.input[0]
                    conv_in_name = conv_node.input[0]
                    conv_in_shape = model.get_tensor_shape(conv_in_name)
                    conv_out_name = conv_node.output[0]
                    conv_out_shape = model.get_tensor_shape(conv_out_name)
                    if all(x == 1 for x in A.shape):
                        # if the mul is scalar, we can simply swap the order of ops
                        # rewire mul input to be conv input
                        conv_node.input[0] = start_name
                        model.set_tensor_shape(start_name, conv_in_shape)
                        # use old conv input tensor as conv output
                        conv_node.output[0] = conv_in_name
                        model.set_tensor_shape(conv_in_name, conv_out_shape)
                        # use new conv output as new mul node input
                        mul_node.input[0] = conv_in_name
                        # use old conv output as new mul node output
                        mul_node.output[0] = conv_out_name
                        # move add node past conv node
                        graph.node.remove(mul_node)
                        graph.node.insert(node_ind, mul_node)
                        graph_modified = True
        model = model.transform(InferShapes())
        return (model, graph_modified)


class MoveMulPastDWConv(Transformation):
    """Move channelwise mul operations past depthwise conv operations. We want to have muls
    next to each other such that they can be collapsed into a single mul."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if (
                n.op_type == "Mul"
                and not model.is_fork_node(n)
                and not model.is_join_node(n)
            ):
                consumer = model.find_consumer(n.output[0])
                if (
                    consumer is not None
                    and consumer.op_type == "Conv"
                    and not model.is_join_node(consumer)
                ):
                    mul_weight_name = n.input[1]
                    A = model.get_initializer(mul_weight_name)
                    if A is None:
                        warnings.warn(
                            """Mul weight tensor is not set. If it is a constant,
                                please use set_initializer to set the tensor."""
                        )
                        continue
                    conv_node = consumer
                    mul_node = n
                    start_name = mul_node.input[0]
                    conv_in_name = conv_node.input[0]
                    conv_in_shape = model.get_tensor_shape(conv_in_name)
                    ifm_ch = conv_in_shape[1]
                    group_attribute = get_by_name(consumer.attribute, "group")
                    if group_attribute is None:
                        continue
                    group_attribute = group_attribute.i
                    conv_out_name = conv_node.output[0]
                    conv_out_shape = model.get_tensor_shape(conv_out_name)
                    if A.shape == (1, ifm_ch, 1, 1) and ifm_ch == group_attribute:
                        # if the mul is channelwise and conv is depthwise,
                        # we can simply swap the order of ops
                        # rewire mul input to be conv input
                        conv_node.input[0] = start_name
                        model.set_tensor_shape(start_name, conv_in_shape)
                        model.set_tensor_datatype(start_name, DataType["FLOAT32"])
                        # use old conv input tensor as conv output
                        conv_node.output[0] = conv_in_name
                        model.set_tensor_shape(conv_in_name, conv_out_shape)
                        model.set_tensor_datatype(conv_in_name, DataType["FLOAT32"])
                        # use new conv output as new mul node input
                        mul_node.input[0] = conv_in_name
                        # use old conv output as new mul node output
                        mul_node.output[0] = conv_out_name
                        model.set_tensor_datatype(conv_out_name, DataType["FLOAT32"])
                        # move mul node past conv node
                        graph.node.remove(mul_node)
                        graph.node.insert(node_ind, mul_node)
                        graph_modified = True
        model = model.transform(InferShapes())
        return (model, graph_modified)


class MoveMulPastMaxPool(Transformation):
    """Move non-negative scalar or channelwise mul operations past max pool operations.
    We want to have muls next to each other such that they can be collapsed into a
    single mul."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if (
                n.op_type == "Mul"
                and not model.is_fork_node(n)
                and not model.is_join_node(n)
            ):
                consumer = model.find_consumer(n.output[0])
                if (
                    consumer is not None
                    and consumer.op_type == "MaxPool"
                    and not model.is_join_node(consumer)
                ):
                    mul_weight_name = n.input[1]
                    A = model.get_initializer(mul_weight_name)
                    if A is None:
                        warnings.warn(
                            """Mul weight tensor is not set. If it is a constant,
                                please use set_initializer to set the tensor."""
                        )
                        continue
                    maxpool_node = consumer
                    mul_node = n
                    start_name = mul_node.input[0]
                    maxpool_in_name = maxpool_node.input[0]
                    maxpool_in_shape = model.get_tensor_shape(maxpool_in_name)
                    ifm_ch = maxpool_in_shape[1]
                    maxpool_out_name = maxpool_node.output[0]
                    maxpool_out_shape = model.get_tensor_shape(maxpool_out_name)

                    # do not support non-2D MaxPool
                    kernel_shape = list(
                        get_by_name(maxpool_node.attribute, "kernel_shape").ints
                    )
                    if len(kernel_shape) != 2:
                        continue

                    # do not move negative multiplication factor(s)
                    if (A < 0).any():
                        continue

                    if all(x == 1 for x in A.shape) or A.shape == (1, ifm_ch, 1, 1):
                        # if the mul is scalar or channelwise,
                        # we can simply swap the order of ops
                        # rewire mul input to be maxpool input
                        maxpool_node.input[0] = start_name
                        model.set_tensor_shape(start_name, maxpool_in_shape)
                        model.set_tensor_datatype(start_name, DataType["FLOAT32"])
                        # use old maxpool input tensor as maxpool output
                        maxpool_node.output[0] = maxpool_in_name
                        model.set_tensor_shape(maxpool_in_name, maxpool_out_shape)
                        model.set_tensor_datatype(maxpool_in_name, DataType["FLOAT32"])
                        # use new maxpool output as new mul node input
                        mul_node.input[0] = maxpool_in_name
                        # use old maxpool output as new mul node output
                        mul_node.output[0] = maxpool_out_name
                        model.set_tensor_datatype(maxpool_out_name, DataType["FLOAT32"])
                        # move mul node past maxpool node
                        graph.node.remove(mul_node)
                        graph.node.insert(node_ind, mul_node)
                        graph_modified = True
        model = model.transform(InferShapes())
        return (model, graph_modified)


class MoveLinearPastEltwiseAdd(Transformation):
    """Move linear operations (mul, add) past elementwise add operations where possible.
    Specifically,matches and transforms the following patterns:
    (x*C) + (y*C) -> (x + y) * C
    (x+A) + (y+B) -> (x + y) + (A + B)
    where x and y are dynamic inputs, A, B, C are constant tensors (in general).
    """

    def move_node(self, graph, n, prod0, prod1, node_ind):
        # found! move one of the muls to output, remove the other one
        lin0_in0 = prod0.input[0]
        lin1_in0 = prod1.input[0]
        in0 = n.input[0]
        out = n.output[0]
        # TODO: check shapes don't change through scalar mul or add
        # connect the eltwise add inputs to mul inputs
        n.input[0] = lin0_in0
        n.input[1] = lin1_in0
        # connect mul0 output to eltwise add output
        prod0.output[0] = out
        # connect the input of mul0 and output of eltwise add together
        n.output[0] = in0
        prod0.input[0] = in0
        # move prod0 node past eltwise add node, and remove prod1
        graph.node.remove(prod1)
        graph.node.remove(prod0)
        graph.node.insert(node_ind - 2, prod0)

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        nodes = [n for n in graph.node]
        for n in nodes:
            node_ind += 1
            if n.op_type == "Add":
                # check for tensors on both inputs (eltwise add)
                # scalar add has an initializer on one input
                in0 = n.input[0]
                in1 = n.input[1]
                if in0 is None or in1 is None:
                    continue
                A = model.get_initializer(in0)
                B = model.get_initializer(in1)
                if A is not None or B is not None:
                    continue
                # check for mul with same initializer on both inputs
                prod0 = model.find_producer(in0)
                prod1 = model.find_producer(in1)
                # Also check case when both branches are empty and come
                # from the same node: (prod0 == prod1)
                # Other transform should handle that
                if prod0 is None or prod1 is None or (prod0 == prod1):
                    continue
                if len(prod0.input) < 2 or len(prod1.input) < 2:
                    continue
                init0 = model.get_initializer(prod0.input[1])
                init1 = model.get_initializer(prod1.input[1])
                # if either initializer is None, skip
                if init0 is None or init1 is None:
                    continue
                if prod0.op_type == "Mul" and prod1.op_type == "Mul":
                    if np.array_equal(init0, init1):
                        self.move_node(graph, n, prod0, prod1, node_ind)
                        node_ind -= 1
                        graph_modified = True
                elif prod0.op_type == "Add" and prod1.op_type == "Add":
                    init = init0 + init1
                    # update initializer of prod0, which we'll move
                    model.set_initializer(prod0.input[1], init)
                    self.move_node(graph, n, prod0, prod1, node_ind)
                    node_ind -= 1
                    graph_modified = True
                else:
                    continue
        model = model.transform(InferShapes())
        return (model, graph_modified)


class MoveScalarLinearPastInvariants(Transformation):
    """Move scalar linear operations (mul, add) past functions which are invariant
    to them. Specifically, matches and transforms the following patterns:
    f(x*C) -> f(x) * C
    f(x+C) -> f(x) + C
    where x is a dynamic input, C is a constant tensor.
    Known f which obey this property are: Reshape, Flatten, Transpose,
    GlobalAveragePool
    """

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        nodes = [n for n in graph.node]
        for n in nodes:
            node_ind += 1
            is_nearest_neighbor_resample = False
            if n.op_type == "Upsample" or n.op_type == "Resize":
                # Extract mode and scales and input shape
                mode = get_by_name(n.attribute, "mode").s.decode("ascii")
                is_nearest_neighbor_resample = mode == "nearest"
            if (
                n.op_type == "GlobalAveragePool"
                or n.op_type == "Reshape"
                or n.op_type == "Transpose"
                or n.op_type == "Flatten"
                or is_nearest_neighbor_resample
            ):
                in0 = n.input[0]
                if in0 is None:
                    continue
                # find and check producer on our input
                prod0 = model.find_producer(in0)
                if prod0 is None:
                    continue

                if prod0.op_type in ["Mul", "Add", "Div"]:
                    # check if second input of producer is an initializer
                    init0 = model.get_initializer(prod0.input[1])
                    # if either initializer is None, skip
                    if init0 is None:
                        continue
                    # if initializer is not scalar, skip
                    if np.prod(init0.shape) != 1:
                        continue
                    # Flatten input if required
                    if len(init0.shape) > 0:
                        init0 = init0.flatten()[0]
                        model.set_initializer(prod0.input[1], init0)
                    # move prod0 from input to output,
                    old_prod0_in = prod0.input[0]
                    old_prod0_out = prod0.output[0]
                    scalar_op_odt = model.get_tensor_datatype(old_prod0_out)
                    old_n_out = n.output[0]
                    in_shape = model.get_tensor_shape(n.input[0])
                    out_shape = model.get_tensor_shape(n.output[0])
                    n.input[0] = old_prod0_in
                    n.output[0] = old_prod0_out
                    prod0.input[0] = old_prod0_out
                    prod0.output[0] = old_n_out
                    model.set_tensor_shape(n.input[0], in_shape)
                    model.set_tensor_shape(n.output[0], out_shape)
                    model.set_tensor_shape(prod0.output[0], out_shape)
                    model.set_tensor_datatype(prod0.output[0], scalar_op_odt)
                    model.set_tensor_datatype(n.output[0], DataType["FLOAT32"])
                    graph.node.remove(prod0)
                    graph.node.insert(node_ind - 1, prod0)
                    graph_modified = True
                else:
                    continue
        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class MakeMaxPoolNHWC(Transformation):
    """Convert (MaxPool, NHWCTranspose) into (NHWCTranspose, MaxPoolNHWC)
    and (NCHWTranspose, MaxPool) into (MaxPoolNHWC, NCHWTranspose)."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "MaxPool":
                consumer = model.find_consumer(n.output[0])
                producer = model.find_producer(n.input[0])
                if consumer is not None and consumer.op_type == "Transpose":
                    perms = list(get_by_name(consumer.attribute, "perm").ints)
                    if perms == [0, 2, 3, 1]:
                        ceil_mode = get_by_name(n.attribute, "ceil_mode")
                        if ceil_mode is not None:
                            ceil_mode = ceil_mode.i
                        else:
                            ceil_mode = (
                                0  # default to ceil_mode=0 (equivalent to np.floor)
                            )
                        n.op_type = "MaxPoolNHWC"
                        n.domain = "qonnx.custom_op.general"
                        start_name = n.input[0]
                        mid_name = consumer.input[0]
                        end_name = consumer.output[0]
                        (b, c, hi, wi) = model.get_tensor_shape(start_name)
                        (b, c, ho, wo) = model.get_tensor_shape(mid_name)
                        consumer.input[0] = start_name
                        consumer.output[0] = mid_name
                        n.input[0] = mid_name
                        n.output[0] = end_name
                        model.set_tensor_shape(mid_name, (b, hi, wi, c))
                        model.set_tensor_shape(end_name, (b, ho, wo, c))
                        getCustomOp(n).set_nodeattr("ceil_mode", ceil_mode)
                        graph.node.remove(consumer)
                        graph.node.insert(node_ind - 1, consumer)
                        graph_modified = True
                elif producer is not None and producer.op_type == "Transpose":
                    perms = list(get_by_name(producer.attribute, "perm").ints)
                    if perms == [0, 3, 1, 2]:
                        ceil_mode = get_by_name(n.attribute, "ceil_mode")
                        if ceil_mode is not None:
                            ceil_mode = ceil_mode.i
                        else:
                            ceil_mode = (
                                0  # default to ceil_mode=0 (equivalent to np.floor)
                            )
                        n.op_type = "MaxPoolNHWC"
                        n.domain = "qonnx.custom_op.general"
                        start_name = producer.input[0]
                        mid_name = n.input[0]
                        end_name = n.output[0]
                        (b, hi, wi, c) = model.get_tensor_shape(start_name)
                        (b, c, ho, wo) = model.get_tensor_shape(end_name)
                        producer.input[0] = mid_name
                        producer.output[0] = end_name
                        n.input[0] = start_name
                        n.output[0] = mid_name
                        model.set_tensor_shape(mid_name, (b, ho, wo, c))
                        model.set_tensor_shape(end_name, (b, c, ho, wo))
                        getCustomOp(n).set_nodeattr("ceil_mode", ceil_mode)
                        graph.node.remove(producer)
                        graph.node.insert(node_ind, producer)
                        graph_modified = True
        return (model, graph_modified)


class MakeScaleResizeNHWC(Transformation):
    """
    Converts the inputs and outputs for all scales Resize and Upsample nodes
    from NCHW to NHWC.
    """

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Upsample" or n.op_type == "Resize":
                if model.get_tensor_layout(n.input[0]) != DataLayout.NCHW:
                    warnings.warn(
                        "%s: Input not NCHW. Can't operate transformation on node."
                        % n.name
                    )
                    continue
                consumer = model.find_consumer(n.output[0])
                producer = model.find_producer(n.input[0])
                if n.op_type == "Upsample":
                    scales_ind = 1
                else:
                    scales_ind = 2
                if producer is not None and producer.op_type == "Transpose":
                    perms = list(get_by_name(producer.attribute, "perm").ints)
                    if perms == [0, 3, 1, 2]:
                        old_value = model.get_initializer(n.input[scales_ind])
                        new_value = np.array(
                            [old_value[idx] for idx in (0, 2, 3, 1)],
                            dtype=np.dtype("float32"),
                        )
                        model.set_initializer(n.input[scales_ind], new_value)
                        start_name = producer.input[0]
                        mid_name = n.input[0]
                        end_name = n.output[0]
                        (b, hi, wi, c) = model.get_tensor_shape(start_name)
                        (b, c, ho, wo) = model.get_tensor_shape(end_name)
                        producer.input[0] = mid_name
                        producer.output[0] = end_name
                        n.input[0] = start_name
                        n.output[0] = mid_name
                        model.set_tensor_shape(mid_name, (b, ho, wo, c))
                        model.set_tensor_shape(end_name, (b, c, ho, wo))
                        graph.node.remove(producer)
                        graph.node.insert(node_ind, producer)
                elif consumer is not None and consumer.op_type == "Transpose":
                    perms = list(get_by_name(consumer.attribute, "perm").ints)
                    if perms == [0, 2, 3, 1]:
                        old_value = model.get_initializer(n.input[scales_ind])
                        new_value = np.array(
                            [old_value[idx] for idx in (0, 2, 3, 1)],
                            dtype=np.dtype("float32"),
                        )
                        model.set_initializer(n.input[scales_ind], new_value)
                        start_name = n.input[0]
                        mid_name = consumer.input[0]
                        end_name = consumer.output[0]
                        (b, c, hi, wi) = model.get_tensor_shape(start_name)
                        (b, c, ho, wo) = model.get_tensor_shape(mid_name)
                        consumer.input[0] = start_name
                        consumer.output[0] = mid_name
                        n.input[0] = mid_name
                        n.output[0] = end_name
                        model.set_tensor_shape(mid_name, (b, hi, wi, c))
                        model.set_tensor_shape(end_name, (b, ho, wo, c))
                        graph.node.remove(consumer)
                        graph.node.insert(node_ind - 1, consumer)
        return (model, False)


class MoveOpPastFork(Transformation):
    """Move node operations past graph forks. Used when a node before a fork
    can be merged with nodes in the branches
    """

    def __init__(self, op_name_list, get_attrs_fxn=lambda x: {}):
        super().__init__()
        self.ops_to_move = op_name_list
        self.get_attrs_fxn = get_attrs_fxn

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        nodes = [n for n in graph.node]
        node_ind = 0
        for n in nodes:
            node_ind += 1
            if (
                n.op_type in self.ops_to_move
                and model.is_fork_node(n)
                and not model.is_join_node(n)
            ):

                # Restrict this transform to operations with constant parameters
                # Assuming parameters is in input 1
                if len(n.input) > 1:
                    op_init_param = model.get_initializer(n.input[1])
                else:
                    op_init_param = None

                # Check case when branches are empty and go
                # to the same node
                consumers = model.find_consumers(n.output[0])
                assert len(consumers) > 1, "Must have >1 consumer"
                unique_consumer = True
                for consum_node in consumers[1:]:
                    if consumers[0] != consum_node:
                        unique_consumer = False
                        break

                if unique_consumer:
                    continue

                for consumer_node in consumers[1:]:
                    # create new node
                    new_output_tensor_name = model.make_new_valueinfo_name()
                    if op_init_param is None:
                        new_inp_list = [n.input[0]]
                    else:
                        new_param_name = model.make_new_valueinfo_name()
                        new_inp_list = [n.input[0], new_param_name]
                        model.set_initializer(new_param_name, op_init_param)
                    attrs = self.get_attrs_fxn(n)
                    # TODO use copy of original node instead to get attrs?
                    new_node = oh.make_node(
                        n.op_type, new_inp_list, [new_output_tensor_name], **attrs
                    )
                    graph.node.insert(node_ind, new_node)
                    node_ind += 1

                    # change consumer input tensor
                    graph.node.remove(consumer_node)
                    for idx, consumer_input in enumerate(consumer_node.input):
                        if consumer_input == n.output[0]:
                            consumer_node.input[idx] = new_output_tensor_name
                            break
                    else:
                        raise Exception(
                            "Consumer should have the current node output as input"
                        )

                    graph.node.insert(node_ind, consumer_node)

                graph_modified = True

        model = model.transform(InferShapes())
        return (model, graph_modified)


class MoveAddPastFork(MoveOpPastFork):
    def __init__(self):
        super().__init__(["Add"])


class MoveMulPastFork(MoveOpPastFork):
    def __init__(self):
        super().__init__(["Mul"])


class MoveLinearPastFork(MoveOpPastFork):
    def __init__(self):
        super().__init__(["Add", "Mul"])


class MoveTransposePastFork(MoveOpPastFork):
    def __init__(self):
        super().__init__(
            ["Transpose"], lambda x: {"perm": get_by_name(x.attribute, "perm").ints}
        )


class MoveMaxPoolPastMultiThreshold(Transformation):
    """Move MaxPool nodes past MultiThreshold nodes on linear segments of the graph."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        nodes = [n for n in graph.node]
        for n in nodes:
            node_ind += 1
            if n.op_type == "MaxPool" and not model.is_fork_node(n):
                consumer = model.find_consumer(n.output[0])
                pads = get_by_name(n.attribute, "pads")
                has_padding = False
                if pads is not None:
                    pads = list(pads.ints)
                    has_padding = np.prod(pads) != 0
                if consumer is not None and consumer.op_type == "MultiThreshold":
                    mt_out = consumer.output[0]
                    mt_odt = model.get_tensor_datatype(mt_out)
                    if mt_odt.signed() and has_padding:
                        warnings.warn(
                            "Skipping padded MaxPool + signed-output MultiThreshold"
                        )
                        continue
                    # check for non-decreasing thresholds and nonnegative
                    # scale factor in MultiThreshold
                    # otherwise we cannot do the reordering
                    T = model.get_initializer(consumer.input[1])
                    T_sorted = np.sort(T, axis=1)
                    assert (
                        T == T_sorted
                    ).all(), "MultiThreshold must have non-decreasing thresholds"
                    mt_inst = getCustomOp(consumer)
                    if mt_inst.get_nodeattr("out_scale") < 0:
                        warnings.warn("Skipping MultiThreshold with negative out_scale")
                        continue

                    # remove old nodes
                    graph.node.remove(n)
                    graph.node.remove(consumer)

                    # swap conections
                    group_in = n.input[0]
                    # new tensor because dims change
                    group_middle = model.make_new_valueinfo_name()
                    group_out = consumer.output[0]

                    consumer.input[0] = group_in
                    consumer.output[0] = group_middle

                    n.input[0] = group_middle
                    n.output[0] = group_out

                    # insert them back in
                    graph.node.insert(node_ind - 1, consumer)
                    graph.node.insert(node_ind, n)

                    graph_modified = True

        model = model.transform(InferShapes())
        return (model, graph_modified)


class MoveFlattenPastTopK(Transformation):
    """Move flatten node past a succeeding topk node, if the "axis" attribute in topk
    is set to -1 and the data layout before the flatten is NHWC with H=W=1"""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Flatten":
                consumer = model.find_consumer(n.output[0])
                if consumer is not None and consumer.op_type == "TopK":
                    axis = get_by_name(consumer.attribute, "axis")
                    if axis is None or axis.i != -1:
                        continue
                    start_name = n.input[0]
                    data_layout = model.get_tensor_layout(start_name)
                    if data_layout != DataLayout.NHWC:
                        warnings.warn(
                            """Transformation can't be applied. The input
                            to flatten has to have DataLayout.NHWC"""
                        )
                        continue
                    (b, h, w, c) = model.get_tensor_shape(start_name)
                    if h != 1 or w != 1:
                        continue
                    # get parameter k from topk
                    k = model.get_tensor_shape(consumer.output[1])[-1]

                    # swap conections
                    # new tensor because dims change
                    middle_name = model.make_new_valueinfo_name()
                    topk_indices = oh.make_tensor_value_info(
                        middle_name, TensorProto.INT64, [b, h, w, k]
                    )
                    end_name = consumer.output[1]
                    graph.value_info.append(topk_indices)

                    # remove old nodes
                    graph.node.remove(n)
                    graph.node.remove(consumer)

                    # set inputs and outputs correctly
                    consumer.input[0] = start_name
                    consumer.output[1] = middle_name
                    model.set_tensor_shape(consumer.output[0], (b, h, w, k))

                    n.input[0] = middle_name
                    n.output[0] = end_name

                    # insert them back in
                    graph.node.insert(node_ind - 1, consumer)
                    graph.node.insert(node_ind, n)

                    graph_modified = True

        model = model.transform(InferShapes())
        return (model, graph_modified)


class MoveFlattenPastAffine(Transformation):
    """Moves a node that implements a (1, -1) reshape past a MatMul, Mul or Add node."""

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        node_ind = 0
        for n in graph.node:
            node_ind += 1
            if (
                n.op_type == "Flatten"
                and not model.is_fork_node(n)
                and not model.is_join_node(n)
            ):
                consumer = model.find_consumer(n.output[0])
                if (
                    consumer is not None
                    and (
                        consumer.op_type == "MatMul"
                        or consumer.op_type == "Mul"
                        or consumer.op_type == "Add"
                    )
                    and not model.is_join_node(consumer)
                ):
                    # move flatten past operation and rewire tensors
                    start_name = n.input[0]
                    # check if datalyout is set to NHWC and H=W=1
                    datalayout = model.get_tensor_layout(start_name)
                    if datalayout == DataLayout.NHWC:
                        (b, h, w, c) = model.get_tensor_shape(start_name)
                        if h != 1 or w != 1:
                            warnings.warn(
                                """The Transformation can only be performed if
                            H=W=1."""
                            )
                            continue
                    else:
                        warnings.warn(
                            """The Transformation can only be performed on
                            operations that operate on data layout NHWC."""
                        )
                        continue
                    middle_name = n.output[0]
                    end_name = consumer.output[0]
                    op_param_name = consumer.input[1]
                    A = model.get_initializer(op_param_name)
                    if A is None:
                        warnings.warn("Param is not constant, skipping")
                        continue
                    op_in_dt = model.get_tensor_datatype(consumer.input[0])
                    op_out_dt = model.get_tensor_datatype(consumer.output[0])
                    start_shape = model.get_tensor_shape(start_name)
                    dummy_in = np.random.uniform(low=0, high=1, size=(start_shape))

                    if consumer.op_type == "MatMul":
                        dummy_out = np.matmul(dummy_in, A)
                    elif consumer.op_type == "Mul":
                        dummy_out = dummy_in * A
                    elif consumer.op_type == "Add":
                        dummy_out = dummy_in + A

                    new_op = oh.make_node(
                        consumer.op_type,
                        [start_name, op_param_name],
                        [middle_name],
                        name=consumer.name,
                    )
                    new_flatten = oh.make_node("Flatten", [middle_name], [end_name])
                    graph.node.insert(node_ind, new_op)
                    graph.node.insert(node_ind + 1, new_flatten)
                    model.set_tensor_shape(middle_name, dummy_out.shape)
                    # because a flatten node doesn't change the datatype we need
                    # only the datatype of the op node
                    model.set_tensor_datatype(start_name, op_in_dt)
                    model.set_tensor_datatype(middle_name, op_out_dt)
                    model.set_tensor_datatype(end_name, op_out_dt)
                    # set datalayout
                    model.set_tensor_layout(start_name, DataLayout.NHWC)
                    model.set_tensor_layout(middle_name, DataLayout.NHWC)
                    # remove old nodes
                    graph.node.remove(n)
                    graph.node.remove(consumer)
                    graph_modified = True

        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        model = model.transform(InferDataLayouts())
        return (model, graph_modified)


class MoveTransposePastScalarMul(Transformation):
    """Moves a Transpose node past a scalar Mul node"""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if (
                n.op_type == "Transpose"
                and not model.is_fork_node(n)
                and not model.is_join_node(n)
            ):
                consumer = model.find_consumer(n.output[0])
                if (
                    consumer is not None
                    and consumer.op_type == "Mul"
                    and not model.is_join_node(consumer)
                ):
                    mul_weight_name = consumer.input[1]
                    A = model.get_initializer(mul_weight_name)
                    if A is None:
                        warnings.warn("Mul param is not constant, skipping")
                        continue
                    transp_node = n
                    mul_node = consumer
                    start_name = transp_node.input[0]
                    middle_name = transp_node.output[0]
                    end_name = mul_node.output[0]
                    transp_in_shape = model.get_tensor_shape(start_name)
                    transp_out_shape = model.get_tensor_shape(middle_name)
                    transp_in_layout = model.get_tensor_layout(start_name)
                    transp_out_layout = model.get_tensor_layout(middle_name)
                    if transp_in_layout is None or transp_out_layout is None:
                        warnings.warn(
                            """Datalayout is not set for tensors.
                            Transformation can't be applied."""
                        )
                        continue
                    if all(x == 1 for x in A.shape):
                        # if the mul is scalar, we can simply swap the order of ops
                        # rewire transpose input to be mul input
                        mul_node.input[0] = start_name
                        model.set_tensor_shape(start_name, transp_in_shape)
                        model.set_tensor_layout(start_name, transp_in_layout)
                        mul_node.output[0] = middle_name
                        model.set_tensor_shape(middle_name, transp_in_shape)
                        model.set_tensor_layout(middle_name, transp_in_layout)
                        transp_node.input[0] = middle_name
                        transp_node.output[0] = end_name
                        model.set_tensor_shape(end_name, transp_out_shape)
                        model.set_tensor_layout(end_name, transp_out_layout)
                        graph.node.remove(transp_node)
                        graph.node.insert(node_ind, transp_node)
                        graph_modified = True

        if graph_modified is True:
            model = model.transform(InferDataLayouts())
            model = model.transform(InferShapes())
        return (model, graph_modified)


class MoveIdenticalOpPastJoinOp(Transformation):
    """
    Move identical operations on different branches past the common join node.
    This transformation assumes that the identical operations only change the
    data layout. For linear operations, see the transformation MoveLinearPastEltwiseAdd.
    Specifically, this transformation matches and transforms the following patterns:
    f(x) + f(y) -> f(x + y)
    where f(.) is currently only supporting 'Transpose', and an 'Add' node is
    the join node.
    """

    def __init__(self, identical_op_list, join_node_list):
        super().__init__()
        self.ops_to_move = identical_op_list
        self.join_node_op = join_node_list

    def move_node(self, model, n, prod0, prod1):
        # Found! move one of the identical_ops to output, remove the other one
        identical_op0_in0 = prod0.input[0]
        identical_op1_in0 = prod1.input[0]
        add_in0 = n.input[0]
        add_out = n.output[0]

        # Rewire
        n.input[0] = identical_op0_in0
        n.input[1] = identical_op1_in0

        # Output tensor of the join node must have the same shape as
        # its input tensor (original shape is preserved)
        new_shape = model.get_tensor_shape(identical_op0_in0)

        # Set new tensor shape
        model.set_tensor_shape(tensor_name=add_in0, tensor_shape=new_shape)

        n.output[0] = add_in0
        prod0.input[0] = add_in0
        prod0.output[0] = add_out

        model.graph.node.remove(prod1)

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        for n in graph.node:
            if n.op_type in self.join_node_op and model.is_join_node(n):
                in0 = n.input[0]
                in1 = n.input[1]
                if in0 is None or in1 is None:
                    continue

                prod0 = model.find_producer(in0)
                prod1 = model.find_producer(in1)
                # Checks if the join node is preceded by
                # two different, but identical operations
                if prod0 == prod1:
                    continue

                identical_op = prod0.op_type == prod1.op_type

                if identical_op and prod0.op_type in self.ops_to_move:
                    self.move_node(model, n, prod0, prod1)
                    graph_modified = True

        if graph_modified:
            model = model.transform(SortGraph(), make_deepcopy=False, cleanup=False)

        return (model, graph_modified)


class MoveTransposePastJoinAdd(MoveIdenticalOpPastJoinOp):
    def __init__(self):
        super().__init__(["Transpose"], ["Add"])
