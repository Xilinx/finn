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
from copy import deepcopy
from onnx import TensorProto
from onnx import helper as oh
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_node
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import SortGraph
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name

# Groups node inputs by dynamic vs. initializer category
from finn.transformation.util import group_inputs_by_category


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
            if n.op_type == "Add" and not model.is_fork_node(n) and not model.is_join_node(n):
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
                        warnings.warn("Mul or add does not have constant params, skipping")
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
                    # Delete the datatype annotation of the parameter tensor
                    # TODO: Maybe we should derive the new type properly...
                    model.set_tensor_datatype(add_weight_name, None)
                    # Delete the shape annotation of the connecting tensors
                    # to be re-done later. This prevents shapes from propagating
                    # backwards.
                    # Note: Do not delete annotation for the input tensor, as
                    # this prevents future shape inference.
                    model.set_tensor_shape(middle_name, None)
                    model.set_tensor_shape(end_name, None)
                    # remove old nodes
                    graph.node.remove(n)
                    graph.node.remove(consumer)
                    graph_modified = True
        # Note: Running shape inference is necessary as shape
        # annotations have been deleted above
        model = model.transform(InferShapes())
        # Note. Running datatype inference is necessary as datatype
        # annotations have been deleted above
        model = model.transform(InferDataTypes())
        return model, graph_modified


# Tests whether a tensor is a scalar, i.e., whether all dimensions are 1
def is_scalar(tensor):
    return tensor is not None and all(x == 1 for x in tensor.shape)


# Tests whether a node is a scalar multiplication with a constant scale factor
def is_const_scalar_mul(node, model):
    # Only handle existing Mul type nodes
    if node is not None and node.op_type == "Mul":
        # The constant must be an initializer
        #   Note: Assumes the constant parameter to always be the second input
        scale = model.get_initializer(node.input[1])
        # Test for existence of a constant scale factor
        return scale is not None and is_scalar(scale)
    # Did not match the operator type
    return False


# Refactored version of the MoveScalarMulPastMatMul transform capable of
# transforming two-input MatMul, like those being part of the attention operator
class MoveScalarMulPastMatMul(Transformation):
    """Move scalar mul operations past matmul operations. We want to have muls
    next to each other such that they can be collapsed into a single mul."""

    # Applies the transform to a whole model graph
    def apply(self, model):
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False

        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # First pattern matching condition: For the transform to be
            # applicable, the node has to be a MatMul operator
            if node.op_type == "MatMul":
                # Note: When touching the following code, remember to treat both
                # branches equivalently!
                # TODO: Can this be enforced or at least be made easier by
                #  extracting common code patterns to a function?

                # Get the left hand side and right hand side inputs
                #   Note: Assumes the ordering of left to right inputs to match
                #   indices 0 to 1. However, it does not "hurt" if it is
                #   reversed as both sides are treated equivalently.
                lhs = model.find_producer(node.input[0])
                rhs = model.find_producer(node.input[1])

                # Give precedence to the left hand side input testing for the
                # presence of a scalar multiplication
                if is_const_scalar_mul(lhs, model):
                    # Cannot handle fork nodes: We would have to distribute the
                    # Mul into all branches
                    # TODO: Maybe reconsider this at some point, there is
                    #  probably nothing preventing this in general, it is just
                    #  more difficult and apparently not necessary right now.
                    if model.is_fork_node(lhs):
                        # Softly skip this node
                        continue
                    # Unpack the connection pattern of a scalar mul feeding the
                    # lhs input of the matmul
                    # Names of the three input tensors to the mul-matmul complex
                    a, b, c = lhs.input[0], lhs.input[1], node.input[1]
                    # Names of the intermediate and the global output
                    m, o = lhs.output[0], node.output[0]  # noqa: Duplicate code
                    # Rewire the operator connections locally, swapping mul and
                    # matmul operator order
                    matmul = oh.make_node("MatMul", [a, c], [m], node.name)
                    mul = oh.make_node("Mul", [m, b], [o], lhs.name)
                    # Insert the rewired nodes into the graph
                    graph.node.insert(index, matmul)
                    graph.node.insert(index + 1, mul)
                    # Adapt the shape of the intermediate tensor as it changed
                    # according to the output shape of the matmul
                    model.set_tensor_shape(m, model.get_tensor_shape(o))
                    # Remove the old nodes from the graph
                    graph.node.remove(lhs)
                    graph.node.remove(node)
                    # The graph has been modified, this needs to be reported
                    # back to the caller
                    graph_modified = True
                    # Cannot further modify the node (i.e., the rhs) as the
                    # index and state of the nodes changed and need to be
                    # queried again from the graph.node at the start of the next
                    # iteration.
                    continue

                # Next try whether the right hand side matches the pattern of a
                # scalar multiplication
                if is_const_scalar_mul(rhs, model):
                    # Cannot handle fork nodes: We would have to distribute the
                    # Mul into all branches
                    # TODO: Maybe reconsider this at some point, there is
                    #  probably nothing preventing this in general, it is just
                    #  more difficult and apparently not necessary right now.
                    if model.is_fork_node(rhs):
                        # Softly skip this node
                        continue
                    # Unpack the connection pattern of a scalar mul feeding the
                    # rhs input of the matmul
                    # Names of the three input tensors to the mul-matmul complex
                    a, b, c = node.input[0], rhs.input[0], rhs.input[1]
                    # Names of the intermediate and the global output
                    m, o = rhs.output[0], node.output[0]  # noqa: Duplicate code
                    # Rewire the operator connections locally, swapping mul and
                    # matmul operator order
                    matmul = oh.make_node("MatMul", [a, b], [m], node.name)
                    mul = oh.make_node("Mul", [m, c], [o], rhs.name)
                    # Insert the rewired nodes into the graph
                    graph.node.insert(index, matmul)
                    graph.node.insert(index + 1, mul)
                    # Adapt the shape of the intermediate tensor as it changed
                    # according to the output shape of the matmul
                    model.set_tensor_shape(m, model.get_tensor_shape(o))
                    # Remove the old nodes from the graph
                    graph.node.remove(rhs)
                    graph.node.remove(node)
                    # The graph has been modified, this needs to be reported
                    # back to the caller
                    graph_modified = True

        # Finalize the transformation by inferring shapes again (as these might
        # have changed)
        model = model.transform(InferShapes())
        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified


class MoveScalarAddPastMatMul(Transformation):
    """Move scalar add operations past matmul operations. We want to have adds
    next to each other such that they can be collapsed into a single add."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Add" and not model.is_fork_node(n) and not model.is_join_node(n):
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
            if n.op_type == "Add" and not model.is_fork_node(n) and not model.is_join_node(n):
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
            if n.op_type == "Mul" and not model.is_fork_node(n) and not model.is_join_node(n):
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


class MoveScalarMulPastConvTranspose(Transformation):
    """Move scalar mul operations past ConvTranspose operations. We want to have muls
    next to each other such that they can be collapsed into a single mul."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Mul" and not model.is_fork_node(n) and not model.is_join_node(n):
                consumer = model.find_consumer(n.output[0])
                if (
                    consumer is not None
                    and consumer.op_type == "ConvTranspose"
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
            if n.op_type == "Mul" and not model.is_fork_node(n) and not model.is_join_node(n):
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
            if n.op_type == "Mul" and not model.is_fork_node(n) and not model.is_join_node(n):
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
                    kernel_shape = list(get_by_name(maxpool_node.attribute, "kernel_shape").ints)
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
                        # Delete shape annotations of connecting tensors to be
                        # re-done later. This prevents wrong shape propagation,
                        # for example in cases where the Add broadcasts shapes.
                        model.set_tensor_shape(n.output[0], None)
                        model.set_tensor_shape(prod0.output[0], None)
                        node_ind -= 1
                        graph_modified = True
                elif prod0.op_type == "Add" and prod1.op_type == "Add":
                    init = init0 + init1
                    # update initializer of prod0, which we'll move
                    model.set_initializer(prod0.input[1], init)
                    self.move_node(graph, n, prod0, prod1, node_ind)
                    # Delete shape annotations of connecting tensors to be
                    # re-done later. This prevents wrong shape propagation,
                    # for example in cases where the Add broadcasts shapes.
                    model.set_tensor_shape(n.output[0], None)
                    model.set_tensor_shape(prod0.output[0], None)
                    node_ind -= 1
                    graph_modified = True
                else:
                    continue

        # Note: Running shape inference is necessary as shape annotations have
        # been deleted above
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        return model, graph_modified


class MoveScalarLinearPastInvariants(Transformation):
    """Move scalar linear operations (mul, add) past functions which are invariant
    to them. Specifically, matches and transforms the following patterns:
    f(x*C) -> f(x) * C
    f(x+C) -> f(x) + C
    where x is a dynamic input, C is a constant tensor.
    Known f which obey this property are: Reshape, Flatten, Transpose,
    GlobalAveragePool
    """

    # Op-types of currently supported invariants
    # Op-types of currently supported invariants
    SUPPORTED_INVARIANTS = {
        "GlobalAveragePool",
        "Identity",
        "Reshape",
        "Transpose",
        "Flatten",
        "Expand",
        "Slice",
        "Squeeze",
        "Unsqueeze",
    }

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
            if n.op_type in self.SUPPORTED_INVARIANTS or is_nearest_neighbor_resample:
                in0 = n.input[0]
                if in0 is None:
                    continue
                # find and check producer on our input
                prod0 = model.find_producer(in0)
                if prod0 is None:
                    continue

                if prod0.op_type in ["Mul", "Div", "Add", "Sub"]:
                    # Cannot handle fork-nodes, try MoveLinearPastFork first
                    if model.is_fork_node(prod0):
                        warnings.warn(
                            f"{self.__class__.__name__}:"
                            f" Skipping near match: {prod0.name} is a fork-node,"
                            f" try MoveLinearPastFork first"
                        )
                        # Skip transforming this node as moving this would lead
                        # to messed up or detached graph
                        continue
                    # check if second input of producer is an initializer
                    init0 = model.get_initializer(prod0.input[1])
                    # if either initializer is None, skip
                    if init0 is None:
                        continue
                    # if initializer is not scalar, skip
                    if np.prod(init0.shape) != 1:
                        continue
                    if model.is_fork_node(prod0):
                        model = model.transform(MoveOpPastFork(prod0.op_type))
                        # topology modified, "ask" ModelWrapper to apply this transform again
                        return (model, True)
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
                            ceil_mode = 0  # default to ceil_mode=0 (equivalent to np.floor)
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
                        # check if the producer is a fork node
                        # (need to move it past the fork before this transform)
                        if model.is_fork_node(producer):
                            model = model.transform(MoveTransposePastFork())
                            # topology modified, "ask" ModelWrapper to apply this transform again
                            return (model, True)
                        ceil_mode = get_by_name(n.attribute, "ceil_mode")
                        if ceil_mode is not None:
                            ceil_mode = ceil_mode.i
                        else:
                            ceil_mode = 0  # default to ceil_mode=0 (equivalent to np.floor)
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
                        "%s: Input not NCHW. Can't operate transformation on node." % n.name
                    )
                    continue
                consumer = model.find_consumer(n.output[0])
                producer = model.find_producer(n.input[0])
                if n.op_type == "Upsample":
                    transformation_ind = 1
                    d_type = "float32"
                else:
                    if len(n.input) == 2:
                        # Resize version 10
                        transformation_ind = 1
                        d_type = "float32"
                    elif len(n.input) == 3:
                        # Resize version 11 and up (no size input)
                        transformation_ind = 2
                        d_type = "float32"
                    elif len(n.input) == 4:
                        # Resize version 11 and up
                        scales_exists = (model.get_initializer(n.input[2]) is not None) and (
                            len(model.get_initializer(n.input[2])) != 0
                        )
                        sizes_exists = (model.get_initializer(n.input[3]) is not None) and (
                            len(model.get_initializer(n.input[3])) != 0
                        )
                        assert scales_exists ^ sizes_exists, (
                            "%s: Either scales or the target output size must "
                            "be specified. Specifying both is prohibited." % n.name
                        )
                        if scales_exists:
                            # Scales input
                            transformation_ind = 2
                            d_type = "float32"
                        else:
                            # Sizes input
                            transformation_ind = 3
                            d_type = "int64"
                if producer is not None and producer.op_type == "Transpose":
                    perms = list(get_by_name(producer.attribute, "perm").ints)
                    if perms == [0, 3, 1, 2]:
                        # check if the producer is a fork node
                        # (need to move it past the fork before this transform)
                        if model.is_fork_node(producer):
                            model = model.transform(MoveTransposePastFork())
                            # topology modified, "ask" ModelWrapper to apply this transform again
                            return (model, True)
                        old_value = model.get_initializer(n.input[transformation_ind])
                        new_value = np.array(
                            [old_value[idx] for idx in (0, 2, 3, 1)],
                            dtype=np.dtype(d_type),
                        )
                        model.set_initializer(n.input[transformation_ind], new_value)
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
                        old_value = model.get_initializer(n.input[transformation_ind])
                        new_value = np.array(
                            [old_value[idx] for idx in (0, 2, 3, 1)],
                            dtype=np.dtype(d_type),
                        )
                        model.set_initializer(n.input[transformation_ind], new_value)
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

    def __init__(self, op_name_list):
        super().__init__()
        self.ops_to_move = op_name_list

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
                    new_node = deepcopy(n)
                    new_node.input[:] = new_inp_list
                    new_node.output[:] = [new_output_tensor_name]
                    graph.node.insert(node_ind, new_node)
                    node_ind += 1

                    # change consumer input tensor
                    graph.node.remove(consumer_node)
                    for idx, consumer_input in enumerate(consumer_node.input):
                        if consumer_input == n.output[0]:
                            consumer_node.input[idx] = new_output_tensor_name
                            break
                    else:
                        raise Exception("Consumer should have the current node output as input")

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
        super().__init__(["Transpose"])


def permute_shape(shape, perm):
    new_shape = np.zeros(len(shape))
    for i, p in enumerate(perm):
        new_shape[i] = shape[p]
    return [int(el) for el in new_shape]


class MoveScalarLinearPastSplit(Transformation):
    """
    Move scalar Mul and Add nodes past channel split operation.
    """

    def __init__(self):
        super().__init__()
        self.ops_to_move = ["Mul", "Add"]
        self.fork_ops = ["Split"]

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        node_ind = 0
        for n in graph.node:
            node_ind += 1
            # if n.op_type in self.fork_ops and model.is_fork_node(n):
            if n.op_type in self.fork_ops:
                producer = model.find_producer(n.input[0])
                if producer is not None and producer.op_type in self.ops_to_move:
                    linear_param = model.get_initializer(producer.input[1])
                    # Check if single input
                    if len(producer.input) != 2 or linear_param is None:
                        continue
                    # Check if scalar
                    if np.prod(linear_param.shape) != 1:
                        continue
                    split_outputs = n.output
                    for split_output_idx, old_split_output in enumerate(split_outputs):
                        new_mul_node = deepcopy(producer)
                        new_split_output = model.make_new_valueinfo_name()
                        model.set_tensor_datatype(
                            new_split_output, model.get_tensor_datatype(producer.input[0])
                        )

                        model.set_tensor_shape(
                            new_split_output, model.get_tensor_shape(old_split_output)
                        )

                        n.output[split_output_idx] = new_split_output
                        new_mul_node.input[0] = new_split_output
                        new_mul_node.output[0] = old_split_output

                        graph.node.insert(node_ind, new_mul_node)
                        node_ind += 1

                    # remove the mul node
                    n.input[0] = producer.input[0]
                    graph.node.remove(producer)
                    graph_modified = True

        if graph_modified:
            model = model.transform(SortGraph(), make_deepcopy=False, cleanup=False)

        return (model, graph_modified)


class MoveTransposePastSplit(Transformation):
    def __init__(self):
        super().__init__()
        self.ops_to_move = ["Transpose"]
        self.fork_ops = ["Split"]

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        node_ind = 0
        for n in graph.node:
            node_ind += 1
            # if n.op_type in self.fork_ops and model.is_fork_node(n):
            if n.op_type in self.fork_ops:
                producer = model.find_producer(n.input[0])
                if producer is not None and producer.op_type in self.ops_to_move:
                    initial_perm = get_by_name(producer.attribute, "perm").ints
                    reverse_perm = np.argsort(initial_perm)
                    split_outputs = n.output
                    for split_output_idx, old_split_output in enumerate(split_outputs):
                        new_trans_node = deepcopy(producer)
                        new_split_output = model.make_new_valueinfo_name()
                        old_split_output_shape = model.get_tensor_shape(old_split_output)
                        model.set_tensor_datatype(
                            new_split_output, model.get_tensor_datatype(producer.input[0])
                        )

                        model.set_tensor_shape(
                            new_split_output, permute_shape(old_split_output_shape, reverse_perm)
                        )

                        n.output[split_output_idx] = new_split_output
                        new_trans_node.input[0] = new_split_output
                        new_trans_node.output[0] = old_split_output

                        graph.node.insert(node_ind, new_trans_node)
                        node_ind += 1

                    # remove the transpose node and change the split axis
                    old_split_axis = get_by_name(n.attribute, "axis").i
                    get_by_name(n.attribute, "axis").i = initial_perm[old_split_axis]
                    n.input[0] = producer.input[0]
                    graph.node.remove(producer)
                    graph_modified = True

        if graph_modified:
            model = model.transform(SortGraph(), make_deepcopy=False, cleanup=False)

        return (model, graph_modified)


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
                        warnings.warn("Skipping padded MaxPool + signed-output MultiThreshold")
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
            if n.op_type == "Flatten" and not model.is_fork_node(n) and not model.is_join_node(n):
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
            if n.op_type == "Transpose" and not model.is_fork_node(n) and not model.is_join_node(n):
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
    Move multiple identical operations on different branches past the common join node.
    It assumes the shape to be preserved by the join op in the default move_node() method
    """

    def __init__(self, identical_op_list, join_node_list):
        super().__init__()
        self.ops_to_move = identical_op_list
        self.join_node_op = join_node_list

    def move_node(self, model, n, producers):
        """
        Should be overwritten for some operations

        Returns:
            bool: whether moving the node was successful
        """
        identical_ops_inputs = [p.input[0] for p in producers]
        # join_in0 = n.input[0]
        join_out = n.output[0]

        # Rewire join op inputs
        for i in range(len(n.input)):
            n.input[i] = identical_ops_inputs[i]

        # Output tensor of the join node must have the same shape as
        # its input tensor (original shape is preserved)
        new_join_output = model.make_new_valueinfo_name()
        new_shape = model.get_tensor_shape(identical_ops_inputs[0])
        new_layout = model.get_tensor_layout(identical_ops_inputs[0])

        # Set new tensor shape
        model.set_tensor_shape(new_join_output, new_shape)
        if new_layout:
            model.set_tensor_layout(new_join_output, new_layout)

        # Rewire join op outputs (reuse the first join input tensor)
        n.output[0] = new_join_output
        producers[0].input[0] = new_join_output
        producers[0].output[0] = join_out

        for prod in producers[1:]:
            model.graph.node.remove(prod)

        return True

    def are_producers_identical(self, model, producers):
        """
        Checks only op_types
        Should be overwritten for additional checks
        """
        op_types = [prod.op_type for prod in producers]
        for op in op_types:
            if op != op_types[0]:
                return False
        return True

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        for n in graph.node:
            if n.op_type in self.join_node_op and model.is_join_node(n):
                inputs = n.input
                if None in inputs:
                    continue

                producers = [model.find_producer(inp) for inp in inputs]
                if producers[0].op_type not in self.ops_to_move:
                    continue
                identical_ops = self.are_producers_identical(model, producers)
                if not identical_ops:
                    warnings.warn("Producers not identical, skipping")
                    continue

                # check for producers that are fork nodes (need to fork them before our transform)
                for prod in producers:
                    if model.is_fork_node(prod) and not model.is_join_node(prod):
                        model = model.transform(MoveOpPastFork(self.ops_to_move))
                        # topology modified, "ask" ModelWrapper to apply this transform again
                        return (model, True)
                graph_modified = self.move_node(model, n, producers)

        if graph_modified:
            model = model.transform(SortGraph(), make_deepcopy=False, cleanup=False)

        return (model, graph_modified)


class MoveTransposePastJoinAdd(MoveIdenticalOpPastJoinOp):
    def __init__(self):
        super().__init__(["Transpose"], ["Add"])

    def are_producers_identical(self, model, producers):
        if not super().are_producers_identical(model, producers):
            return False
        first_perm = get_by_name(producers[0].attribute, "perm").ints
        for producer in producers:
            if first_perm != get_by_name(producer.attribute, "perm").ints:
                False
        return True


class MoveTransposePastJoinMul(MoveIdenticalOpPastJoinOp):
    def __init__(self):
        super().__init__(["Transpose"], ["Mul"])

    def are_producers_identical(self, model, producers):
        if not super().are_producers_identical(model, producers):
            return False
        first_perm = get_by_name(producers[0].attribute, "perm").ints
        for producer in producers:
            if first_perm != get_by_name(producer.attribute, "perm").ints:
                False
        return True


class MoveMulPastJoinAdd(MoveIdenticalOpPastJoinOp):
    def __init__(self):
        super().__init__(["Mul"], ["Add"])

    def are_producers_identical(self, model, producers):
        if not super().are_producers_identical(model, producers):
            return False
        first_mul = model.get_initializer(producers[0].input[1])
        if first_mul is None:
            return False
        for producer in producers:
            if first_mul != model.get_initializer(producer.input[1]):
                return False
        return True


class MoveAddPastJoinAdd(MoveIdenticalOpPastJoinOp):
    def __init__(self):
        super().__init__(["Add"], ["Add"])

    def are_producers_identical(self, model, producers):
        if not super().are_producers_identical(model, producers):
            return False
        for producer in producers:
            if model.get_initializer(producer.input[1]) is None:
                return False
        return True

    def move_node(self, model, n, producers):
        """
        We use the base move_node method to move the first producer
        past the join node (and delete the rest)
        """
        add_inits = [model.get_initializer(producer.input[1]) for producer in producers]
        new_init = np.sum(add_inits)
        model.set_initializer(producers[0].input[1], new_init)
        super().move_node(model, n, producers)

        return True


class MoveTransposePastJoinConcat(MoveIdenticalOpPastJoinOp):
    def __init__(self):
        super().__init__(["Transpose"], ["Concat"])

    def are_producers_identical(self, model, producers):
        if not super().are_producers_identical(model, producers):
            return False
        first_perm = get_by_name(producers[0].attribute, "perm").ints
        for producer in producers:
            if first_perm != get_by_name(producer.attribute, "perm").ints:
                False
        return True

    def move_node(self, model, n, producers):
        trans_inputs = [prod.input[0] for prod in producers]
        # concat_in0 = n.input[0]
        concat_out = n.output[0]
        # Rewire concat inputs
        for i in range(len(n.input)):
            n.input[i] = trans_inputs[i]

        new_concat_out = model.make_new_valueinfo_name()  # reuse tensor
        # reverse the permutation of the concat output
        transpose_perm = get_by_name(producers[0].attribute, "perm").ints
        reverse_perm = np.argsort(transpose_perm)
        new_concat_out_shape = permute_shape(model.get_tensor_shape(concat_out), reverse_perm)
        new_concat_out_layout = model.get_tensor_layout(trans_inputs[0])
        # Set tensor layout and shape of the new concatenation output
        model.set_tensor_shape(new_concat_out, new_concat_out_shape)
        if new_concat_out_layout:
            model.set_tensor_layout(new_concat_out, new_concat_out_layout)
        # Change concatenation axis
        old_concat_axis = get_by_name(n.attribute, "axis").i
        get_by_name(n.attribute, "axis").i = transpose_perm[old_concat_axis]

        # Rewire concat output
        n.output[0] = new_concat_out
        producers[0].input[0] = new_concat_out
        producers[0].output[0] = concat_out

        for prod in producers[1:]:
            model.graph.node.remove(prod)

        return True


class MoveAffinePastJoinConcat(MoveIdenticalOpPastJoinOp):
    """
    Applies to scalar linear or channelwise affine ops with the same parameter value
    """

    def __init__(self, linear_ops=["Mul", "Add"]):
        super().__init__(linear_ops, ["Concat"])

    def are_producers_identical_scalar_ops(self, model, producers):
        first_param = model.get_initializer(producers[0].input[1])
        for producer in producers:
            producer_param = model.get_initializer(producer.input[1])
            if (first_param != producer_param).any() or np.prod(producer_param.shape) != 1:
                return False

        return True

    def are_producers_channelwise_ops(self, channel_dim, model, producers):
        for producer in producers:
            producer_input = producer.input[0]
            num_channels = model.get_tensor_shape(producer_input)[channel_dim]
            producer_param = model.get_initializer(producer.input[1])
            if (
                len(producer_param.shape) < channel_dim
                or producer_param.shape[channel_dim] != num_channels
            ):
                return False

        return True

    def move_node(self, model, n, producers):
        # check if single input
        for producer in producers:
            producer_init = model.get_initializer(producer.input[1])
            if len(producer.input) != 2 or producer_init is None:
                warnings.warn("Producer found that is not single-input, skipping")
                return False

        # decide if producers are identical scalar ops or channelwise ops
        channelwise_op = False
        identical_scalar_op = self.are_producers_identical_scalar_ops(model, producers)
        if not identical_scalar_op:
            channel_dim = get_by_name(n.attribute, "axis").i
            channelwise_op = self.are_producers_channelwise_ops(channel_dim, model, producers)
            if not channelwise_op:
                warnings.warn(
                    "Producers are neither identical scalar ops nor channelwise ops, skipping"
                )
                return False

        # Rewire concat inputs
        producers_inputs = [prod.input[0] for prod in producers]
        concat_out = n.output[0]
        for i in range(len(n.input)):
            n.input[i] = producers_inputs[i]
        # Set tensor layout and shape of the new concatenation output
        new_concat_out = model.make_new_valueinfo_name()
        new_concat_out_layout = model.get_tensor_layout(producers_inputs[0])
        model.set_tensor_shape(new_concat_out, model.get_tensor_shape(concat_out))
        if new_concat_out_layout:
            model.set_tensor_layout(new_concat_out, new_concat_out_layout)
        model.set_tensor_datatype(new_concat_out, model.get_tensor_datatype(producers_inputs[0]))

        if channelwise_op:
            # concatenate op params of producers into one mul tensor
            producers_params = [model.get_initializer(prod.input[1]) for prod in producers]
            new_mul_tensor = np.concatenate(producers_params, axis=channel_dim)
            model.set_initializer(producers[0].input[1], new_mul_tensor)

        # Rewire concat output
        n.output[0] = new_concat_out
        producers[0].input[0] = new_concat_out
        producers[0].output[0] = concat_out

        for prod in producers[1:]:
            model.graph.node.remove(prod)

        return True


class MoveMulPastJoinConcat(MoveAffinePastJoinConcat):
    def __init__(self):
        super().__init__(["Mul"])


class MoveAddPastJoinConcat(MoveAffinePastJoinConcat):
    def __init__(self):
        super().__init__(["Add"])


# Moves a Squeeze operation past MultiThresholds
# TODO: extend to all operations invariant to or compatible with squeezing
class MoveSqueezePastMultiThreshold(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Applies to Squeeze operation types
            if node.op_type == "Squeeze":
                # Currently does not handle fork- or join-nodes
                if model.is_fork_node(node) or model.is_join_node(node):
                    # Softly skip this node
                    continue
                # As this is not a fork-node, there can be at most one successor
                successor = model.find_direct_successors(node)
                # If Squeeze is the final operation in the graph, there might
                # be no successor
                if successor is None:
                    # Softly skip this node
                    continue
                # Now there is exactly one successor which needs to be extracted
                # from the list
                successor = successor[0]
                # Applies to MultiThreshold
                if successor.op_type in {"MultiThreshold"}:
                    # Get names of all tensors involved in connecting the nodes
                    inp = node.input[0]  # noqa: Duplicate
                    mid = node.output[0]
                    out = successor.output[0]
                    # Rewire the graph to feed original into the MultiThreshold
                    # node first
                    successor.input[0] = inp
                    # Repurpose the middle tensor for the output of the
                    # MultiThreshold
                    successor.output[0] = mid
                    # The Squeeze operator now gets the middle tensor as its
                    # input
                    node.input[0] = mid
                    # Squeeze now produces the original output tensor
                    node.output[0] = out
                    # Delete the shape annotation of the connecting tensors
                    # to be re-done later
                    model.set_tensor_shape(mid, None)
                    model.set_tensor_shape(out, None)
                    # Track whether the graph has been modified, never
                    # resets to False
                    graph_modified = True
                    # Break the loop after deleting shape annotations to
                    # immediately re-do these before changing the next
                    # operator
                    break
        # Need to redo the shape inference after potentially deleting them
        model = model.transform(InferShapes())  # noqa: Shadows model
        # Return the transformed model and indicate whether the graph
        # actually has been transformed
        return model, graph_modified


# Moves a Squeeze operation past MatMul
# TODO: extend to all operations invariant to or compatible with squeezing
class MoveSqueezePastMatMul(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Applies to Squeeze operation types
            if node.op_type == "Squeeze":
                # Currently does not handle fork- or join-nodes
                if model.is_fork_node(node) or model.is_join_node(node):
                    # Softly skip this node
                    continue
                # As this is not a fork-node, there can be at most one successor
                successor = model.find_direct_successors(node)
                # If Squeeze is the final operation in the graph, there might
                # be no successor
                if successor is None:
                    # Softly skip this node
                    continue
                # Now there is exactly one successor which needs to be extracted
                # from the list
                successor = successor[0]
                # Applies to MatMul
                # TODO: Check behavior for multi-dimensional and potentially
                #  broadcasting MatMuls...
                if successor.op_type in {"MatMul"}:
                    # Get names of all tensors involved in  # noqa: Duplicate
                    # connecting the nodes
                    inp = node.input[0]  # noqa: Duplicate
                    mid = node.output[0]
                    out = successor.output[0]
                    # Rewire the graph to feed original into the MultiThreshold
                    # node first
                    successor.input[0] = inp
                    # Repurpose the middle tensor for the output of the
                    # MultiThreshold
                    successor.output[0] = mid
                    # The Squeeze operator now gets the middle tensor as its
                    # input
                    node.input[0] = mid
                    # Squeeze now produces the original output tensor
                    node.output[0] = out
                    # Delete the shape annotation of the connecting tensors
                    # to be re-done later
                    model.set_tensor_shape(mid, None)
                    model.set_tensor_shape(out, None)
                    # Track whether the graph has been modified, never
                    # resets to False
                    graph_modified = True
                    # Break the loop after deleting shape annotations to
                    # immediately re-do these before changing the next
                    # operator
                    break
        # Need to redo the shape inference after potentially deleting them
        model = model.transform(InferShapes())  # noqa: Shadows model
        # Return the transformed model and indicate whether the graph
        # actually has been transformed
        return model, graph_modified


# Moves a transpose operator past elementwise addition or multiplication
class MoveTransposePastEltwise(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Applies to Transpose operation types
            if node.op_type == "Transpose":
                # Currently does not handle fork- or join-nodes
                if model.is_fork_node(node) or model.is_join_node(node):
                    # Softly skip this node
                    continue
                # As this is not a fork-node, there can be at most one successor
                successor = model.find_direct_successors(node)
                # If Transpose is the final operation in the graph, there might
                # be no successor
                if successor is None:
                    # Softly skip this node
                    continue
                # Now there is exactly one successor which needs to be extracted
                # from the list
                successor = successor[0]
                # Applies to elementwise add and mul operations
                if successor.op_type in {"Add", "Mul"}:
                    # Get names of all tensors involved in connecting the nodes
                    inp = node.input[0]
                    mid = node.output[0]
                    out = successor.output[0]

                    # y = x^T + a <=> y = (x + a^T)^T

                    # Assume left-to-right order of input to the Add operator
                    xt, a = successor.input
                    # Check whether the assumption holds true
                    if xt != mid:
                        # Leaves only the option of a and xt commuting
                        xt, a = a, xt
                    # If this assumption still does not hold true, something is
                    # wrong with the graph
                    assert xt == mid, f"Messed up graph pattern at {node.name}"

                    # Get the (optional) permutation indices of the transpose in
                    # case it is a multi-axis transpose
                    perm = get_by_name(node.attribute, "perm")
                    # Convert permutation indices to list of integers
                    perm = list(perm.ints) if perm is not None else None

                    # Inverse permutation needs to be applied to the initializer
                    # fmt: off
                    inverse_perm = None if not perm else [
                        perm.index(i) for i in range(len(perm))
                    ]
                    # fmt: on

                    # This transformation does only apply to Add nodes where the
                    # second input is a constant initializer
                    if (value := model.get_initializer(a)) is not None:
                        # Do not transpose scalar or effectively scalar
                        # initializers
                        # fmt: off
                        if not (value.shape is None or all(
                                x == 1 for x in value.shape)):
                            # fmt: on
                            # Transpose the initializer and re-insert into the
                            # model
                            # fmt: off
                            model.set_initializer(
                                a, value.transpose(inverse_perm)
                            )
                            # fmt: on
                        # Rewire the graph to feed original input and the
                        # transposed initializer into the Add node first
                        successor.input[:] = [inp, a]
                        # Repurpose the middle tensor for the output of the
                        # addition
                        successor.output[0] = mid
                        # The Transpose operator now gets the middle tensor as
                        # its input
                        node.input[0] = mid
                        # Transpose now produces the original output tensor
                        node.output[0] = out
                        # Delete the shape annotation of the connecting tensors
                        # to be re-done later
                        model.set_tensor_shape(inp, None)
                        model.set_tensor_shape(mid, None)
                        model.set_tensor_shape(out, None)
                        # Track whether the graph has been modified, never
                        # resets to False
                        graph_modified = True
                        # Break the loop after deleting shape annotations to
                        # immediately re-do these before changing the next
                        # operator
                        break
        # Need to redo the shape inference after potentially removing nodes
        model = model.transform(InferShapes())  # noqa: Shadows model
        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified


# Moves elementwise additions past MatMul operations: Applicable if each
# operation has one initializer input
class MoveAddPastMatMul(Transformation):
    # Applies the transform to a whole model graph  # noqa: Duplicate
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Applies to Add operations
            if node.op_type == "Add":
                # If the add is a join operation, we do not have a constant
                # added to the input
                if model.is_join_node(node):
                    # Skip transforming this
                    continue
                # If the Add is a fork operation we should first distribute the
                # Add into the branches
                if model.is_fork_node(node):
                    # Issue a warning to make the use aware of this potential
                    # transformation if the fork is moved first
                    warnings.warn(
                        f"{self.__class__.__name__}:"
                        f" Skipping near match: {node.name} is a fork-node,"
                        f" try MoveLinearPastFork first"
                    )
                    # Skip transforming this node as moving this would lead
                    # to messed up or detached graph
                    continue
                # Decompose the inputs into the dynamic and the constant
                # initializer input
                (x_name,), (c_name,) = group_inputs_by_category(node, model)
                # Now check the successor node which must be a MatMul
                consumer = model.find_direct_successors(node)
                # If there is no consumer, this Add seems to be last node of the
                # graph
                if not consumer:
                    # Skip transforming this
                    continue
                # There must be exactly one consumer now
                consumer = consumer[0]
                # This transformation only applies to Add in front of MatMul
                if not consumer.op_type == "MatMul":
                    # Skip this if not MatMul
                    continue
                # MatMul may not be a join operation to apply this
                # transformation
                if model.is_join_node(consumer):
                    # Skip transforming without warning (there is nothing we can
                    # do about this)
                    continue
                # Decompose the inputs to the MatMul to get the weight tensor
                # name (the other input is the output of the Add)
                _, (w_name,) = group_inputs_by_category(consumer, model)
                # Read the weights and the constant addition tensor
                w = model.get_initializer(w_name)
                c = model.get_initializer(c_name)
                # Determine whether the weights are the left or right input to
                # the MatMul
                left = w_name == consumer.input[0]
                # Apply the weights to the constant tensor
                c = np.matmul(w, c) if left else np.matmul(c, w)
                # Insert the transformed tensor back into the mode as an
                # initializer
                model.set_initializer(c_name, c)
                # The connecting tensors of this pattern
                inp = x_name
                mid = node.output[0]
                out = consumer.output[0]
                # Rewire the graph pattern connecting the input to the MatMul
                # and the MatMul output to the Add node
                consumer.input[1 if left else 0] = inp
                # The Add now produces the original MatMul output
                node.output[0] = out
                # The middel tensor connects to the Add input
                node.input[0 if node.input[0] == x_name else 1] = mid
                # The MatMul feeds the middle tensors
                consumer.output[0] = mid
                # Delete the shape annotation of the connecting tensors
                # to be re-done later
                model.set_tensor_shape(mid, None)
                model.set_tensor_shape(out, None)
                # Delete the type annotations of the connecting tensors
                # to be re-done later
                # model.set_tensor_datatype(mid, None)
                # model.set_tensor_datatype(out, None)
                # Track whether the graph has been modified, never
                # resets to False
                graph_modified = True
                # Break the loop after deleting shape annotations to
                # immediately re-do these before changing the next
                # operator
                break
        # Redo datatype and shape annotations
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        # Return the transformed model and indicate whether the transformation
        # needs to be applied again
        return model, graph_modified


# Moves constant elementwise multiplication past another joining multiplication
class MoveConstMulPastJoinMul(Transformation):
    # Applies the transform to a whole model graph  # noqa: Duplicate
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Applies to Mul operation types
            if node.op_type == "Mul":
                # Currently does not handle fork- or join-nodes
                if model.is_fork_node(node) or model.is_join_node(node):
                    # Softly skip this node
                    continue
                # As this is not a fork-node, there can be at most one successor
                successor = model.find_direct_successors(node)
                # If Squeeze is the final operation in the graph, there might
                # be no successor
                if successor is None:
                    # Softly skip this node
                    continue
                # Now there is exactly one successor which needs to be extracted
                # from the list
                successor = successor[0]
                # Applies to Multiplications
                if successor.op_type in {"Mul"}:
                    # Applies only if the second multiplication is a join-node
                    if model.is_join_node(successor):
                        # Get names of all tensors involved in connecting the
                        # nodes
                        inp = node.input[0]  # noqa: Duplicate
                        mid = node.output[0]
                        out = successor.output[0]
                        # Need to match the correct input of the joining second
                        # multiplication
                        for i, name in enumerate(successor.input):
                            # If the successors input currently matches the
                            # intermediate tensors, this input needs to be
                            # rewired
                            if name == mid:
                                # Rewire the graph to feed original into the
                                # second Mul node first
                                successor.input[i] = inp
                                # Note: Do not break here as it is perfectly
                                # legal to connect the same tensor multiple
                                # times to different inputs
                        # Repurpose the middle tensor for the output of the
                        # second Mul
                        successor.output[0] = mid
                        # The first Mul operator now gets the middle tensor as
                        # its input
                        node.input[0] = mid
                        # The first Mul now produces the original output tensor
                        node.output[0] = out
                        # Delete the shape annotation of the connecting tensors
                        # to be re-done later
                        model.set_tensor_shape(mid, None)
                        model.set_tensor_shape(out, None)
                        # Track whether the graph has been modified, never
                        # resets to False
                        graph_modified = True
                        # Break the loop after deleting shape annotations to
                        # immediately re-do these before changing the next
                        # operator
                        break
        # Redo datatype and shape annotations
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        # Return the transformed model and indicate whether the transformation
        # needs to be applied again
        return model, graph_modified


# Moves elementwise multiplication past elementwise addition if one input to
# each of the operators is a known constant
# Note: Reverse of MoveAddPastMul
class MoveMulPastAdd(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Applies to Mul operation types
            if node.op_type == "Mul":
                # Currently does not handle fork- or join-nodes
                if model.is_fork_node(node) or model.is_join_node(node):
                    # Softly skip this node
                    continue
                # As this is not a fork-node, there can be at most one successor
                successor = model.find_direct_successors(node)
                # If Squeeze is the final operation in the graph, there might
                # be no successor
                if successor is None:
                    # Softly skip this node
                    continue
                # Now there is exactly one successor which needs to be extracted
                # from the list
                successor = successor[0]
                # Applies to additions
                if successor.op_type in {"Add"}:
                    # The addition may not join as we need to know the second
                    # input
                    if not model.is_join_node(successor):
                        # Get the constant initializer tensors for both
                        # operations: y = s * x + b
                        _, s_name = group_inputs_by_category(node, model)
                        _, b_name = group_inputs_by_category(successor, model)
                        # Skip if either node has no constant initializer
                        if not s_name or not b_name:
                            # Skip without warning ok?
                            continue
                        # There must be exactly one constant per operations
                        assert len(s_name) == 1, f"To many constant inputs for {node}"
                        assert len(b_name) == 1, f"To many constant inputs for {successor}"
                        # Now read the initializer tensors
                        s = model.get_initializer(*s_name)
                        b = model.get_initializer(*b_name)
                        # Update the addition initializer according to the
                        # distributive law
                        model.set_initializer(*b_name, b / s)
                        # Get names of all tensors involved in connecting the
                        # nodes
                        inp = node.input[0]  # noqa: Duplicate
                        mid = node.output[0]
                        out = successor.output[0]
                        # Rewire the graph to feed original input into the
                        # Add node first
                        successor.input[0] = inp
                        # Repurpose the middle tensor for the output of the Add
                        successor.output[0] = mid
                        # The Mul operator now gets the middle tensor as its
                        # input
                        node.input[0] = mid
                        # Mul now produces the original output tensor
                        node.output[0] = out
                        # Delete the shape annotation of the connecting tensors
                        # to be re-done later
                        model.set_tensor_shape(mid, None)
                        model.set_tensor_shape(out, None)
                        # Track whether the graph has been modified, never
                        # resets to False
                        graph_modified = True
                        # Break the loop after deleting shape annotations to
                        # immediately re-do these before changing the next
                        # operator
                        break
        # Redo datatype and shape annotations
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        # Return the transformed model and indicate whether the transformation
        # needs to be applied again
        return model, graph_modified


# Moves scalar linear elementwise operations past fork nodes, applies to Add,
# Mul, Sub, Div, etc.
class MoveScalarLinearPastFork(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Applies to Mul-like and Add-like operation types
            if node.op_type in {"Add", "Sub", "Mul", "Div"}:
                # Only handles non-joining forks for now
                if not model.is_fork_node(node) or model.is_join_node(node):
                    # Softly skip this node
                    continue
                # Only handles one forking output for now
                if len(node.output) > 1:
                    # Softly skip this node
                    continue
                # Left and right side of the operation
                (inp,), (const,) = group_inputs_by_category(node, model)
                # Test whether the node initializer is a scalar...
                if not is_scalar(model.get_initializer(const)):
                    # Softly skip this node
                    continue
                # We need to insert a replica of this operation in front of each
                # consumer node
                for consumer in model.find_direct_successors(node):
                    # Create an exact replica of this operator
                    copy = deepcopy(node)
                    # Insert a new unique tensor connecting the output of the
                    # copy to the consumer
                    copy.output[0] = model.make_new_valueinfo_name()
                    # The original node might be connecting to multiple inputs
                    # of the consumer...
                    for idx, inp in enumerate(consumer.input):
                        # Find each instance of connection from original node
                        if inp == node.output[0]:
                            # Rewire to connect to the replica
                            consumer.input[idx] = copy.output[0]
                    # Insert the new replica node into the graph
                    graph.node.insert(index + 1, copy)
                # Remove the original node from the graph
                graph.node.remove(node)
        # Redo datatype and shape annotations
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        # Return the transformed model and indicate whether the transformation
        # needs to be applied again
        return model, graph_modified


# Moves scalar linear channel-wise operations past fork nodes, applies to Add,
# Mul, Sub, Div, etc.
class MoveChannelwiseLinearPastFork(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Applies to Mul-like and Add-like operation types
            if node.op_type in {"Add", "Sub", "Mul", "Div"}:
                # Only handles non-joining forks for now
                if not model.is_fork_node(node) or model.is_join_node(node):
                    # Softly skip this node
                    continue
                # Only handles one forking output for now
                if len(node.output) > 1:
                    # Softly skip this node
                    continue

                # Left and right side of the operation
                (inp,), (const,) = group_inputs_by_category(node, model)

                # First try to consider the tensor layout of the input for
                # determining the number of input channels
                layout = model.get_tensor_layout(inp)
                # If there is no layout annotation, guess based on rank of the
                # tensor
                if layout is None:
                    # Maps tensor rank to layout annotation
                    rank_to_layout = {0: None, 1: "C", 2: "NC", 3: "NWC", 4: "NCHW"}
                    # Lookup the layout required by this input shape
                    layout = rank_to_layout[len(model.get_tensor_shape(inp))]
                # If there is a layout annotation, use this to determine the
                # index of the channel dimension
                if layout is not None and "C" in layout:
                    # Lookup the index in list
                    cdim = layout.index("C")
                # If no layout has been annotated or there is no channel
                # dimension, fall back to the previous default assumption
                else:
                    # Assume the channels to be in axis 1
                    cdim = 1
                    # Issue a warning to the user, so they are aware of this
                    warnings.warn(
                        f"{self.__class__.__name__}: No layout for {inp}:"
                        f" Assuming channel dimension at index {cdim}"
                    )

                # Tests whether two shapes can be broadcast according to NumPy
                # semantics
                def can_broadcast_to(lhs, rhs):
                    # Broadcasting might raise an exception
                    try:
                        # Try broadcasting the shapes
                        if np.broadcast_to(np.zeros(lhs), rhs).shape == rhs:
                            # These tensors can be broadcast, preserving the
                            # left-hand-side shape
                            return True
                        # These tensors cannot be broadcast
                        return False
                    # Failing to broadcast the tensors raises ValueError
                    except ValueError:
                        # These tensors cannot be broadcast
                        return False

                # Per-tensor or per-channel means we have some parameter tensor
                # which can be broadcast to the channel dimension of the output
                if not can_broadcast_to(
                    model.get_tensor_shape(const), (model.get_tensor_shape(node.output[0])[cdim],)
                ):
                    # Issue a warning to the user, so they are aware of this
                    warnings.warn(f"{self.__class__.__name__}: Not channel-wise {const}:")
                    # Softly skip this node
                    continue

                # We need to insert a replica of this operation in front of each
                # consumer node
                for consumer in model.find_direct_successors(node):
                    # Create an exact replica of this operator
                    copy = deepcopy(node)
                    # Insert a new unique tensor connecting the output of the
                    # copy to the consumer
                    copy.output[0] = model.make_new_valueinfo_name()
                    # The original node might be connecting to multiple inputs
                    # of the consumer...
                    for idx, inp in enumerate(consumer.input):
                        # Find each instance of connection from original node
                        if inp == node.output[0]:
                            # Rewire to connect to the replica
                            consumer.input[idx] = copy.output[0]
                    # Insert the new replica node into the graph
                    graph.node.insert(index + 1, copy)
                # Remove the original node from the graph
                graph.node.remove(node)
        # Redo datatype and shape annotations
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        # Return the transformed model and indicate whether the transformation
        # needs to be applied again
        return model, graph_modified


# Moves scale factor, i.e., scalar Mul and Div, past Im2Col (and Col2Im): These
# cannot be handled by MoveScalarLinearPastInvariants as potential padding makes
# Add-Im2Col not commute to Im2Col-Add
class MoveScalesPastIm2Col(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Applies to Mul operation types
            if node.op_type in {"Mul", "Div"}:
                # Cannot handle fork- or join-multiplications
                if model.is_fork_node(node) or model.is_join_node(node):
                    # Softly skip this node
                    continue
                # Only handles one forking output for now
                if len(node.output) > 1:
                    # Softly skip this node
                    continue
                # The first input must be dynamically received from upstream
                if model.get_initializer(node.input[0]) is not None:
                    # Softly skip this node
                    continue
                # Test whether the node initializer is a scalar...
                if not is_scalar(model.get_initializer(node.input[1])):
                    # Softly skip this node
                    continue
                # As this is not a fork-node, there can be at most one successor
                successor = model.find_direct_successors(node)
                # If this is the final operation in the graph, there might be no
                # successor
                if successor is None:
                    # Softly skip this node
                    continue
                # Now there is exactly one successor which needs to be extracted
                # from the list
                successor = successor[0]
                # Handle both, Im2Col and the inverse Col2Im, as well as padding
                if successor.op_type in {"Im2Col", "Col2Im", "Pad"}:
                    # Get names of all tensors involved in connecting the
                    # nodes
                    inp = node.input[0]  # noqa: Duplicate
                    mid = node.output[0]
                    out = successor.output[0]
                    # Rewire the graph to feed original input into the
                    # Add node first
                    successor.input[0] = inp
                    # Repurpose the middle tensor for the output of the Add
                    successor.output[0] = mid
                    # The Mul operator now gets the middle tensor as its
                    # input
                    node.input[0] = mid
                    # Mul now produces the original output tensor
                    node.output[0] = out
                    # Delete the shape annotation of the connecting tensors
                    # to be re-done later
                    model.set_tensor_shape(mid, None)
                    model.set_tensor_shape(out, None)
                    # Track whether the graph has been modified, never
                    # resets to False
                    graph_modified = True
                    # Break the loop after deleting shape annotations to
                    # immediately re-do these before changing the next
                    # operator
                    break
        # Redo datatype and shape annotations
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        # Return the transformed model and indicate whether the transformation
        # needs to be applied again
        return model, graph_modified
