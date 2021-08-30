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
from onnx import helper as oh

import finn.core.data_layout as DataLayout
from finn.core.datatype import DataType
from finn.custom_op.registry import getCustomOp
from finn.transformation.base import Transformation
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_shapes import InferShapes
from finn.util.basic import get_by_name


class AbsorbSignBiasIntoMultiThreshold(Transformation):
    """Absorb scalar bias originating from signed int export back into
    MultiThreshold and re-evaluate the output datatype."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            # search for (MultiThreshold, Add) pair
            node_ind += 1
            if (
                n.op_type == "MultiThreshold"
                and not model.is_fork_node(n)
                and not model.is_join_node(n)
            ):
                consumer = model.find_consumer(n.output[0])
                if consumer is not None and consumer.op_type == "Add":
                    mt_node = n
                    add_node = consumer
                    threshold_name = mt_node.input[1]
                    add_weight_name = add_node.input[1]
                    T = model.get_initializer(threshold_name)
                    A = model.get_initializer(add_weight_name)
                    if (A is None) or (T is None):
                        warnings.warn("Threshold or add bias not constant, skipping")
                        continue
                    end_name = add_node.output[0]
                    # we can only absorb scalar adds
                    is_scalar = A.ndim == 0 or all(x == 1 for x in A.shape)
                    if not is_scalar:
                        continue
                    bias = A.flatten()[0]
                    # set MultiThreshold bias property
                    mt_inst = getCustomOp(mt_node)
                    bias += mt_inst.get_nodeattr("out_bias")
                    mt_inst.set_nodeattr("out_bias", bias)
                    graph_modified = True
                    # compute new DataType for MultiThreshold output
                    steps = T.shape[-1]
                    new_min = bias
                    new_max = steps + bias
                    odt = DataType.get_smallest_possible(steps).name.replace(
                        "UINT", "INT"
                    )
                    odt = DataType[odt]
                    assert odt.allowed(new_max) and odt.allowed(
                        new_min
                    ), """Could
                    not compute new MultiThreshold DataType (min = %d max = %d)""" % (
                        new_min,
                        new_max,
                    )
                    mt_inst.set_nodeattr("out_dtype", odt.name)
                    # remove Add node, rewire MultiThreshold
                    graph.node.remove(add_node)
                    mt_node.output[0] = end_name
                    # set datatype
                    model.set_tensor_datatype(end_name, odt)
        if graph_modified:
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class AbsorbAddIntoMultiThreshold(Transformation):
    """Absorb preceding Add ops into MultiThreshold by updating the threshold
    values. Only scalar/1D add vectors can be absorbed."""

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
                if consumer is not None and consumer.op_type == "MultiThreshold":
                    add_weight_name = n.input[1]
                    threshold_name = consumer.input[1]
                    A = model.get_initializer(add_weight_name)
                    T = model.get_initializer(threshold_name)
                    assert A is not None, "Initializer for add weights is not set."
                    assert T is not None, "Initializer for thresholds is not set."
                    start_name = n.input[0]
                    # we can only absorb 0d or 1d adds
                    is_scalar = A.ndim == 0 or all(x == 1 for x in A.shape)
                    actual_ndims = len(tuple(filter(lambda x: x > 1, A.shape)))
                    is_1d = actual_ndims == 1
                    if is_scalar or is_1d:
                        Tnew = T - A.reshape(-1, 1)
                        # Tnew = T - A.reshape(-1, T.shape[1])
                        # compute new thresholds and set initializer
                        model.set_initializer(threshold_name, Tnew)
                        # wire add input directly to MultiThreshold
                        consumer.input[0] = start_name
                        # remove the add node
                        graph.node.remove(n)
                        graph_modified = True
        return (model, graph_modified)


class AbsorbMulIntoMultiThreshold(Transformation):
    """Absorb preceding Mul ops into MultiThreshold by updating the threshold
    values. Only *positive* scalar/1D mul vectors can be absorbed."""

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
                mul_weight_name = n.input[1]
                A = model.get_initializer(mul_weight_name)
                assert A is not None, "Initializer for mul weights is not set."
                is_signed = (A < 0).any()
                is_scalar = A.ndim == 0 or all(x == 1 for x in A.shape)
                actual_ndims = len(tuple(filter(lambda x: x > 1, A.shape)))
                is_1d = actual_ndims == 1
                consumer = model.find_consumer(n.output[0])
                if consumer is not None and consumer.op_type == "MultiThreshold":
                    if not is_signed and (is_1d or is_scalar):
                        threshold_name = consumer.input[1]
                        T = model.get_initializer(threshold_name)
                        assert T is not None, "Initializer for thresholds is not set."
                        start_name = n.input[0]
                        # compute new thresholds and set initializer
                        Tnew = T / A.reshape(-1, 1)
                        # TODO: need to handle negative A values correctly; produce
                        # mul sign mask and merge into preceding matmul?
                        model.set_initializer(threshold_name, Tnew)
                        # wire add input directly to MultiThreshold
                        consumer.input[0] = start_name
                        # remove the mul node
                        graph.node.remove(n)
                        graph_modified = True
        return (model, graph_modified)


class FactorOutMulSignMagnitude(Transformation):
    """Split multiply-by-constant nodes into two multiply-by-constant nodes,
    where the first node is a bipolar vector (of signs) and the second is a
    vector of magnitudes."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Mul":
                mul_weight_name = n.input[1]
                A = model.get_initializer(mul_weight_name)
                assert A is not None, "Initializer for mul weights is not set."
                is_scalar = np.prod(A.shape) == 1
                actual_ndims = len(tuple(filter(lambda x: x > 1, A.shape)))
                is_1d = actual_ndims == 1
                is_not_bipolar = (
                    model.get_tensor_datatype(mul_weight_name) != DataType.BIPOLAR
                )
                is_signed = (A < 0).any()
                if is_signed and (is_scalar or is_1d) and is_not_bipolar:
                    start_name = n.input[0]
                    in_shape = model.get_tensor_shape(start_name)
                    middle_name = model.make_new_valueinfo_name()
                    model.set_tensor_shape(middle_name, in_shape)
                    sign_mul_param_name = model.make_new_valueinfo_name()
                    # create new mul node with sign(A) as the operand
                    sgn = np.sign(A)
                    model.set_initializer(sign_mul_param_name, sgn)
                    model.set_tensor_datatype(sign_mul_param_name, DataType.BIPOLAR)
                    # replace original mul weight by magnitudes
                    model.set_initializer(mul_weight_name, np.abs(A))
                    new_mul = oh.make_node(
                        "Mul", [start_name, sign_mul_param_name], [middle_name]
                    )
                    n.input[0] = middle_name
                    graph.node.insert(node_ind - 1, new_mul)
                    graph_modified = True
        return (model, graph_modified)


class Absorb1BitMulIntoMatMul(Transformation):
    """Absorb bipolar or binary multiplications into the preciding matrix
    multiply."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "MatMul":
                matmul_weight_name = n.input[1]
                W = model.get_initializer(matmul_weight_name)
                Wdt = model.get_tensor_datatype(matmul_weight_name)
                assert W is not None, "Initializer for matmul weights is not set."
                consumer = model.find_consumer(n.output[0])
                if consumer is not None and consumer.op_type == "Mul":
                    mul_weight_name = consumer.input[1]
                    A = model.get_initializer(mul_weight_name)
                    assert A is not None, "Initializer for mul weights is not set."
                    is_1bit = model.get_tensor_datatype(mul_weight_name).bitwidth() == 1
                    if is_1bit:
                        Wnew = A * W
                        assert (
                            Wnew.shape == W.shape
                        ), """Shape of new weights is not
                        the same as the shape of the weight matrix before."""
                        check_fxn = np.vectorize(lambda x: Wdt.allowed(x))
                        # only absorb if permitted by W datatype
                        if check_fxn(Wnew).all():
                            model.set_initializer(matmul_weight_name, Wnew)
                            n.output[0] = consumer.output[0]
                            graph.node.remove(consumer)
                            graph_modified = True
        return (model, graph_modified)


class Absorb1BitMulIntoConv(Transformation):
    """Absorb bipolar or binary multiplications into the preciding convolution."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Conv":
                conv_weight_name = n.input[1]
                W = model.get_initializer(conv_weight_name)
                Wdt = model.get_tensor_datatype(conv_weight_name)
                assert W is not None, "Initializer for conv weights is not set."
                consumer = model.find_consumer(n.output[0])
                if consumer is not None and consumer.op_type == "Mul":
                    mul_weight_name = consumer.input[1]
                    A = model.get_initializer(mul_weight_name)
                    assert A is not None, "Initializer for mul weights is not set."
                    is_1bit = model.get_tensor_datatype(mul_weight_name).bitwidth() == 1
                    is_scalar = np.prod(A.shape) == 1
                    actual_ndims = len(tuple(filter(lambda x: x > 1, A.shape)))
                    is_1d = actual_ndims == 1
                    if is_1bit and (is_1d or is_scalar):
                        # move the mul to the OFM position, since the mul is
                        # applied on the outputs channelwise or as scalar
                        Wnew = A.reshape(-1, 1, 1, 1) * W
                        assert (
                            Wnew.shape == W.shape
                        ), """Shape of new weights is not
                        the same as the shape of the conv weights before."""
                        check_fxn = np.vectorize(lambda x: Wdt.allowed(x))
                        # only absorb if permitted by W datatype
                        if check_fxn(Wnew).all():
                            model.set_initializer(conv_weight_name, Wnew)
                            n.output[0] = consumer.output[0]
                            graph.node.remove(consumer)
                            graph_modified = True
        return (model, graph_modified)


class AbsorbTransposeIntoMultiThreshold(Transformation):
    """For (NCHWTranspose -> MultiThreshold) move Transpose past MultiThreshold
    and set its data_layout mode to NHWC."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Transpose" and not model.is_fork_node(n):
                perms = list(get_by_name(n.attribute, "perm").ints)
                if perms == [0, 3, 1, 2]:
                    mt_cand = model.find_consumer(n.output[0])
                    if (
                        mt_cand is not None
                        and mt_cand.op_type == "MultiThreshold"
                        # and not model.is_fork_node(mt_cand)
                    ):
                        final_t_cands = model.find_consumers(mt_cand.output[0])
                        mt = getCustomOp(mt_cand)
                        mt.set_nodeattr("data_layout", "NHWC")
                        # get rid of first tranpose node
                        mt_cand.input[0] = n.input[0]
                        graph.node.remove(n)
                        # fix output shape for MultiThreshold
                        mt_orig_oshape = model.get_tensor_shape(mt_cand.output[0])
                        mt_ishape = model.get_tensor_shape(mt_cand.input[0])
                        model.set_tensor_shape(mt_cand.output[0], mt_ishape)
                        # re-insert Transpose behind MultiThreshold
                        transpose_output = model.make_new_valueinfo_name()
                        model.set_tensor_shape(transpose_output, mt_orig_oshape)
                        new_transpose = oh.make_node(
                            "Transpose",
                            [mt_cand.output[0]],
                            [transpose_output],
                            perm=[0, 3, 1, 2],
                        )
                        graph.node.insert(node_ind + 1, new_transpose)
                        if final_t_cands is not None:
                            # rewire next nodes' inputs
                            for final_t_cand in final_t_cands:
                                final_t_cand.input[0] = transpose_output
                        else:
                            # replace graph top-level output
                            get_by_name(
                                model.graph.output, mt_cand.output[0]
                            ).name = transpose_output
                            model.set_tensor_shape(mt_cand.output[0], mt_ishape)
                        graph_modified = True
        if graph_modified:
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class AbsorbTransposeIntoFlatten(Transformation):
    """Absorb transpose node into succeeding flatten node, if H=W=1 and the first
    dimension stays the same. Can also be applied if flatten is implemented implicitly
    by a reshape node with shape [1, -1] and the first input dimension is 1"""

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        node_ind = 0
        for n in graph.node:
            node_ind += 1
            if (
                n.op_type == "Reshape"
                and (model.get_initializer(n.input[1]) == [1, -1]).all()
            ) or n.op_type == "Flatten":
                prod = model.find_producer(n.input[0])
                if (
                    prod is not None
                    and prod.op_type == "Transpose"
                    # we ensure that the first dimension is not changed from the
                    # transpose operation
                    and get_by_name(prod.attribute, "perm").ints[0] == 0
                ):
                    data_layout = model.get_tensor_layout(prod.input[0])
                    # check for the data layout to interpret input shape correctly
                    if data_layout is None:
                        warnings.warn(
                            """Data layout for input tensor of Transpose node is not set.
                                To use AbsorbTransposeIntoFlatten transformation
                                please set tensor data layout."""
                        )
                        continue
                    elif data_layout == DataLayout.NCHW:
                        (b, c, h, w) = model.get_tensor_shape(prod.input[0])
                        # if h=w=1 the transposition can be absorbed, otherwise
                        # the absorption would lead to an error in the behavior
                        if h != 1 or w != 1:
                            continue
                        # the flatten node from onnx keeps by default the first
                        # dim and flattens the rest, that is why this transformation
                        # can only work with b != 1 if the model contains already a
                        # flatten node and not a reshape node with shape = [1, -1].
                        # If the first  dim of the input tensor is not 1, flatten and
                        # reshape (with shape = [1, -1]) would lead to different results
                        if n.op_type == "Reshape" and b != 1:
                            continue
                    elif data_layout == DataLayout.NHWC:
                        (b, h, w, c) = model.get_tensor_shape(prod.input[0])
                        if h != 1 or w != 1:
                            continue
                        if n.op_type == "Reshape" and b != 1:
                            continue
                    # create single flatten node and remove obsolete nodes
                    node = oh.make_node("Flatten", [prod.input[0]], [n.output[0]])
                    graph.node.remove(n)
                    graph.node.remove(prod)
                    graph.node.insert(node_ind, node)
                    graph_modified = True
        if graph_modified:
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class AbsorbScalarMulAddIntoTopK(Transformation):
    """Remove mul/add node prior to topk node if the op is scalar. Note that
    the TopK output probabilities will change, but the indices won't."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "TopK":
                prod = model.find_producer(n.input[0])
                if prod is not None and (prod.op_type in ["Mul", "Add"]):
                    prod_input = prod.input[0]
                    param_name = prod.input[1]
                    A = model.get_initializer(param_name)
                    if A is None:
                        warnings.warn("Param is not constant, skipping")
                        continue
                    is_scalar = all(x == 1 for x in A.shape)
                    is_scalar_pos_mul = is_scalar and (prod.op_type == "Mul") and A > 0
                    is_scalar_add = is_scalar and (prod.op_type == "Add")
                    if is_scalar_pos_mul or is_scalar_add:
                        # if the mul is scalar and positive, we can just delete the
                        # mul node and rewire the top k node. Because the top k node
                        # works with probabilities and their relation to each other
                        # the relation doesn't change if every value is multiplied
                        # with a scalar
                        graph.node.remove(prod)
                        n.input[0] = prod_input
                        # to avoid error the dataype is set to float32
                        model.set_tensor_datatype(n.input[0], DataType.FLOAT32)
                        graph_modified = True
        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class AbsorbConsecutiveTransposes(Transformation):
    """Remove (Transpose -> Transpose) patterns when the input and output
    of the pattern have the same layout."""

    def Are_opposite_permutations(self, perms1, perms2):
        if len(perms1) != len(perms2):
            return False
        assert 0 <= max(perms2) < len(perms2), "invalid permutation"
        assert 0 <= max(perms1) < len(perms1), "invalid permutation"

        for i, p in enumerate(perms2):
            if perms1[p] != i:
                return False

        return True

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        for n in graph.node:
            if n.op_type == "Transpose":
                if model.is_fork_node(n):
                    next_nodes = model.find_direct_successors(n)
                    perms1 = list(get_by_name(n.attribute, "perm").ints)

                    # check if all nodes after fork are opposite transposes
                    all_opposite_transposes = True
                    for next_node in next_nodes:
                        if next_node is not None and next_node.op_type == "Transpose":
                            perms2 = list(get_by_name(next_node.attribute, "perm").ints)
                            if not self.Are_opposite_permutations(perms1, perms2):
                                all_opposite_transposes = False
                                break
                        else:
                            all_opposite_transposes = False
                            break

                    if not all_opposite_transposes:
                        continue

                    prod = model.find_producer(n.input[0])
                    for next_node in next_nodes:
                        # connect next_node's consumer input to n's producer output
                        # TODO implement this to allow for forks as producers and
                        # joins as consumers
                        cons = model.find_consumer(next_node.output[0])
                        cons.input[0] = prod.output[0]

                        # remove consumer transpose
                        graph.node.remove(next_node)

                    # remove producer transpose
                    graph.node.remove(n)
                    graph_modified = True

                else:
                    next_node = model.find_consumer(n.output[0])
                    if next_node is not None and next_node.op_type == "Transpose":
                        perms1 = list(get_by_name(n.attribute, "perm").ints)
                        perms2 = list(get_by_name(next_node.attribute, "perm").ints)
                        if self.Are_opposite_permutations(perms1, perms2):

                            # connect next_node's consumer input to n's producer output
                            # TODO implement this to allow for forks as producers
                            consumers = model.find_direct_successors(next_node)
                            prod = model.find_producer(n.input[0])
                            for cons in consumers:
                                for cons_in in cons.input:
                                    if cons_in == next_node.output[0]:
                                        prod.output[0] = cons_in
                                        break
                            # remove both transposes
                            graph.node.remove(n)
                            graph.node.remove(next_node)

                            graph_modified = True
        if graph_modified:
            model = model.transform(InferDataTypes())
        return (model, graph_modified)
