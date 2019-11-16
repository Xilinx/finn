import numpy as np
from onnx import TensorProto
from onnx import helper as oh

from finn.core.datatype import DataType
from finn.core.utils import get_by_name
from finn.transformation import Transformation
from finn.transformation.infer_shapes import InferShapes


class ConvertBipolarMatMulToXnorPopcount(Transformation):
    """Convert MatMul nodes with all-bipolar inputs to XnorPopcountMatMul
    and associated result correction."""

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
                i_bp = model.get_tensor_datatype(mm_input) == DataType.BIPOLAR
                w_bp = model.get_tensor_datatype(mm_weight) == DataType.BIPOLAR
                if i_bp and w_bp:
                    graph_modified = True
                    # change node type and domain
                    n.op_type = "XnorPopcountMatMul"
                    n.domain = "finn"
                    # convert weights into binary (-1,+1) -> (0,1)
                    Wbin = (model.get_initializer(mm_weight) + 1) / 2
                    # extract vector length (common matrix dim)
                    K = Wbin.shape[0]
                    model.set_initializer(mm_weight, Wbin)
                    model.set_tensor_datatype(mm_weight, DataType.BINARY)
                    # find producing threshold node and adjust output to binary
                    mt = model.find_producer(mm_input)
                    if mt is not None and mt.op_type == "MultiThreshold":
                        bin_dt_attr = "BINARY".encode("utf-8")
                        get_by_name(mt.attribute, "out_dtype").s = bin_dt_attr
                        get_by_name(mt.attribute, "out_scale").f = 1.0
                        get_by_name(mt.attribute, "out_bias").f = 0
                        model.set_tensor_datatype(mm_input, DataType.BINARY)
                    else:
                        raise Exception(
                            """Requires Bipolar2Binary, not yet
                                        implemented."""
                        )
                    # make new output node with correct shape
                    mm_out_shape = model.get_tensor_shape(mm_output)
                    xnorpcout = oh.make_tensor_value_info(
                        model.make_new_valueinfo_name(), TensorProto.FLOAT, mm_out_shape
                    )
                    n.output[0] = xnorpcout.name
                    model.set_tensor_datatype(xnorpcout.name, DataType.UINT32)
                    # add mul-add nodes to produce correct dot product result
                    # need to derive P-N from P and K = P+N
                    # so we need 2*P-K
                    A = np.asarray([2.0], dtype=np.float32)
                    B = np.asarray([-K], dtype=np.float32)
                    # create value_info and initializers for Mul and Add constants
                    mul_const = oh.make_tensor_value_info(
                        model.make_new_valueinfo_name(), TensorProto.FLOAT, A.shape
                    )
                    graph.value_info.append(mul_const)
                    model.set_initializer(mul_const.name, A)
                    mul_output = oh.make_tensor_value_info(
                        model.make_new_valueinfo_name(), TensorProto.FLOAT, mm_out_shape
                    )
                    graph.value_info.append(mul_output)
                    add_const = oh.make_tensor_value_info(
                        model.make_new_valueinfo_name(), TensorProto.FLOAT, B.shape
                    )
                    graph.value_info.append(add_const)
                    model.set_initializer(add_const.name, B)
                    # create Mul and Add nodes to replace the batchnorm
                    mul_node = oh.make_node(
                        "Mul", [xnorpcout.name, mul_const.name], [mul_output.name]
                    )
                    add_node = oh.make_node(
                        "Add", [mul_output.name, add_const.name], [mm_output]
                    )
                    # insert where the batchnorm is to preserve topological ordering
                    graph.node.insert(node_ind, mul_node)
                    graph.node.insert(node_ind + 1, add_node)
        model = model.transform(InferShapes())
        return (model, graph_modified)


class ConvertSignToThres(Transformation):
    """Convert Sign node instances to MultiThreshold with threshold at 0."""

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        node_ind = 0
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Sign":
                sign_in_name = n.input[0]
                sign_out_name = n.output[0]
                # find consumer
                consumer = model.find_consumer(sign_out_name)
                assert consumer is not None
                # create thresholds
                thres_param_name = model.make_new_valueinfo_name()
                thres_param = np.asarray([[0]], dtype=np.float32)
                model.set_initializer(thres_param_name, thres_param)
                # create a new node
                mt_node = oh.make_node(
                    "MultiThreshold",
                    [sign_in_name, thres_param_name],
                    [sign_out_name],
                    domain="finn",
                    out_scale=2.0,
                    out_bias=-1.0,
                    out_dtype="BIPOLAR",
                )
                # remove old node, add new node to graph at correct position
                graph.node.insert(node_ind, mt_node)
                graph.node.remove(n)
                # add quantization annotations
                model.set_tensor_datatype(sign_out_name, DataType.BIPOLAR)
                graph_modified = True
        return (model, graph_modified)


class CollapseRepeatedOp(Transformation):
    """Collapse repeated consecutive operations with constant parameters into
    a single operation. make_collapsed_param_fxn must take two tensors and
    return a tensor which gives the equivalent result using a single op. """

    def __init__(self, op_name, make_collapsed_param_fxn):
        super().__init__()
        self.op_name = op_name
        self.make_collapsed_param_fxn = make_collapsed_param_fxn

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == self.op_name:
                consumer = model.find_consumer(n.output[0])
                if consumer is not None and consumer.op_type == self.op_name:
                    op0_param_name = n.input[1]
                    op1_param_name = consumer.input[1]
                    op0_param = model.get_initializer(op0_param_name)
                    op1_param = model.get_initializer(op1_param_name)
                    assert op0_param is not None
                    assert op1_param is not None
                    start_name = n.input[0]
                    end_name = consumer.output[0]
                    # compute the new parameter
                    new_param = self.make_collapsed_param_fxn(op0_param, op1_param)
                    # make and insert new node
                    new_node_param_name = op0_param_name
                    new_node = oh.make_node(
                        self.op_name, [start_name, new_node_param_name], [end_name]
                    )
                    graph.node.insert(node_ind, new_node)
                    # replace parameter value
                    model.set_initializer(new_node_param_name, new_param)
                    # remove old nodes
                    graph.node.remove(n)
                    graph.node.remove(consumer)
                    graph_modified = True
        model = model.transform(InferShapes())
        return (model, graph_modified)


class CollapseRepeatedAdd(CollapseRepeatedOp):
    def __init__(self):
        super().__init__("Add", lambda x, y: y + x)


class CollapseRepeatedMul(CollapseRepeatedOp):
    def __init__(self):
        super().__init__("Mul", lambda x, y: y * x)


class MoveAddPastMul(Transformation):
    """Move add operations past multiply operations. The aim is to have them
    next to each other such that they can be collapsed into a single add."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Add":
                consumer = model.find_consumer(n.output[0])
                if consumer is not None and consumer.op_type == "Mul":
                    # have: (x) -> add(,B) -> (x+B) -> mul(,A) -> (xA+BA)
                    # want: (x) -> mul(,A) -> (xA) -> add(,BA) -> (xA+BA)
                    # assume input 0 is from the previous layer, input 1 is the
                    # trained (constant) parameter
                    mul_weight_name = consumer.input[1]
                    add_weight_name = n.input[1]
                    A = model.get_initializer(mul_weight_name)
                    B = model.get_initializer(add_weight_name)
                    assert A is not None
                    assert B is not None
                    start_name = n.input[0]
                    middle_name = n.output[0]
                    end_name = consumer.output[0]
                    # compute new param value for add
                    BA = B * A
                    # make and insert new nodes
                    new_mul = oh.make_node(
                        "Mul", [start_name, mul_weight_name], [middle_name]
                    )
                    new_add = oh.make_node(
                        "Add", [middle_name, add_weight_name], [end_name]
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
            if n.op_type == "Mul":
                consumer = model.find_consumer(n.output[0])
                if consumer is not None and consumer.op_type == "MatMul":
                    mul_weight_name = n.input[1]
                    matmul_weight_name = consumer.input[1]
                    A = model.get_initializer(mul_weight_name)
                    W = model.get_initializer(matmul_weight_name)
                    assert A is not None
                    assert W is not None
                    start_name = n.input[0]
                    middle_name = n.output[0]
                    end_name = consumer.output[0]
                    mm_out_shape = model.get_tensor_shape(end_name)
                    if all(x == 1 for x in A.shape):
                        # if the mul is scalar, we can simply swap the order of ops
                        # make and insert new nodes
                        new_matmul = oh.make_node(
                            "MatMul", [start_name, matmul_weight_name], [middle_name]
                        )
                        new_mul = oh.make_node(
                            "Mul", [middle_name, mul_weight_name], [end_name]
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
            if n.op_type == "Add":
                consumer = model.find_consumer(n.output[0])
                if consumer is not None and consumer.op_type == "MatMul":
                    add_weight_name = n.input[1]
                    matmul_weight_name = consumer.input[1]
                    A = model.get_initializer(add_weight_name)
                    W = model.get_initializer(matmul_weight_name)
                    assert A is not None
                    assert W is not None
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
                            "MatMul", [start_name, matmul_weight_name], [middle_name]
                        )
                        new_add = oh.make_node(
                            "Add", [middle_name, add_weight_name], [end_name]
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


class AbsorbAddIntoMultiThreshold(Transformation):
    """Absorb preceding Add ops into MultiThreshold by updating the threshold
    values."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Add":
                consumer = model.find_consumer(n.output[0])
                if consumer is not None and consumer.op_type == "MultiThreshold":
                    add_weight_name = n.input[1]
                    threshold_name = consumer.input[1]
                    A = model.get_initializer(add_weight_name)
                    T = model.get_initializer(threshold_name)
                    assert A is not None
                    assert T is not None
                    start_name = n.input[0]
                    # compute new thresholds and set initializer
                    Tnew = T - A.reshape(-1, T.shape[1])
                    model.set_initializer(threshold_name, Tnew)
                    # wire add input directly to MultiThreshold
                    consumer.input[0] = start_name
                    # remove the add node
                    graph.node.remove(n)
                    graph_modified = True
        return (model, graph_modified)


class AbsorbMulIntoMultiThreshold(Transformation):
    """Absorb preceding Mul ops into MultiThreshold by updating the threshold
    values. Only *positive* scalar/1D vectors can be absorbed."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Mul":
                mul_weight_name = n.input[1]
                A = model.get_initializer(mul_weight_name)
                assert A is not None
                is_signed = (A < 0).any()
                is_scalar = np.prod(A.shape) == 1
                is_1d = len(A.shape) == 2 and A.shape[0] == 1
                consumer = model.find_consumer(n.output[0])
                if consumer is not None and consumer.op_type == "MultiThreshold":
                    if not is_signed and (is_1d or is_scalar):
                        threshold_name = consumer.input[1]
                        T = model.get_initializer(threshold_name)
                        assert T is not None
                        start_name = n.input[0]
                        # compute new thresholds and set initializer
                        Tnew = T / A.reshape(-1, T.shape[1])
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
                assert A is not None
                is_scalar = np.prod(A.shape) == 1
                is_1d = len(A.shape) == 2 and A.shape[0] == 1
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
                assert W is not None
                consumer = model.find_consumer(n.output[0])
                if consumer is not None and consumer.op_type == "Mul":
                    mul_weight_name = consumer.input[1]
                    A = model.get_initializer(mul_weight_name)
                    assert A is not None
                    is_1bit = model.get_tensor_datatype(mul_weight_name).bitwidth() == 1
                    if is_1bit:
                        Wnew = A * W
                        assert Wnew.shape == W.shape
                        model.set_initializer(matmul_weight_name, Wnew)
                        n.output[0] = consumer.output[0]
                        graph.node.remove(consumer)
                        graph_modified = True
        return (model, graph_modified)


class RoundThresholds(Transformation):
    """For MultiThreshold nodes operating on integer inputs, round up
    thresholds values to the nearest integer."""

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        for n in graph.node:
            if n.op_type == "MultiThreshold":
                idtype = model.get_tensor_datatype(n.input[0])
                T = model.get_initializer(n.input[1])
                Tnew = np.ceil(T)
                if idtype.is_integer() and (T != Tnew).any():
                    # round up the thresholds to nearest integer
                    model.set_initializer(n.input[1], Tnew)
                    # use same datatype as inputs for thresholds
                    model.set_tensor_datatype(n.input[1], idtype)
                    graph_modified = True
        return (model, graph_modified)
