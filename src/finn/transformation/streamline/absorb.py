import numpy as np
from onnx import helper as oh

from finn.core.datatype import DataType
from finn.transformation import Transformation


class AbsorbAddIntoMultiThreshold(Transformation):
    """Absorb preceding Add ops into MultiThreshold by updating the threshold
    values. Only scalar/1D add vectors can be absorbed."""

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
                    assert A is not None, "Initializer for add weights is not set."
                    assert T is not None, "Initializer for thresholds is not set."
                    start_name = n.input[0]
                    # we can only absorb 0d or 1d adds
                    is_scalar = A.ndim == 0 or all(x == 1 for x in A.shape)
                    is_1d = A.ndim > 0 and np.prod(A.shape) == A.shape[-1]
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
            if n.op_type == "Mul":
                mul_weight_name = n.input[1]
                A = model.get_initializer(mul_weight_name)
                assert A is not None, "Initializer for mul weights is not set."
                is_signed = (A < 0).any()
                is_scalar = A.ndim == 0 or all(x == 1 for x in A.shape)
                is_1d = A.ndim > 0 and np.prod(A.shape) == A.shape[-1]
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
                assert W is not None, "Initializer for matmul weights is not set."
                consumer = model.find_consumer(n.output[0])
                if consumer is not None and consumer.op_type == "Mul":
                    mul_weight_name = consumer.input[1]
                    A = model.get_initializer(mul_weight_name)
                    assert A is not None, "Initializer for mul weights is not set."
                    is_1bit = model.get_tensor_datatype(mul_weight_name).bitwidth() == 1
                    if is_1bit:
                        Wnew = A * W
                        assert Wnew.shape == W.shape, """Shape of new weights is not
                        the same as the shape of the weight matrix before."""
                        model.set_initializer(matmul_weight_name, Wnew)
                        n.output[0] = consumer.output[0]
                        graph.node.remove(consumer)
                        graph_modified = True
        return (model, graph_modified)
