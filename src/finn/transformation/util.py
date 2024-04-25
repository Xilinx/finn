# fmt: off
# Disable formatter. This is deliberately formatted to stay within 80 characters
# per line. Black, however, formats some lines going beyond this.

# Protobuf onnx graph node type
from onnx import NodeProto
# QONNX wrapper of ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper


# Tests whether a node is a join-node MatMul operation, i.e., a MatMul with two
# runtime inputs but no weights initializers
def is_join_matmul(node: NodeProto, model: ModelWrapper):  # noqa
    # Only handle existing MatMul type nodes
    if node is not None and node.op_type in {"MatMul"}:
        # No input must have an initializer
        return all(model.get_initializer(i) is None for i in node.input)
    # Did not match the operator type
    return False


# Tests whether a node is a MatMul operator
def is_matmul(node: NodeProto):
    # Node must exist and be of type MatMul
    return node is not None and node.op_type in {"MatMul"}


# Tests whether a node is a Softmax operator
def is_softmax(node: NodeProto):
    # Node must exist and be of type Softmax
    return node is not None and node.op_type in {"Softmax"}


# Tests whether a node is an element-wise Mul
def is_mul(node: NodeProto):
    # Node must exist and be of type Mul
    return node is not None and node.op_type in {"Mul"}


# Tests whether a node is an element-wise Add
def is_add(node: NodeProto):
    # Node must exist and be of type Add
    return node is not None and node.op_type in {"Add"}


def is_end(node: NodeProto, model: ModelWrapper):  # noqa
    return node is not None and not model.find_direct_predecessors(node)


# Follow all input branches of a node until reaching a matmul
def all_upstream_to_matmul(node: NodeProto, model: ModelWrapper):  # noqa
    # Check whether the node is either a matmul node or the end of the graph
    def is_matmul_or_end(n: NodeProto):
        return is_matmul(n) or is_end(n, model)

    # Enumerate all inputs and collect everything upstream until finding the
    # next matmul operation
    return (model.find_upstream(i, is_matmul_or_end, True) for i in node.input)


# Projects a list of ONNX graph nodes to the string representation of the
# operator types
def op_types(nodes: list[NodeProto]) -> list[str]:
    return [node.op_type if node is not None else "None" for node in nodes]


# Tests whether a node is a Reshape operator
def is_reshape(node: NodeProto):
    return node is not None and node.op_type in {"Reshape"}


# Tests whether a node is a Transpose operator
def is_transpose(node: NodeProto):
    return node is not None and node.op_type in {"Transpose"}


# Tests whether a node is a Reshape-Transpose operator chain
def is_reshape_transpose(node: NodeProto, model: ModelWrapper):  # noqa
    # Reshape-transpose pattern detection is triggered by detecting a reshape
    # operation
    if is_reshape(node):
        # The reshape may not be a join or fork node
        if model.is_join_node(node) or model.is_fork_node(node):
            # Reject detection of the pattern
            return False
        # Get the single successor node
        transpose = model.find_direct_successors(node)[0]
        # The consumer must be Transpose finalizing the reshaping
        if not is_transpose(transpose):
            # Reject detection of the pattern
            return False
        # The transpose may not fork or join either
        if model.is_join_node(transpose) or model.is_fork_node(transpose):
            # Reject detection of the pattern
            return False
        # Accept detecting the pattern
        return True
    # Reject detection of the pattern
    return False


# Tests whether a node is a Transpose-Reshape operator chain
def is_transpose_reshape(node: NodeProto, model: ModelWrapper):  # noqa
    # Transpose-Reshape pattern detection is triggered by detecting a transpose
    # operation
    if is_transpose(node):
        # The transpose may not be a join or fork node
        if model.is_join_node(node) or model.is_fork_node(node):
            # Reject detection of the pattern
            return False
        # Get the single successor node
        reshape = model.find_direct_successors(node)[0]
        # The consumer must be a reshape finalizing the transpose-reshape
        if not is_reshape(reshape):
            # Reject detection of the pattern
            return False
        # The reshape may not fork or join either
        if model.is_join_node(reshape) or model.is_fork_node(reshape):
            # Reject detection of the pattern
            return False
        # Accept detecting the pattern
        return True
    # Reject detection of the pattern
    return False
