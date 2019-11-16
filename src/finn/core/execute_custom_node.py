# import onnx.helper as helper

import finn.custom_op.registry as registry


def execute_custom_node(node, context, graph):
    """Call custom implementation to execute a single custom node.
    Input/output provided via context."""
    op_type = node.op_type
    try:
        # lookup op_type in registry of CustomOps
        inst = registry.custom_op[op_type]()
        inst.execute_node(node, context, graph)
    except KeyError:
        # exception if op_type is not supported
        raise Exception("Custom op_type %s is currently not supported." % op_type)
