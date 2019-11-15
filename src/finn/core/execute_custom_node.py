# import onnx.helper as helper

import finn.core.multithreshold as multiThresh
from finn.core.utils import get_by_name


def execute_custom_node(node, context, graph):
    """Call custom implementation to execute a single custom node.
    Input/output provided via context."""

    if node.op_type == "MultiThreshold":
        # save inputs
        v = context[node.input[0]]
        thresholds = context[node.input[1]]
        # retrieve attributes if output scaling is used
        try:
            out_scale = get_by_name(node.attribute, "out_scale").f
        except AttributeError:
            out_scale = None
        try:
            out_bias = get_by_name(node.attribute, "out_bias").f
        except AttributeError:
            out_bias = None
        # calculate output
        output = multiThresh.execute(v, thresholds, out_scale, out_bias)
        # setting context according to output
        context[node.output[0]] = output

    else:
        # exception if op_type is not supported
        raise Exception("This custom node is currently not supported.")
