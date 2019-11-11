# import onnx.helper as helper
import sys
import finn.core.multithreshold as multiThresh
import finn.core.utils as util


def execute_custom_node(node, context, graph):
    """Call custom implementation to execute a single custom node.
    Input/output provided via context."""

    if (util.get_by_name(node.attribute, 'backend')) is not None:
        if node.op_type == "StreamingMaxPool":
            sys.exit(0)
        else:
            # exception if op_type is not supported
            raise Exception("This hls lib custom node is currently not supported.")

    
    else:
        if node.op_type == "MultiThreshold":
            # save inputs
            v = context[node.input[0]]
            thresholds = context[node.input[1]]
            # calculate output
            output = multiThresh.execute(v, thresholds)
            # setting context according to output
            context[node.output[0]] = output

        else:
            # exception if op_type is not supported
            raise Exception("This custom node is currently not supported.")
