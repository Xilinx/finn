# import onnx.helper as helper
import sys
import os
import tempfile
import numpy as np
import finn.core.multithreshold as multiThresh
import finn.core.utils as util
import finn.backend.fpgadataflow.code_gen_for_single_node_execution as code_gen


def execute_custom_node(node, context, graph):
    """Call custom implementation to execute a single custom node.
    Input/output provided via context."""

    if (util.get_by_name(node.attribute, 'backend')) is not None:
        if node.op_type == "StreamingMaxPool":
            in_ind = 0
            temp_files = []
            for inputs in node.input:
                np.save("input_{}.npy".format(in_ind), context[inputs].astype(np.float32))
                temp_files.append("input_{}.npy".format(in_ind))
                in_ind += 1
            code_gen.execute(node, context, graph)
            
            # deleting temporary files
            for temp_file in temp_files:
                os.remove(temp_file)
            sys.exit(1)
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
