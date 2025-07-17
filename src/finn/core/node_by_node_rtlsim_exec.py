import qonnx.analysis.topology as ta
from qonnx.util.basic import (
    get_sanitize_quant_tensors,
    sanitize_quant_values,
)
from pathlib import Path

from finn.util.kernel_util import get_node_attr
from finn.kernels import gkr


def execute_kernel(node, model, context, graph):
    """Call custom implementation to execute a single custom node.
    Input/output provided via context."""
    if gkr.kernel_exists(node.op_type):
        kernel = gkr.kernel(node.op_type, get_node_attr(node, model))
        rtlsim_dir = model.get_metadata_prop("rtlsim_dir")
        kernel_rtlsim_dir = Path(rtlsim_dir) / Path(f"rtlsim_{kernel.name}_")
        kernel.execute_rtlsim(context, graph, kernel_rtlsim_dir, node, model.get_metadata_prop("rtlsim_trace"))
    else:
        raise Exception("Custom op_type %s is currently not supported." % node.op_type)

def node_by_node_rtlsim_exec(model, input_dict, return_full_exec_context=False, start_node=None, end_node=None):
    """Executes given ONNX ModelWrapper with given named inputs.

    If return_full_exec_context is False, a dict of named outputs is returned
    as indicated by the model.graph.output.

    If return return_full_exec_context is True, the full set of tensors used by
    the execution (including inputs, weights, activations and final outputs)
    will be returned as a dict.

    When start_node and end_node are set to None, the whole graph is executed.
    If they are set to particular ONNX nodes, only the subgraph between (and
    including) those nodes is executed.
    """

    if not model.check_all_tensor_shapes_specified():
        raise Exception("Found unspecified tensor shapes, try infer_shapes")
    ret = model.analysis(ta.nodes_topologically_sorted)
    assert (
        ret["nodes_topologically_sorted"] is True
    ), """Nodes must be
    topologically sorted."""

    graph = model.graph
    # first, we need to make sure that every variable required by the graph has
    # some buffer associated with it. this includes graph inputs (which includes
    # the input data as well as the trained parameters) and the graph ValueInfo
    # (intermediate tensors between layers)
    # this is provided by the execution_context, which is a dict of np.ndarray
    execution_context = model.make_empty_exec_context()
    # fill in any inputs provided to this function
    for inp_name in input_dict.keys():
        if inp_name in execution_context:
            if execution_context[inp_name].shape == input_dict[inp_name].shape:
                execution_context[inp_name] = input_dict[inp_name]
            else:
                raise Exception(
                    "Shape mismatch for provided input %s: found %s expected %s "
                    % (
                        inp_name,
                        str(execution_context[inp_name].shape),
                        str(input_dict[inp_name].shape),
                    )
                )
        # else:
        # raise Exception("Provided input not found in graph context: %s" % inp_name)

    # execute the model node by node
    # we can simply walk down the list since the ONNX spec guarantees that it is
    # topologically sorted
    subgraph = []
    if start_node is None:
        start_node = model.graph.node[0]
    if end_node is None:
        end_node = model.graph.node[-1]
    # select the nodes between specified start/end nodes
    start_ind = model.get_node_index(start_node)
    end_ind = model.get_node_index(end_node) + 1
    assert end_ind >= start_ind, "Start/end nodes must define valid subgraph"
    subgraph = graph.node[start_ind:end_ind]
    for node in subgraph:
        if get_sanitize_quant_tensors() != 0:
            # round input values to match quantization annotation
            execution_context = sanitize_quant_values(model, node.input, execution_context)
        execute_kernel(node, model, execution_context, graph)
        if get_sanitize_quant_tensors() != 0:
            # round output values to quantization annotation
            execution_context = sanitize_quant_values(model, node.output, execution_context)

    if return_full_exec_context:
        return execution_context
    else:
        # provide outputs as dict
        output_dict = dict()
        for out_tensor in graph.output:
            out_name = out_tensor.name
            output_dict[out_name] = execution_context[out_name]
        return output_dict
