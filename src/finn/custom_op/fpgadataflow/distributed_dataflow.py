import os
import subprocess
import psutil

from concurrent.futures import ThreadPoolExecutor, as_completed

import qonnx.analysis.topology as ta
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_node
from qonnx.custom_op.base import CustomOp
from qonnx.util.basic import (
    get_sanitize_quant_tensors,
    sanitize_quant_values,
)


def execute_distributed_onnx(
    model,
    input_dict,
    return_full_exec_context=False,
    start_node=None,
    end_node=None
):
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

    # check if model has an execution mode set
    # if None, execute model node by node using execute_node()
    model_exec_mode = model.get_metadata_prop("exec_mode")

    if (model_exec_mode is None) or (model_exec_mode == ""):
        # extract opset version for node-by-node execution
        opset_version = model.model.opset_import[0].version
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
                execution_context = sanitize_quant_values(
                    model, node.input, execution_context)

        with ThreadPoolExecutor() as executor:
            futures = []
            for node in subgraph:
                futures.append(executor.submit(
                    execute_node,
                    node,
                    execution_context,
                    graph,
                    return_full_exec_context,
                    opset_version
                ))

        for future in as_completed(futures):
            future.result()

        for node in subgraph:
            if get_sanitize_quant_tensors() != 0:
                # round output values to quantization annotation
                execution_context = sanitize_quant_values(
                    model, node.output, execution_context)
    else:
        raise Exception(
            'Metadata property "exec_mode" is set to an unknown value.')

    if return_full_exec_context:
        return execution_context
    else:
        # provide outputs as dict
        output_dict = dict()
        for out_tensor in graph.output:
            out_name = out_tensor.name
            output_dict[out_name] = execution_context[out_name]
        return output_dict


class DistributedDataflow(CustomOp):
    def get_nodeattr_types(self):
        return {
            "model": ("s", True, ""),
            "instance_name": ("s", False, ""),
            "return_full_exec_context": ("i", False, 0),
            "world_size": ("i", True, -1),
        }

    def make_shape_compatible_op(self, model):
        pass

    def infer_node_datatype(self, model):
        pass

    def verify_node(self):
        return []

    def execute_node(self, context, graph):
        model = ModelWrapper(self.get_nodeattr("model"))
        return_full_exec_context = self.get_nodeattr(
            "return_full_exec_context") == 1
        node = self.onnx_node

        inp_ctx = dict(filter(lambda x: x[0] in node.input, context.items()))
        for i, old_iname in enumerate(node.input):
            new_iname = model.graph.input[i].name
            if old_iname != new_iname:
                inp_ctx[new_iname] = inp_ctx[old_iname]
                del inp_ctx[old_iname]

        emulator_dir = f"{os.environ['FINN_ROOT']}/ACCL/test/model/emulator"

        subprocess.run(["/usr/bin/cmake", "."],
                       cwd=emulator_dir, stdout=subprocess.PIPE)
        emulator = subprocess.Popen(
            ["python3", "run.py", "-n 2", "--no-kernel-loopback", "-l 1"], cwd=emulator_dir)

        ret = execute_distributed_onnx(
            model, inp_ctx, return_full_exec_context)

        parent_proc = psutil.Process(emulator.pid)
        for child in parent_proc.children(recursive=True):
            child.kill()
        emulator.kill()

        for i, node_oname in enumerate(node.output):
            model_oname = model.graph.output[i].name
            context[node_oname] = ret[model_oname]

        if return_full_exec_context:
            for tname in ret.keys():
                if tname not in [x.name for x in model.graph.output]:
                    context[node.name + "_" + tname] = ret[tname]
