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


import copy

import numpy as np
import onnx.helper as helper
import onnxruntime as rt

import finn.core.execute_custom_node as ex_cu_node
from finn.core.modelwrapper import ModelWrapper
from finn.core.remote_exec import remote_exec
from finn.core.rtlsim_exec import rtlsim_exec
from finn.custom_op.registry import getCustomOp
import finn.analysis.topology as ta
from finn.util.basic import sanitize_quant_values, get_sanitize_quant_tensors


def execute_node(node, context, graph):
    """Executes a single node by using onnxruntime, with custom function or
    if dataflow partition by using remote execution or rtlsim.

    Input/output provided via context."""

    if node.op_type == "StreamingDataflowPartition":
        sdp_node = getCustomOp(node)
        model = ModelWrapper(sdp_node.get_nodeattr("model"))
        inp_ctx = dict(filter(lambda x: x[0] in node.input, context.items()))
        # input may have been renamed in partition
        assert len(inp_ctx) == 1
        old_iname = node.input[0]
        new_iname = model.graph.input[0].name
        if old_iname != new_iname:
            inp_ctx[new_iname] = inp_ctx[old_iname]
            del inp_ctx[old_iname]
        ret = execute_onnx(model, inp_ctx, False)
        # if the model was in ip-stitched rtlsim mode, may get annotation
        # for numbet of elapsed cycles, save again
        if model.get_metadata_prop("exec_mode") == "rtlsim":
            model.save(sdp_node.get_nodeattr("model"))
        # output may have been renamed in partition
        assert len(ret) == 1
        node_oname = node.output[0]
        model_oname = model.graph.output[0].name
        context[node_oname] = ret[model_oname]
    else:
        if node.domain == "finn":

            ex_cu_node.execute_custom_node(node, context, graph)

        else:

            # onnxruntime unfortunately does not implement run_node as defined by ONNX,
            # it can only execute entire models -- so we create a model which solely
            # consists of our current node.
            # note: ensure that the same ValueInfo does not appear both in
            # graph.value_info as well as graph.output or graph.input
            # nodes with multiple outputs that are a mix of value_info and
            # input/outputs may get them reordered below
            node_inputs = list(filter(lambda x: x.name in node.input, graph.input))
            node_inputs += list(
                filter(lambda x: x.name in node.input, graph.value_info)
            )
            node_outputs = list(filter(lambda x: x.name in node.output, graph.output))
            node_outputs += list(
                filter(lambda x: x.name in node.output, graph.value_info)
            )
            node_graph = helper.make_graph(
                nodes=[node],
                name="single-node-exec",
                inputs=node_inputs,
                outputs=node_outputs,
            )
            node_model = helper.make_model(node_graph)
            input_dict = dict()
            for inp in node.input:
                input_dict[inp] = context[inp]

            sess = rt.InferenceSession(node_model.SerializeToString())
            output_list = sess.run(None, input_dict)

            for output_ind in range(len(node.output)):
                # get the name of the target buffer from node.output
                outp = node.output[output_ind]

                # retrieve the index of that name in node_outputs
                for i in range(len(node_outputs)):
                    if outp == node_outputs[i].name:
                        list_ind = i

                # use that index to index output_list
                if output_list[list_ind].shape != context[outp].shape:
                    raise Exception(
                        """Output shapes disagree after node execution:
                        found %s vs expected %s"""
                        % (str(output_list[list_ind].shape), str(context[outp].shape))
                    )
                context[outp] = output_list[list_ind]


def execute_onnx(
    model, input_dict, return_full_exec_context=False, start_node=None, end_node=None
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
        # else:
        # raise Exception("Provided input not found in graph context: %s" % inp_name)

    # check if model has an execution mode set
    # if None, execute model node by node using execute_node()
    # if set to "remote_pynq" execute model on PYNQ board
    # if set to "rtlsim" execute model using pyverilator
    model_exec_mode = model.get_metadata_prop("exec_mode")
    if (model_exec_mode is None) or (model_exec_mode == ""):
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
                    model, node.input, execution_context
                )
            execute_node(node, execution_context, graph)
            if get_sanitize_quant_tensors() != 0:
                # round output values to quantization annotation
                execution_context = sanitize_quant_values(
                    model, node.output, execution_context
                )
    elif model_exec_mode == "remote_pynq":
        # use remote exec metadata built into model to execute on a remote PYNQ
        remote_exec(model, execution_context)
    elif model_exec_mode == "rtlsim":
        # use stitched IP for rtlsim
        rtlsim_exec(model, execution_context)
    else:
        raise Exception(
            """Metadata property "exec_mode" is set to an unknown value.
        Can be left unset or has to be set to "remote_pynq" for remote execution
        on PYNQ board or "rtlsim" for execution using pyverilator!"""
        )

    if return_full_exec_context:
        return execution_context
    else:
        # provide outputs as dict
        output_dict = dict()
        for out_tensor in graph.output:
            out_name = out_tensor.name
            output_dict[out_name] = execution_context[out_name]
        return output_dict


def execute_onnx_and_make_model(model, input_dict):
    """Executes given ONNX ModelWrapper with given named inputs and return a new
    ModelWrapper where an initializer is provided for each tensor as taken from
    the execution. This new model is useful for debugging, since it contains
    all the intermediate activation values."""

    # retrieve the full execution context
    execution_context = execute_onnx(model, input_dict, True)
    new_model = copy.deepcopy(model)
    # create value_info entries and initializers for everything
    for i in execution_context.keys():
        new_model.set_initializer(i, execution_context[i])
    for vi in new_model.graph.value_info:
        new_model.graph.output.append(vi)
    # import pdb; pdb.set_trace()
    return new_model


def compare_execution(
    model_a,
    model_b,
    input_dict,
    compare_fxn=lambda x, y: np.isclose(x, y, atol=1e-3).all(),
):
    """Executes two ONNX models and compare their outputs using given function.

    compare_fxn should take in two tensors and return a Boolean"""
    # compare values from first output tensors produced
    res_a = list(execute_onnx(model_a, input_dict).items())[0][1]
    res_b = list(execute_onnx(model_b, input_dict).items())[0][1]
    return compare_fxn(res_a, res_b)
