# Copyright (c) 2019, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    1. Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#    3. Neither the name of the <organization> nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import copy

import onnx.helper as helper
import onnxruntime as rt

import finn.core.execute_custom_node as ex_cu_node


def execute_node(node, context, graph):
    """Call onnxruntime to execute a single node. Input/output provided via context."""

    # run node with custom function or by using onnxruntime

    if node.domain == "finn":

        ex_cu_node.execute_custom_node(node, context, graph)

    else:

        # onnxruntime unfortunately does not implement run_node as defined by ONNX,
        # it can only execute entire models -- so we create a model which solely
        # consists of our current node.
        node_inputs = list(filter(lambda x: x.name in node.input, graph.input))
        node_inputs += list(filter(lambda x: x.name in node.input, graph.value_info))
        node_outputs = list(filter(lambda x: x.name in node.output, graph.output))
        node_outputs += list(filter(lambda x: x.name in node.output, graph.value_info))
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
            outp = node.output[output_ind]
            if output_list[output_ind].shape != context[outp].shape:
                raise Exception(
                    """Output shapes disagree after node execution:
                    found %s vs expected %s"""
                    % (
                        str(output_list[output_ind].shape.shape),
                        str(context[outp].shape),
                    )
                )
            context[outp] = output_list[output_ind]


def execute_onnx(model, input_dict, return_full_exec_context=False):
    """Execute given ONNX ModelWrapper with given named inputs.
    If return_full_exec_context is False, a dict of named outputs is returned
    as indicated by the model.graph.output.
    If return return_full_exec_context is True, the full set of tensors used by
    the execution (including inputs, weights, activations and final outputs)
    will be returned as a dict."""

    if not model.check_all_tensor_shapes_specified():
        raise Exception("Found unspecified tensor shapes, try infer_shapes")

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
        else:
            raise Exception("Provided input not found in graph context: %s" % inp_name)
    # now call each node in the graph nodes list
    # we can simply walk down the list since the ONNX spec guarantees that it is
    # topologically sorted
    for node in graph.node:
        execute_node(node, execution_context, graph)
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
    """Execute given ONNX ModelWrapper with given named inputs and return a new
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
