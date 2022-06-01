# Copyright (c) 2022, Advanced Micro Devices, Inc.
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

import qonnx.analysis.topology as ta
from qonnx.core.onnx_exec import execute_onnx as execute_onnx_base

from finn.core.remote_exec import remote_exec
from finn.core.rtlsim_exec import rtlsim_exec


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

    # check if model has an execution mode set
    # if None, execute model node using the QONNX-provided execute_onnx impl
    # if set to "remote_pynq" execute model on PYNQ board
    # if set to "rtlsim" execute model using pyverilator
    model_exec_mode = model.get_metadata_prop("exec_mode")
    if (model_exec_mode is None) or (model_exec_mode == ""):
        return execute_onnx_base(
            model, input_dict, return_full_exec_context, start_node, end_node
        )

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
        return execute_onnx_base()
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
