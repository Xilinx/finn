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

from finn.custom_op.registry import getCustomOp
from finn.util.fpgadataflow import is_fpgadataflow_node


def res_estimation(model):
    """Estimates the resources needed for the given model.
    Ensure that all nodes have unique names (by calling the GiveUniqueNodeNames
    transformation) prior to calling this analysis pass to ensure all nodes are
    visible in the results.

    Returns {node name : resource estimation}."""

    res_dict = {}
    for node in model.graph.node:
        if is_fpgadataflow_node(node) is True:
            inst = getCustomOp(node)
            res_dict[node.name] = inst.node_res_estimation()

    return res_dict


def res_estimation_complete(model):
    """Estimates the resources needed for the given model and all values for
    resource-related switches.
    Ensure that all nodes have unique names (by calling the GiveUniqueNodeNames
    transformation) prior to calling this analysis pass to ensure all nodes are
    visible in the results.

    Returns {node name : [resource estimation(s)]}."""

    res_dict = {}
    for node in model.graph.node:
        if is_fpgadataflow_node(node) is True:
            op_type = node.op_type
            inst = getCustomOp(node)
            if op_type == "StreamingFCLayer_Batch" or op_type == "Vector_Vector_Activate_Batch":
                orig_restype = inst.get_nodeattr("resType")
                res_dict[node.name] = []
                inst.set_nodeattr("resType", "dsp")
                res_dict[node.name].append(inst.node_res_estimation())
                inst.set_nodeattr("resType", "lut")
                res_dict[node.name].append(inst.node_res_estimation())
                inst.set_nodeattr("resType", orig_restype)
            elif op_type == "ConvolutionInputGenerator":
                orig_ramstyle = inst.get_nodeattr("ram_style")
                res_dict[node.name] = []
                inst.set_nodeattr("ram_style", "block")
                res_dict[node.name].append(inst.node_res_estimation())
                inst.set_nodeattr("ram_style", "distributed")
                res_dict[node.name].append(inst.node_res_estimation())
                inst.set_nodeattr("ram_style", "ultra")
                res_dict[node.name].append(inst.node_res_estimation())
                inst.set_nodeattr("ram_style", orig_ramstyle)
            else:
                res_dict[node.name] = [inst.node_res_estimation()]

    return res_dict
