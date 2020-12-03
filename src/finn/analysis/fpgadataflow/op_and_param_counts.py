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

import finn.custom_op.registry as registry
from finn.util.basic import is_finn_op


def aggregate_dict_keys(res_dict):
    total_dict = {}
    for layer in res_dict:
        layer_res_dict = res_dict[layer]
        for r_type in layer_res_dict.keys():
            if "efficiency" in r_type:
                continue
            r_amount = layer_res_dict[r_type]
            r_amount = float(r_amount)
            if r_type in total_dict.keys():
                total_dict[r_type] += r_amount
            else:
                total_dict[r_type] = r_amount
    return total_dict


def op_and_param_counts(model):
    """Return per-node and aggregate op counts per inference."""

    ret_dict = {}
    for node in model.graph.node:
        if is_finn_op(node.domain):
            inst = registry.getCustomOp(node)
            if hasattr(inst, "get_op_and_param_counts"):
                node_op_and_param_counts = inst.get_op_and_param_counts()
                ret_dict[node.name] = node_op_and_param_counts
    ret_dict["total"] = aggregate_dict_keys(ret_dict)
    return ret_dict
