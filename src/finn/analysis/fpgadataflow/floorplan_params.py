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

from finn.util.fpgadataflow import is_fpgadataflow_node
from finn.custom_op.registry import getCustomOp


def floorplan_params(model):
    """Gathers SLR and partition IDs from nodes.

    Returns {node name : {slr, device id, partition id, memory port}}."""

    ret_dict = {
        "Defaults": {
            "slr": [-1, ["all"]],
            "partition_id": [0, ["all"]],
            "device_id": [0, ["all"]],
            "mem_port": ["", ["all"]],
        }
    }
    for node in model.graph.node:
        if is_fpgadataflow_node(node) is True:
            node_inst = getCustomOp(node)
            node_slr = node_inst.get_nodeattr("slr")
            node_pid = node_inst.get_nodeattr("partition_id")
            node_mport = node_inst.get_nodeattr("mem_port")
            ret_dict[node.name] = {
                "slr": node_slr,
                "partition_id": node_pid,
                "device_id": 0,
                "mem_port": node_mport,
            }

    return ret_dict
