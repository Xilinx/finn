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
from finn.transformation import NodeLocalTransformation


class SetMemMode(NodeLocalTransformation):
    """Set attribute mem_mode in all FC layer nodes to specify which
    kind of weight storage to use, based on memory depth. Use simple rules:
    -memories below a min_threshold are set to const and ram_style distributed
    -memories above a max_threshold are set to external
    -everything else is set to decoupled and ram_style block"""

    def __init__(self, min_threshold=128, max_threshold=None):
        super().__init__()
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def applyNodeLocal(self, node):
        op_type = node.op_type
        if op_type == "StreamingFCLayer_Batch":
            node_inst = getCustomOp(node)
            wmem = node_inst.calc_wmem()
            if wmem <= self.min_threshold:
                node_inst.set_nodeattr("mem_mode", "const")
            else:
                node_inst.set_nodeattr("mem_mode", "decoupled")
                node_inst.set_nodeattr("ram_style", "block")
            # set to external if upper threshold exists
            if self.max_threshold is not None:
                if wmem > self.max_threshold:
                    node_inst.set_nodeattr("mem_mode", "external")

        return (node, False)
