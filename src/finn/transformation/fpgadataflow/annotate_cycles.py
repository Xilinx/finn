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
from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.registry import getCustomOp
from finn.transformation.base import Transformation
from finn.transformation.move_reshape import _is_fpgadataflow_node


class AnnotateCycles(Transformation):
    """Annotate the estimate of clock cycles per sample taken by each fpgadataflow
    node as an attribute on the node.
    """

    def __init__(self):
        super().__init__()

    def apply(self, model):
        graph = model.graph
        # annotate node cycles
        for node in graph.node:
            if _is_fpgadataflow_node(node):
                op_inst = registry.getCustomOp(node)
                cycles = op_inst.get_exp_cycles()
                op_inst.set_nodeattr("cycles_estimate", cycles)
            elif node.op_type == "StreamingDataflowPartition":
                # recurse into model to manually annotate per-layer cycles
                sdp_model_filename = getCustomOp(node).get_nodeattr("model")
                sdp_model = ModelWrapper(sdp_model_filename)
                sdp_model = sdp_model.transform(AnnotateCycles())
                # save transformed model
                sdp_model.save(sdp_model_filename)
        return (model, False)
