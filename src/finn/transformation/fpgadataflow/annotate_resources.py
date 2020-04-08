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
from finn.transformation import Transformation
from finn.transformation.move_reshape import _is_fpgadataflow_node
from finn.analysis.fpgadataflow.res_estimation import res_estimation
from finn.analysis.fpgadataflow.hls_synth_res_estimation import hls_synth_res_estimation


class AnnotateResources(Transformation):
    """Annotate the amount of FPGA resources taken by each fpgadataflow
    node as an attribute on the node, depending on the mode parameter:
    * 'estimate' -- use the analytical estimation model
    * 'hls' -- use results from the HLS synthesis report

    No annotations can be provided unless the relevant transformation for the
    chosen mode (e.g. HLSSynth_IPGen for hls) was previously run.
    """

    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def apply(self, model):
        graph = model.graph
        if self.mode == "estimate":
            res_fxn = res_estimation
        elif self.mode == "hls":
            res_fxn = hls_synth_res_estimation
        else:
            raise Exception("Unrecognized mode for AnnotateResources")
        res_dict = model.analysis(res_fxn)
        for node in graph.node:
            if _is_fpgadataflow_node(node) and node.name in res_dict.keys():
                op_inst = registry.getCustomOp(node)
                op_inst.set_nodeattr("res_" + self.mode, str(res_dict[node.name]))

        return (model, False)
