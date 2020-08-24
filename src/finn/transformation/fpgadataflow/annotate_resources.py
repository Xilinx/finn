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
from finn.analysis.fpgadataflow.post_synth_res import post_synth_res
from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.registry import getCustomOp


class AnnotateResources(Transformation):
    """Annotate the amount of FPGA resources taken by each fpgadataflow
    node as an attribute on the node, depending on the mode parameter:
    * 'estimate' -- use the analytical estimation model
    * 'hls' -- use results from the HLS synthesis report
    * 'synth' -- use post-synthesis (Vivado or Vitis) report

    No annotations can be provided unless the relevant transformation for the
    chosen mode (e.g. HLSSynthIP for hls) was previously run.
    """

    def __init__(self, mode, override_res_dict=None):
        super().__init__()
        self.mode = mode
        self.res_dict = override_res_dict

    def apply(self, model):
        graph = model.graph
        if self.mode == "estimate":
            res_fxn = res_estimation
        elif self.mode == "hls":
            res_fxn = hls_synth_res_estimation
        elif self.mode == "synth":
            res_fxn = post_synth_res
        else:
            raise Exception("Unrecognized mode for AnnotateResources")
        if self.res_dict is None:
            self.res_dict = model.analysis(res_fxn)
        children_dict = {}
        # annotate node resources
        for node in graph.node:
            if _is_fpgadataflow_node(node) and node.name in self.res_dict.keys():
                op_inst = registry.getCustomOp(node)
                op_inst.set_nodeattr("res_" + self.mode, str(self.res_dict[node.name]))
                children_dict[node.name] = self.res_dict[node.name]
            elif node.op_type == "StreamingDataflowPartition":
                # recurse into model to manually annotate per-layer resources
                sdp_model_filename = getCustomOp(node).get_nodeattr("model")
                sdp_model = ModelWrapper(sdp_model_filename)
                sdp_model = sdp_model.transform(
                    AnnotateResources(self.mode, self.res_dict)
                )
                sdp_dict = sdp_model.get_metadata_prop("res_total_" + self.mode)
                sdp_dict = eval(sdp_dict)
                # save transformed model
                sdp_model.save(sdp_model_filename)
                # set res attribute for sdp node
                getCustomOp(node).set_nodeattr("res_" + self.mode, str(sdp_dict))
                children_dict[node.name] = sdp_dict
        self.res_dict.update(children_dict)
        total_dict = {}
        for lname in children_dict.keys():
            layer_res_dict = self.res_dict[lname]
            for r_type in layer_res_dict.keys():
                r_amount = layer_res_dict[r_type]
                r_amount = float(r_amount)
                if r_type in total_dict.keys():
                    total_dict[r_type] += r_amount
                else:
                    total_dict[r_type] = r_amount
        for k in total_dict.keys():
            if "efficiency" in k:
                total_dict[k] = total_dict[k] / len(graph.node)
        model.set_metadata_prop("res_total_" + self.mode, str(total_dict))
        return (model, False)
