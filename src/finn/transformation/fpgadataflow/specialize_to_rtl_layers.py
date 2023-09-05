# Copyright (c) 2023, AMD
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

from qonnx.transformation.base import Transformation
from qonnx.custom_op.registry import getCustomOp
from qonnx.core.datatype import DataType
from onnx import helper
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.general import GiveUniqueNodeNames
from finn.transformation.fpgadataflow.minimize_accumulator_width import MinimizeAccumulatorWidth

class InferRTLMatrixVectorActivation(Transformation):
    """Convert (HLS-based) MatrixVectorActivation layers to specialized RTL layers if supported."""

    def __init__(self):
        super().__init__()

    def _is_rtl_variant_compatible(self, n):
        no_activation = getCustomOp(n).get_nodeattr("noActivation") == 1
        act_width_in_range = (DataType[getCustomOp(n).get_nodeattr("inputDataType")].bitwidth() <= 8) or (DataType[getCustomOp(n).get_nodeattr("inputDataType")].bitwidth() == 9 and DataType[getCustomOp(n).get_nodeattr("inputDataType")].min() < 0)
        weight_width_in_range = DataType[getCustomOp(n).get_nodeattr("weightDataType")].bitwidth() <= 8
        folding_supported = (getCustomOp(n).get_nodeattr("MH") % getCustomOp(n).get_nodeattr("PE") == 0) and (getCustomOp(n).get_nodeattr("MW") % getCustomOp(n).get_nodeattr("SIMD") == 0)

        if (no_activation and act_width_in_range and weight_width_in_range and folding_supported):
            return True
        else:
            return False


    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "MatrixVectorActivation":
                preferred_in_rtl = getCustomOp(n).get_nodeattr("impl") == "rtl" and getCustomOp(n).get_nodeattr("resType") == "dsp"
                supported_in_rtl = self._is_rtl_variant_compatible(n)
                if (preferred_in_rtl and supported_in_rtl):
                    mvau_input = n.input[0]
                    mvau_weight = n.input[1]
                    mvau_output = n.output[0]
                    inputDataType = getCustomOp(n).get_nodeattr("inputDataType")
                    weightDataType = getCustomOp(n).get_nodeattr("weightDataType")
                    outputDataType = getCustomOp(n).get_nodeattr("outputDataType")
                    numInputVectors = getCustomOp(n).get_nodeattr("numInputVectors")
                    mw = getCustomOp(n).get_nodeattr("MW")
                    mh = getCustomOp(n).get_nodeattr("MH")
                    simd = getCustomOp(n).get_nodeattr("SIMD")
                    pe = getCustomOp(n).get_nodeattr("PE")
                    mem_mode = getCustomOp(n).get_nodeattr("mem_mode")
                    ram_style = getCustomOp(n).get_nodeattr("ram_style")
                    runtime_writeable_weights = getCustomOp(n).get_nodeattr("runtime_writeable_weights")

                    new_node = helper.make_node(
                        "MatrixVectorActivation_rtl",
                        [mvau_input, mvau_weight],
                        [mvau_output],
                        domain="finn.custom_op.fpgadataflow",
                        backend="fpgadataflow",
                        MW=mw,
                        MH=mh,
                        SIMD=simd,
                        PE=pe,
                        inputDataType=inputDataType,
                        weightDataType=weightDataType,
                        outputDataType=outputDataType,
                        numInputVectors=numInputVectors,
                        mem_mode=mem_mode,
                        name=n.name + "_rtl",
                        ram_style=ram_style,
                        runtime_writeable_weights=runtime_writeable_weights
                    )
                    graph.node.insert(node_ind, new_node)
                    # remove old node
                    graph.node.remove(n)
                    graph_modified=True
        
        if graph_modified:
            model = model.transform(MinimizeAccumulatorWidth())
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
            model = model.transform(GiveUniqueNodeNames())
        
        return (model, graph_modified)