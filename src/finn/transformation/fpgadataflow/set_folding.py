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
from finn.transformation import Transformation


def divisors(num):
    for x in range(1, num + 1):
        if (num % x) == 0:
            yield x


class SetFolding(Transformation):
    """Set parallelism attributes in all nodes to meet a specific
    target expressed as cycles per frame. Applies the following rules
    when folding conv layers which have two attributes (PE and SIMD):
    -first increases SIMD while weight stream width per PE is <= 36
    -then increases PE until the target is met or max PE reached"""

    def __init__(self, cycles_target=1000):
        super().__init__()
        self.cycles_target = cycles_target

    def optimize_attribute_val(self, node_inst, max_val, attr_name):
        node_inst.set_nodeattr(attr_name, 1)
        for val in divisors(max_val):
            node_inst.set_nodeattr(attr_name, val)
            cyc = node_inst.get_exp_cycles()
            if cyc < self.cycles_target:
                # finish if target met
                break

    def apply(self, model):
        graph = model.graph
        for node in graph.node:
            op_type = node.op_type
            # TODO: ensure node is fpgadataflow
            node_inst = getCustomOp(node)
            if op_type == "StreamingFCLayer_Batch":
                max_simd = node_inst.get_nodeattr("MW")
                max_pe = node_inst.get_nodeattr("MH")
                node_inst.set_nodeattr("PE", 1)
                node_inst.set_nodeattr("SIMD", 1)
                # increase SIMD until either we meet
                # the target or weight stream becomes
                # too wide
                for simd_val in divisors(max_simd):
                    prev_simd_val = node_inst.get_nodeattr("SIMD")
                    node_inst.set_nodeattr("SIMD", simd_val)
                    cyc = node_inst.get_exp_cycles()
                    if cyc < self.cycles_target:
                        # finish if target met
                        break
                    if (
                        node_inst.get_weight_datatype().bitwidth()
                        * node_inst.get_nodeattr("SIMD")
                        > 36
                    ):
                        # revert if we've gone above width threshold
                        node_inst.set_nodeattr("SIMD", prev_simd_val)
                        break

                # increase PE until target met or reached max_pe
                self.optimize_attribute_val(node_inst, max_pe, "PE")

            elif op_type == "Vector_Vector_Activate_Batch":
                max_pe = node_inst.get_nodeattr("Channels")
                self.optimize_attribute_val(node_inst, max_pe, "PE")
                # also set the folding of the upsteam DW SWU
                # which must be identical to this node
                swu_node = model.find_producer(node.input[0])
                if swu_node.op_type == "ConvolutionInputGenerator":
                    swu_node_inst = getCustomOp(swu_node)
                    pe = node_inst.get_nodeattr("PE")
                    swu_node_inst.set_nodeattr("SIMD", pe)
                else:
                    raise Exception("Expected SWU on input")
            elif op_type == "AddStreams_Batch":
                max_pe = node_inst.get_nodeattr("NumChannels")
                self.optimize_attribute_val(node_inst, max_pe, "PE")
            elif op_type == "ChannelwiseOp_Batch":
                max_pe = node_inst.get_nodeattr("NumChannels")
                self.optimize_attribute_val(node_inst, max_pe, "PE")
            elif op_type == "DuplicateStreams_Batch":
                max_pe = node_inst.get_nodeattr("NumChannels")
                self.optimize_attribute_val(node_inst, max_pe, "PE")
            elif op_type == "GlobalAccPool_Batch":
                max_pe = node_inst.get_nodeattr("NumChannels")
                self.optimize_attribute_val(node_inst, max_pe, "PE")
            elif op_type == "Pool_Batch":
                max_pe = node_inst.get_nodeattr("Channels")
                self.optimize_attribute_val(node_inst, max_pe, "PE")
                # also set the folding of the upsteam DW SWU
                # which must be identical to this node
                swu_node = model.find_producer(node.input[0])
                if swu_node.op_type == "ConvolutionInputGenerator":
                    swu_node_inst = getCustomOp(swu_node)
                    pe = node_inst.get_nodeattr("PE")
                    swu_node_inst.set_nodeattr("SIMD", pe)
                else:
                    raise Exception("Expected SWU on input")
            elif op_type == "Thresholding_Batch":
                max_pe = node_inst.get_nodeattr("NumChannels")
                self.optimize_attribute_val(node_inst, max_pe, "PE")
            elif op_type == "DownSampler":
                max_simd = node_inst.get_nodeattr("NumChannels")
                self.optimize_attribute_val(node_inst, max_simd, "SIMD")
            elif op_type == "FMPadding_Batch":
                max_simd = node_inst.get_nodeattr("NumChannels")
                self.optimize_attribute_val(node_inst, max_simd, "SIMD")
            elif op_type == "ConvolutionInputGenerator":
                depthwise = node_inst.get_nodeattr("depthwise")
                if depthwise == 0:
                    max_simd = node_inst.get_nodeattr("IFMChannels")
                    self.optimize_attribute_val(node_inst, max_simd, "SIMD")
                else:
                    continue

        return (model, False)
