# Copyright (c) 2020 Xilinx, Inc.
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
# * Neither the name of Xilinx nor the names of its
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

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.base import CustomOp

from finn.core.onnx_exec import execute_onnx

# TODO move StreamingDataflowPartition to HLSCustomOp base class


class StreamingDataflowPartition(CustomOp):
    """Class that corresponds to the meta/container node StreamingDataflowPartition
    which is a placeholder for a group of fpgadataflow nodes that have been separated
    out into a FINN-ONNX model of its own. Note that is does not produce any HLS or
    bitfile by itself."""

    def get_nodeattr_types(self):
        return {
            "model": ("s", True, ""),
            "res_estimate": ("s", False, ""),
            "res_hls": ("s", False, ""),
            "res_synth": ("s", False, ""),
            "slr": ("i", False, -1),
            "partition_id": ("i", False, 0),
            "device_id": ("i", False, 0),
            "mem_port": ("s", False, ""),
            "instance_name": ("s", False, ""),
            "return_full_exec_context": ("i", False, 0),
        }

    def make_shape_compatible_op(self, model):
        pass

    def infer_node_datatype(self, model):
        pass

    def execute_node(self, context, graph):
        model = ModelWrapper(self.get_nodeattr("model"))
        return_full_exec_context = self.get_nodeattr("return_full_exec_context") == 1
        node = self.onnx_node
        inp_ctx = dict(filter(lambda x: x[0] in node.input, context.items()))
        # inputs may have been renamed in partition
        for i, old_iname in enumerate(node.input):
            new_iname = model.graph.input[i].name
            if old_iname != new_iname:
                inp_ctx[new_iname] = inp_ctx[old_iname]
                del inp_ctx[old_iname]
        ret = execute_onnx(model, inp_ctx, return_full_exec_context)
        # outputs may have been renamed in partition
        for i, node_oname in enumerate(node.output):
            model_oname = model.graph.output[i].name
            context[node_oname] = ret[model_oname]
        # prefix and insert exec context entries
        if return_full_exec_context:
            for tname in ret.keys():
                if tname not in [x.name for x in model.graph.output]:
                    context[node.name + "_" + tname] = ret[tname]
        pass

    def verify_node(self):
        info_messages = []

        # verify number of attributes
        num_of_attr = 1
        if len(self.onnx_node.attribute) == num_of_attr:
            info_messages.append("The number of attributes is correct")
        else:
            info_messages.append(
                """The number of attributes is incorrect,
            {} should have {} attributes""".format(
                    self.onnx_node.op_type, num_of_attr
                )
            )
        # verify that all necessary attributes exist
        try:
            self.get_nodeattr("model")
            info_messages.append("All necessary attributes exist")
        except Exception:
            info_messages.append(
                """The necessary attributes do not exist.
                StreamingDataflowPartition needs the following attribute(s):
                model"""
            )

        # verify the number of inputs
        if len(self.onnx_node.input) >= 1:
            info_messages.append("The number of inputs is correct")
        else:
            info_messages.append("StreamingDataflowPartition needs 1 data input")

        return info_messages
