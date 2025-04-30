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
from qonnx.util.basic import get_by_name, qonnx_make_model

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class FINNLoop(HWCustomOp):
    """Class that corresponds to the meta/container node FINN loop
    which is a placeholder for a group of fpgadataflow nodes that have been separated
    out into a FINN-ONNX model of its own and are meant to be executed in a loop."""

    def get_nodeattr_types(self):
        return {
            "body": ("g", True, ""),
            "iteration": ("i", False, 1),
        }

    def get_nodeattr(self, name):
        """Get a node attribute by name. Data is stored inside the ONNX node's
        AttributeProto container. Attribute must be part of get_nodeattr_types.
        Default value is returned if attribute is not set."""
        try:
            (dtype, req, def_val, allowed_values) = self.get_nodeattr_def(name)
            attr = get_by_name(self.onnx_node.attribute, name)
            if attr is not None:
                # dtype indicates which ONNX Attribute member to use
                # g : graph
                if dtype == "g":
                    ret = attr.__getattribute__(dtype)
                    ret = ModelWrapper(qonnx_make_model(ret))
                return ret
        except KeyError:
            super().get_nodeattr(self, name)

    def get_normal_input_shape(self, ind=0):
        pass

    def get_normal_output_shape(self, ind=0):
        pass

    def get_folded_input_shape(self, ind=0):
        pass

    def get_folded_output_shape(self, ind=0):
        pass

    def infer_node_datatype(self, model):
        pass

    def get_input_datatype(self, ind=0):
        pass

    def get_output_datatype(self, ind=0):
        pass

    def get_instream_width(self, ind=0):
        pass

    def get_outstream_width(self, ind=0):
        pass

    def get_number_output_values(self):
        pass

    def execute_node(self, context, graph):
        pass
