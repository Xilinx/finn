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
import numpy as np
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import get_by_name, is_finn_op, qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class FINNLoop(HWCustomOp):
    """Class that corresponds to the meta/container node FINN loop
    which is a placeholder for a group of fpgadataflow nodes that have been separated
    out into a FINN-ONNX model of its own and are meant to be executed in a loop."""

    def get_nodeattr_types(self):
        return {
            "body": ("g", True, ""),
            "iteration": ("i", False, 1),
            # FINN input datatype
            "inputDataType": ("s", True, ""),
            # FINN output datatype
            "outputDataType": ("s", True, ""),
            # list with loop body node names (!)
            # corresponding to input index
            "paramNodes": ("strings", True, [""]),
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
                else:
                    return super().get_nodeattr(name)
        except KeyError:
            raise AttributeError("Op has no such attribute: " + name)

    def set_nodeattr(self, name, value):
        """Set a node attribute by name. Data is stored inside the ONNX node's
        AttributeProto container. Attribute must be part of get_nodeattr_types."""
        try:
            (dtype, req, def_val, allowed_values) = self.get_nodeattr_def(name)
            attr = get_by_name(self.onnx_node.attribute, name)
            if attr is not None:
                # dtype indicates which ONNX Attribute member to use
                # g : graph
                if dtype == "g":
                    attr.g.CopyFrom(value)
                else:
                    return super().set_nodeattr(name, value)
        except KeyError:
            raise AttributeError("Op has no such attribute: " + name)

    def get_normal_input_shape(self, ind=0):
        loop_body = self.get_nodeattr("body")
        if ind == 0:
            # get first node in loop body and return
            # normal input shape
            node = loop_body.graph.node[0]
            if is_finn_op(node.op_type):
                inst = getCustomOp(node)
                ishape = inst.get_normal_input_shape(0)
            else:
                ishape = loop_body.get_tensor_shape(node.input[0])
        else:
            paramNodes = self.get_nodeattr("paramNodes")
            node = loop_body.get_node_from_name(paramNodes[ind - 1])
            if is_finn_op(node.op_type):
                inst = getCustomOp(node)
                ishape = inst.get_normal_input_shape(1)
            else:
                ishape = loop_body.get_tensor_shape(node.input[1])
        return ishape

    def get_normal_output_shape(self, ind=0):
        loop_body = self.get_nodeattr("body")
        # get last node in loop body and return
        # normal output shape
        node = loop_body.graph.node[-1]
        if is_finn_op(node.op_type):
            inst = getCustomOp(node)
            oshape = inst.get_normal_output_shape(0)
        else:
            oshape = loop_body.get_tensor_shape(node.output[0])
        return oshape

    def get_folded_input_shape(self, ind=0):
        loop_body = self.get_nodeattr("body")
        if ind == 0:
            # get first node in loop body and return
            # normal input shape
            node = loop_body.graph.node[0]
            inst = getCustomOp(node)
            ishape = inst.get_folded_input_shape(0)
        else:
            paramNodes = self.get_nodeattr("paramNodes")
            node = loop_body.get_node_from_name(paramNodes[ind - 1])
            inst = getCustomOp(node)
            ishape = inst.get_folded_input_shape(1)
        return ishape

    def get_folded_output_shape(self, ind=0):
        loop_body = self.get_nodeattr("body")
        # get last node in loop body and return
        # normal output shape
        node = loop_body.graph.node[-1]
        inst = getCustomOp(node)
        return inst.get_folded_output_shape(0)

    def infer_node_datatype(self, model):
        pass

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        if ind == 0:
            idt = DataType[self.get_nodeattr("inputDataType")]
        else:
            loop_body = self.get_nodeattr("body")
            paramNodes = self.get_nodeattr("paramNodes")
            node = loop_body.get_node_from_name(paramNodes[ind - 1])
            if is_finn_op(node.op_type):
                inst = getCustomOp(node)
                idt = inst.get_input_datatype(1)
            else:
                idt = loop_body.get_tensor_datatype(node.input[1])
        return idt

    def get_output_datatype(self, ind=0):
        odt = DataType[self.get_nodeattr("outputDataType")]
        return odt

    def get_instream_width(self, ind=0):
        loop_body = self.get_nodeattr("body")
        if ind == 0:
            # get first node in loop body and return
            # normal input shape
            node = loop_body.graph.node[0]
            inst = getCustomOp(node)
            iwidth = inst.get_instream_width(0)
        else:
            paramNodes = self.get_nodeattr("paramNodes")
            node = loop_body.get_node_from_name(paramNodes[ind - 1])
            inst = getCustomOp(node)
            iwidth = inst.get_instream_width(1)
        return iwidth

    def get_outstream_width(self, ind=0):
        loop_body = self.get_nodeattr("body")
        # get last node in loop body and return
        # normal output shape
        node = loop_body.graph.node[-1]
        inst = getCustomOp(node)
        return inst.get_outstream_width(0)

    def get_number_output_values(self):
        loop_body = self.get_nodeattr("body")
        # get last node in loop body and return
        # normal output values
        node = loop_body.graph.node[-1]
        inst = getCustomOp(node)
        return inst.get_number_output_values()

    def execute_node(self, context, graph):
        node = self.onnx_node
        inp_values = context[node.input[0]]
        loop_body = self.get_nodeattr("body")
        # for each iteration run execution
        iteration = self.get_nodeattr("iteration")
        input_dict = {loop_body.graph.input[0].name: inp_values}
        for i_iter in range(iteration):
            for i_input, input_name in enumerate(node.input[1:]):
                # get the paramter from the node input
                params = context[input_name]
                # pass the values in through the input dict
                input_dict[loop_body.graph.input[i_input+1].name] = params[i_iter]
            outp_dict = oxe.execute_onnx(loop_body, input_dict)
            inp_values = outp_dict[loop_body.graph.output[0].name]
            input_dict = {loop_body.graph.input[0].name: inp_values}
        result = outp_dict[loop_body.graph.output[0].name]
        context[node.output[0]] = np.asarray(result, dtype=np.float32)
