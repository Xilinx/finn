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

import numpy as np
import onnx.helper as helper

from finn.core.datatype import DataType
from finn.custom_op import CustomOp


def xnorpopcountmatmul(inp0, inp1):
    """Simulates XNOR-popcount matrix multiplication as a regular bipolar
    matrix multiplication followed by some post processing."""
    # extract the operand shapes
    # (M, K0) = inp0.shape
    # (K1, N) = inp1.shape
    K0 = inp0.shape[-1]
    K1 = inp1.shape[0]
    # make sure shapes are compatible with matmul
    assert K0 == K1, "Matrix shapes are not compatible with matmul."
    K = K0
    # convert binary inputs to bipolar
    inp0_bipolar = 2.0 * inp0 - 1.0
    inp1_bipolar = 2.0 * inp1 - 1.0
    # call regular numpy matrix multiplication
    out = np.matmul(inp0_bipolar, inp1_bipolar)
    # XNOR-popcount does not produce the regular dot product result --
    # it returns the number of +1s after XNOR. let P be the number of +1s
    # and N be the number of -1s. XNOR-popcount returns P, whereas the
    # regular dot product result from numpy is P-N, so we need to apply
    # some correction.
    # out = P-N
    # K = P+N
    # out + K = 2P, so P = (out + K)/2
    return (out + K) * 0.5


class XnorPopcountMatMul(CustomOp):
    """Class that corresponds to a XNOR-popcount matrix
    multiplication node."""

    def get_nodeattr_types(self):
        return {}

    def make_shape_compatible_op(self, model):
        node = self.onnx_node
        return helper.make_node(
            "MatMul", [node.input[0], node.input[1]], [node.output[0]]
        )

    def infer_node_datatype(self, model):
        node = self.onnx_node
        # ensure inputs are binary
        assert (
            model.get_tensor_datatype(node.input[0]) == DataType["BINARY"]
        ), """FINN
        DataType of first input is not set to BINARY as it should be."""
        assert (
            model.get_tensor_datatype(node.input[1]) == DataType["BINARY"]
        ), """FINN
        DataTypes of second input is not set to BINARY as it should be."""
        # XNOR-popcount produces unsigned integers, assume uint32
        model.set_tensor_datatype(node.output[0], DataType["UINT32"])

    def execute_node(self, context, graph):
        node = self.onnx_node
        # save inputs
        inp0 = context[node.input[0]]
        inp1 = context[node.input[1]]
        # calculate output
        output = xnorpopcountmatmul(inp0, inp1)
        # set context according to output name
        context[node.output[0]] = output

    def verify_node(self):
        info_messages = []

        # verify number of attributes
        num_of_attr = 0
        if len(self.onnx_node.attribute) == num_of_attr:
            info_messages.append("The number of attributes is correct")
        else:
            info_messages.append(
                """The number of attributes is incorrect,
            {} should have {} attributes""".format(
                    self.onnx_node.op_type, num_of_attr
                )
            )

        # verify that "domain" is set to "finn"
        domain_value = self.onnx_node.domain
        if domain_value == "finn":
            info_messages.append("Attribute domain is set correctly")
        else:
            info_messages.append('Attribute domain should be set to "finn"')

        # verify that all necessary attributes exist
        info_messages.append("XnorPopcountMatMul should not have any attributes")

        # verify the number of inputs
        if len(self.onnx_node.input) == 2:
            info_messages.append("The number of inputs is correct")
        else:
            info_messages.append("XnorPopcountMatMul needs 2 data inputs")

        return info_messages
