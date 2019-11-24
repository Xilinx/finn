import numpy as np
import onnx.helper as helper

from finn.core.datatype import DataType
from finn.custom_op import CustomOp


def xnorpopcountmatmul(inp0, inp1):
    # extract the operand shapes
    (M, K0) = inp0.shape
    (K1, N) = inp1.shape
    # make sure shapes are compatible with matmul
    assert K0 == K1
    K = K0
    # we simulate XNOR-popcount matrix multiplication as a regular bipolar
    # matrix multiplication followed by some post processing
    # first, convert binary inputs to bipolar
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
    def get_nodeattr_types(self):
        return {}

    def make_shape_compatible_op(self):
        node = self.onnx_node
        return helper.make_node(
            "MatMul", [node.input[0], node.input[1]], [node.output[0]]
        )

    def infer_node_datatype(self, model):
        node = self.onnx_node
        # ensure inputs are binary
        assert model.get_tensor_datatype(node.input[0]) == DataType["BINARY"]
        assert model.get_tensor_datatype(node.input[1]) == DataType["BINARY"]
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
