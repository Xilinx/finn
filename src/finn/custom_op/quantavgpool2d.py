import numpy as np
from onnx import TensorProto, helper

from finn.custom_op import CustomOp
from finn.custom_op.im2col import compute_conv_output_dim


class QuantAvgPool2d(CustomOp):
    """Class that corresponds to the quantized average pooling
    layer from brevitas"""

    def get_nodeattr_types(self):
        return {
            "stride": ("i", True, 1),
            "kernel": ("i", True, 1),
            "ibits": ("s", True, ""),
            "obits": ("i", False, 0),
            "signed": ("i", False, 0),
        }

    def make_shape_compatible_op(self, model):
        node = self.onnx_node
        inp = node.input[0]
        ishape = model.get_tensor_shape(inp)
        # we assume that the shape is (NCHW) and H=W
        assert len(ishape) == 4, "Unexpected input shape for QuantAvgPool2d"
        assert (
            ishape[2] == ishape[3]
        ), "QuantAvgPool2d for non-square images unsupported"
        ch = ishape[1]
        idim = ishape[2]
        k = self.get_nodeattr("kernel")
        stride = self.get_nodeattr("stride")
        odim = compute_conv_output_dim(idim, k, stride)

        # implement tensor with correct shape
        values = np.random.randn(1, ch, odim, odim).astype(np.float32)
        return helper.make_node(
            "Constant",
            inputs=[],
            outputs=[node.output[0]],
            value=helper.make_tensor(
                name="const_tensor",
                data_type=TensorProto.FLOAT,
                dims=values.shape,
                vals=values.flatten().astype(float),
            ),
        )

    def infer_node_datatype(self, model):
        node = self.onnx_node
        # data type stays the same
        dtype = model.get_tensor_datatype(node.input[0])
        model.set_tensor_datatype(node.output[0], dtype)

    def execute_node(self, context, graph):
        pass

    def verify_node(self):
        pass
