import numpy as np
from onnx import TensorProto, helper
import onnxruntime as rt

from finn.custom_op import CustomOp
from finn.custom_op.im2col import compute_conv_output_dim
from finn.core.datatype import DataType


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
        # create a standard average pooling node to help calculate the result
        node = self.onnx_node
        k = self.get_nodeattr("kernel")
        s = self.get_nodeattr("stride")
        ishape = context[node.input[0]].shape
        oshape = context[node.output[0]].shape
        inp = helper.make_tensor_value_info(node.input[0], TensorProto.FLOAT, ishape)
        outp = helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, oshape)
        node_avgpool = helper.make_node(
            "AveragePool",
            inputs=[node.input[0]],
            outputs=[node.output[0]],
            kernel_shape=[k, k],
            strides=[s, s]
        )
        graph_avgpool = helper.make_graph(
                nodes=[node_avgpool],
                name="single-avgpool-exec",
                inputs=[inp],
                outputs=[outp],
                )
        model_avgpool = helper.make_model(graph_avgpool)
        idict = {node.input[0] : context[node.input[0]]}
        sess = rt.InferenceSession(model_avgpool.SerializeToString())
        result_temp = sess.run(None, idict)
        # remove scaling introduced by average
        result_temp = (result_temp[0] * (k * k)).astype(int)
        max_value = np.max(result_temp)
        max_bit_width = int(max_value).bit_length()
        shift_bits = max_bit_width - self.get_nodeattr("obits")
        shift_array = np.ones(result_temp.shape, dtype=np.int) * shift_bits
        result = np.right_shift(result_temp, shift_array)

        context[node.output[0]] = result

    def verify_node(self):
        pass
