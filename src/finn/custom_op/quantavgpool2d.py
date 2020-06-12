import numpy as np
from onnx import TensorProto, helper
import onnxruntime as rt

from finn.custom_op import CustomOp
from finn.core.datatype import DataType


class QuantAvgPool2d(CustomOp):
    """Class that corresponds to the quantized average pooling
    layer from brevitas"""

    def get_nodeattr_types(self):
        return {
            "stride": ("i", True, 1),
            "kernel": ("i", True, 1),
            "ibits": ("i", True, 1),
            "obits": ("i", True, 1),
            "signed": ("i", True, 0),
        }

    def make_shape_compatible_op(self, model):
        node = self.onnx_node
        k = self.get_nodeattr("kernel")
        s = self.get_nodeattr("stride")
        return helper.make_node(
            "AveragePool",
            inputs=[node.input[0]],
            outputs=[node.output[0]],
            kernel_shape=[k, k],
            strides=[s, s],
        )

    def infer_node_datatype(self, model):
        node = self.onnx_node
        bw = self.get_nodeattr("obits")
        if bw in [2,4,8,16,32]:
            if self.get_nodeattr("signed") == 0:
                dtype = DataType["UINT%d" % bw]
            else:
                dtype = DataType["INT%d" % bw]
        else:
            raise Exception("Unsupported output datatype for QuantAvgPool2d")
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
            strides=[s, s],
        )
        graph_avgpool = helper.make_graph(
            nodes=[node_avgpool],
            name="single-avgpool-exec",
            inputs=[inp],
            outputs=[outp],
        )
        model_avgpool = helper.make_model(graph_avgpool)
        idict = {node.input[0]: np.round(context[node.input[0]])}
        sess = rt.InferenceSession(model_avgpool.SerializeToString())
        result_temp = sess.run(None, idict)
        # remove scaling introduced by average
        result_temp = result_temp[0] * (k * k)
        ibits = self.get_nodeattr("ibits")
        max_value = 2 ** ibits - 1
        max_value = max_value * k * k
        max_bit_width = int(max_value).bit_length()
        shift_bits = max_bit_width - self.get_nodeattr("obits")
        result = np.right_shift(result_temp.astype(int), shift_bits)
        context[node.output[0]] = result.astype(np.float32)

    def verify_node(self):
        pass
