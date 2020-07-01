import numpy as np
from onnx import TensorProto, helper
import onnxruntime as rt

from finn.custom_op import CustomOp
from finn.core.datatype import DataType
from finn.custom_op.maxpoolnhwc import compute_pool_output_dim


class QuantAvgPool2d(CustomOp):
    """Class that corresponds to the quantized average pooling
    layer from brevitas"""

    def get_nodeattr_types(self):
        return {
            "stride": ("i", True, 1),
            "kernel": ("i", True, 1),
            "ibits": ("i", True, 1),
            "obits": ("i", True, 1),
            # determines if values are signed (set to "1") or unsigned ("0")
            "signed": ("i", True, 0),
            # data layout attribute can be set to "NCHW" or "NHWC"
            "data_layout": ("s", False, "NCHW"),
        }

    def make_shape_compatible_op(self, model):
        node = self.onnx_node
        k = self.get_nodeattr("kernel")
        s = self.get_nodeattr("stride")
        data_layout = self.get_nodeattr("data_layout")
        if data_layout == "NCHW":
            return helper.make_node(
                "AveragePool",
                inputs=[node.input[0]],
                outputs=[node.output[0]],
                kernel_shape=[k, k],
                strides=[s, s],
            )
        elif data_layout == "NHWC":
            iname = node.input[0]
            ishape = model.get_tensor_shape(iname)
            (n, hi, wi, c) = ishape
            ho = compute_pool_output_dim(hi, k, s)
            wo = compute_pool_output_dim(wi, k, s)
            oshape = (n, ho, wo, c)
            # implement tensor with correct shape
            values = np.random.randn(*oshape).astype(np.float32)
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

        else:
            raise Exception(
                """Datalayout for QuantAvgPool2d is set to an invalid value.
                    Has to be set to "NCHW" or "NHWC"."""
            )

    def infer_node_datatype(self, model):
        node = self.onnx_node
        bw = self.get_nodeattr("obits")
        if bw in [2, 4, 8, 16, 32]:
            if self.get_nodeattr("signed") == 0:
                dtype = DataType["UINT%d" % bw]
            else:
                dtype = DataType["INT%d" % bw]
        else:
            raise Exception("Unsupported output datatype for QuantAvgPool2d")
        model.set_tensor_datatype(node.output[0], dtype)

    def get_accum_size(self):
        ibits = self.get_nodeattr("ibits")
        k = self.get_nodeattr("kernel")
        max_value = 2 ** ibits - 1
        max_value = max_value * k * k
        max_bit_width = int(max_value).bit_length()
        return max_bit_width

    def get_shifts(self):
        shift_bits = self.get_accum_size() - self.get_nodeattr("obits")
        shift_bits = shift_bits if shift_bits >= 0 else 0
        return shift_bits

    def execute_node(self, context, graph):
        # create a standard average pooling node to help calculate the result
        node = self.onnx_node
        k = self.get_nodeattr("kernel")
        s = self.get_nodeattr("stride")
        inp_values = context[node.input[0]]
        oshape = context[node.output[0]].shape
        if self.get_nodeattr("data_layout") == "NHWC":
            inp_values = inp_values.transpose(0, 3, 1, 2)
            oshape = (context[node.output[0]]).transpose(0, 3, 1, 2).shape
        ishape = inp_values.shape
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
        idict = {node.input[0]: inp_values}
        sess = rt.InferenceSession(model_avgpool.SerializeToString())
        result_temp = sess.run(None, idict)
        # remove scaling introduced by average
        result_temp = result_temp[0] * (k * k)
        result = np.right_shift(result_temp.astype(int), self.get_shifts())
        if self.get_nodeattr("data_layout") == "NHWC":
            result = result.transpose(0, 2, 3, 1)
        context[node.output[0]] = result.astype(np.float32)

    def verify_node(self):
        info_messages = []
        # verify that "domain" is set to "finn"
        domain_value = self.onnx_node.domain
        if domain_value == "finn":
            info_messages.append("Attribute domain is set correctly")
        else:
            info_messages.append('Attribute domain should be set to "finn"')
        return info_messages
