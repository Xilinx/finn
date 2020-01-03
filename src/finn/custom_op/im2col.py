import numpy as np
from onnx import TensorProto, helper

from finn.custom_op import CustomOp


def get_im2col_indices(x_shape, k, stride):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert H == W
    assert (W - k) % stride == 0
    ofm_dim = int((W - k) / stride + 1)

    i0 = np.repeat(np.arange(k), k)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(ofm_dim), ofm_dim)
    j0 = np.tile(np.arange(k), k * C)
    j1 = stride * np.tile(np.arange(ofm_dim), ofm_dim)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), k * k).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(x, k, stride):
    """ An implementation of im2col based on indexing """

    l, i, j = get_im2col_indices(x.shape, k, stride)

    cols = x[:, l, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(k * k * C, -1)
    cols = cols.transpose(1, 0)

    # rearranging the output so it matches with finn-hlslib function
    # swapping the columns according to the input channel
    # if C > 1 :
    parts = {}
    for ch in range(C):
        parts[ch] = []

    for i in range(cols.shape[1]):
        if i % C == 0:
            parts[0].append(i)
        elif (i + (C - 1)) % C == 0:
            parts[1].append(i)
        elif (i + (C - 2)) % C == 0:
            parts[2].append(i)
        elif (i + (C - 3)) % C == 0:
            parts[3].append(i)
    permutation = []
    for i in parts:
        for num in parts[i]:
            permutation.append(num)

    i = np.argsort(permutation)
    cols = cols[:, i]
    return cols.reshape(1, -1, k * k * C)


class Im2Col(CustomOp):
    def get_nodeattr_types(self):
        return {
            "stride": ("i", True, 1),
            "kernel_size": ("i", True, 1),
            "input_shape": ("s", True, ""),
        }

    def make_shape_compatible_op(self):
        k = self.get_nodeattr("kernel_size")
        stride = self.get_nodeattr("stride")
        ishape = self.get_nodeattr("input_shape")

        # convert string into list of integers
        ishape = ishape.strip("(")
        ishape = ishape.strip(")")
        ishape = ishape.split(",")
        for i in range(0, len(ishape)):
            ishape[i] = int(ishape[i])

        # extract all necessary information and determine output dimensions
        ifm_ch = ishape[1]
        ifm_dim = ishape[2]
        ofm_dim = int(((ifm_dim - k) / stride) + 1)
        outpix = ofm_dim * ofm_dim

        # implement tensor with correct shape
        values = np.random.randn(1, outpix, k * k * ifm_ch).astype(np.float32)
        return helper.make_node(
            "Constant",
            inputs=[],
            outputs=["values"],
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
        node = self.onnx_node
        k = self.get_nodeattr("kernel_size")
        stride = self.get_nodeattr("stride")
        x = context[node.input[0]]
        output = im2col_indices(x, k, stride)
        context[node.output[0]] = output

    def verify_node(self):
        node = self.onnx_node

        info_messages = []

        # verify number of attributes
        num_of_attr = 3
        if len(node.attribute) == num_of_attr:
            info_messages.append("The number of attributes is correct")
        else:
            info_messages.append(
                """The number of attributes is incorrect,
            {} should have {} attributes""".format(
                    node.op_type, num_of_attr
                )
            )

        # verify that "domain" is set to "finn"
        domain_value = node.domain
        if domain_value == "finn":
            info_messages.append("Attribute domain is set correctly")
        else:
            info_messages.append('Attribute domain should be set to "finn"')

        # verify that all necessary attributes exist
        try:
            self.get_nodeattr("stride")
            self.get_nodeattr("kernel_size")
            info_messages.append("All necessary attributes exist")
        except Exception:
            info_messages.append(
                """The necessary attributes do not exist.
                Im2Col needs the following attributes:
                stride, kernel_size"""
            )

        # verify the number of inputs
        if len(node.input) == 1:
            info_messages.append("The number of inputs is correct")
        else:
            info_messages.append("{} needs 1 data input".format(node.op_type))

        return info_messages
