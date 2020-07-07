import numpy as np
from onnx import TensorProto, helper

from finn.custom_op import CustomOp
import finn.util.basic as util
from finn.core.datatype import DataType

# adapted from A. Karpathy's CS231 im2col code
# utilities to generate a patch matrix from a multichannel image
# of shape (batches, channels, height, width)


def compute_conv_output_dim(ifm_dim, k, stride, pad=0):
    """Returns spatial output dimension size for convolution with given params."""
    return int(((ifm_dim + 2 * pad - k) / stride) + 1)


def get_im2col_indices_nchw(
    x_shape, field_height, field_width, padding=0, stride_y=1, stride_x=1
):
    """Returns im2col indices."""
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    out_height = compute_conv_output_dim(H, field_height, stride_y, padding)
    out_width = compute_conv_output_dim(W, field_width, stride_x, padding)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride_y * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride_x * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices_nchw(
    x, field_height, field_width, padding=0, stride_y=1, stride_x=1, pad_val=0
):
    """Performs im2col on x with given field height and width, as well as values
    for padding and stride size.
    Returns result of im2col."""
    # Zero-pad the input
    p = padding
    x_padded = np.pad(
        x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant", constant_values=pad_val
    )

    k, i, j = get_im2col_indices_nchw(
        x.shape, field_height, field_width, padding, stride_y, stride_x
    )

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


# ONNX i/o tensor shape assumptions for Im2Col:
# input 0 is the input vector, shape (1, ih, iw, ifm)
# output 0 is the output vector, shape (1, oh, ow, k*k*ifm)
# where:
# * ih, iw are the height and width of the input image
# * oh, ow are the height and width of the output (lowered) image
# * ifm is the number of input channels
# * k is the convolutional kernel size

# note: for the innermost (dot product) dimension of k*k*ifm, we
# assume an internal ordering (k, k, ifm)


class Im2Col(CustomOp):
    def get_nodeattr_types(self):
        return {
            "stride": ("i", True, 1),
            "kernel_size": ("i", True, 1),
            "input_shape": ("s", True, ""),
            "pad_amount": ("i", False, 0),
            "pad_value": ("i", False, 0),
            # depthwise: if != 0, infer ConvolutionInputGenerator with depthwise == 1
            "depthwise": ("i", False, 0),
        }

    def make_shape_compatible_op(self, model):
        k = self.get_nodeattr("kernel_size")
        stride = self.get_nodeattr("stride")
        ishape = self.get_nodeattr("input_shape")
        pad = self.get_nodeattr("pad_amount")

        # convert string into list of integers
        ishape = ishape.strip("(")
        ishape = ishape.strip(")")
        ishape = ishape.split(",")
        for i in range(0, len(ishape)):
            ishape[i] = int(ishape[i])

        # extract all necessary information and determine output dimensions
        ifm_ch = ishape[-1]
        assert len(ishape) == 4, "Unexpected input shape for Im2Col"
        assert ishape[1] == ishape[2], "Im2Col for non-square images unsupported"
        ifm_dim = ishape[1]
        ofm_dim = compute_conv_output_dim(ifm_dim, k, stride, pad)

        # implement tensor with correct shape
        values = np.random.randn(1, ofm_dim, ofm_dim, k * k * ifm_ch).astype(np.float32)
        return helper.make_node(
            "Constant",
            inputs=[],
            outputs=[self.onnx_node.output[0]],
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
        pad = self.get_nodeattr("pad_amount")
        pad_val = self.get_nodeattr("pad_value")
        iname = node.input[0]
        x = context[iname]
        qnt_annotations = graph.quantization_annotation
        ret = util.get_by_name(qnt_annotations, iname, "tensor_name")
        ret = util.get_by_name(ret.quant_parameter_tensor_names, "finn_datatype", "key")
        idt = DataType[ret.value]
        if pad != 0:
            assert idt.allowed(pad_val), "Im2Col dtype must allow pad_val"
        # check that input is NHWC
        assert x.ndim == 4, "Unexpected number of input dims for Im2Col"
        N, H, W, C = x.shape
        assert H == W, "Unexpected input shape for Im2Col"
        out_dim = compute_conv_output_dim(H, k, stride, pad)
        # internally convert input to NCHW
        x = x.transpose(0, 3, 1, 2)
        # call NCHW im2col implementation
        ret = im2col_indices_nchw(x, k, k, pad, stride, stride, pad_val=pad_val)
        # result shape is (k*k*N, out_dim*out_dim), convert to NCHW
        ret = ret.reshape(N, C, k, k, out_dim, out_dim)
        # (N=0,C=1,kh=2,kw=3,H=4,W=5) -> (N=0,H=4,W=5,kh=2,kw=3,C=1)
        ret = ret.transpose(0, 4, 5, 2, 3, 1)
        ret = ret.reshape(N, out_dim, out_dim, k * k * C)

        # ret = ret.reshape(N, k * k * C, out_dim, out_dim)
        # convert output back to NHWC
        # ret = ret.transpose(0, 2, 3, 1)
        context[node.output[0]] = ret

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
