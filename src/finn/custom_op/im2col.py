import numpy as np

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
    return cols


class Im2Col(CustomOp):
    def get_nodeattr_types(self):
        return {
            "stride": ("i", True, 1),
            "kernel_size": ("i", True, 1),
        }

    def make_shape_compatible_op(self):
        pass

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
        pass
