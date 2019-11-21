import numpy as np
import onnx.helper as helper

from finn.core.datatype import DataType
from finn.core.utils import get_by_name
from finn.custom_op import CustomOp


class MultiThreshold(CustomOp):
    def make_shape_compatible_op(self, node):
        return helper.make_node("Relu", [node.input[0]], [node.output[0]])

    def infer_node_datatype(self, node, model):
        try:
            odt = get_by_name(node.attribute, "out_dtype").s.decode("utf-8")
            model.set_tensor_datatype(node.output[0], DataType[odt])
        except AttributeError:
            # number of thresholds decides # output bits
            # use get_smallest_possible, assuming unsigned
            n_thres = model.get_tensor_shape(node.input[1])[1]
            odtype = DataType.get_smallest_possible(n_thres)
            model.set_tensor_datatype(node.output[0], odtype)

    def execute_node(self, node, context, graph):
        # save inputs
        v = context[node.input[0]]
        thresholds = context[node.input[1]]
        # retrieve attributes if output scaling is used
        try:
            out_scale = get_by_name(node.attribute, "out_scale").f
        except AttributeError:
            out_scale = None
        try:
            out_bias = get_by_name(node.attribute, "out_bias").f
        except AttributeError:
            out_bias = None
        # calculate output
        output = self._execute(v, thresholds, out_scale, out_bias)
        # setting context according to output
        context[node.output[0]] = output

    def _compare(self, x, y):
        if x >= y:
            return 1.0
        else:
            return 0.0

    def _execute(self, v, thresholds, out_scale=None, out_bias=None):
        # the inputs are expected to be in the shape (N,C,H,W)
        # N : Batch size
        # C : Number of channels
        # H : Heigth of the input images
        # W : Width of the input images
        #
        # the thresholds are expected to be in the shape (C, B)
        # C : Number of channels (must be the same value as C in input tensor
        #     or 1 if all channels use the same threshold value)
        # B : Desired activation steps => i.e. for 4-bit activation,
        #     B=7 (2^(n)-1 and n=4)
        # the output tensor will be scaled by out_scale and biased by out_bias
        # assert threshold shape
        is_global_threshold = thresholds.shape[0] == 1
        assert (v.shape[1] == thresholds.shape[0]) or is_global_threshold
        # save the required shape sizes for the loops (N, C and B)
        num_batch = v.shape[0]
        num_channel = v.shape[1]
        num_act = thresholds.shape[1]
        # reshape inputs to enable channel-wise reading
        vr = v.reshape((v.shape[0], v.shape[1], -1))
        # save the new shape size of the images
        num_img_elem = vr.shape[2]
        # initiate output tensor
        ret = np.zeros_like(vr)
        # iterate over thresholds channel-wise
        for t in range(num_channel):
            channel_thresh = thresholds[0] if is_global_threshold else thresholds[t]
            # iterate over batches
            for b in range(num_batch):
                # iterate over image elements on which the thresholds will be applied
                for elem in range(num_img_elem):
                    # iterate over the different thresholds for one channel
                    for a in range(num_act):
                        # apply successive thresholding to every element
                        ret[b][t][elem] += self._compare(
                            vr[b][t][elem], channel_thresh[a]
                        )
        if out_scale is None:
            out_scale = 1.0
        if out_bias is None:
            out_bias = 0.0
        return out_scale * ret.reshape(v.shape) + out_bias
