import numpy as np
import warnings
from onnx.helper import make_node
from qonnx.core.datatype import DataType
from scipy.special import softmax

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class QuantSoftmax(HWCustomOp):
    """Abstraction layer for HW implementation of VectorVectorActivation layers."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            "ifm_dim": ("ints", True, []),
            "simd": ("i", False, 1),
            # FINN DataTypes for inputs, weights, outputs
            "input_data_type": ("s", True, ""),
            "output_data_type": ("s", True, ""),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_normal_input_shape(self, ind=0):
        return self.get_nodeattr("ifm_dim")

    def get_normal_output_shape(self, ind=0):
        return self.get_normal_input_shape()

    def get_number_output_values(self):
        raise NotImplementedError("This function is not yet implemented.")

    def quantise_to_int(self, arr, dtype):
        max_val = np.iinfo(dtype).max
        output = np.zeros_like(arr, dtype=dtype)
        frac_part = arr - np.floor(arr)
        scaled_frac = frac_part * max_val
        output = scaled_frac.astype(dtype)
        output[arr >= 1.0] = max_val
        return output

    def execute_node(self, context, graph):
        node = self.onnx_node
        input_data = context[node.input[0]]
        output_data = softmax(input_data, axis=-1)
        qsm_out = self.quantise_to_int(output_data, np.int8)
        context[node.output[0]] = qsm_out

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        data_type = DataType[self.get_nodeattr("input_data_type")]
        # the hlslib op always pads with zeros, so ensure that the DataType
        # is able to represent zeros
        assert data_type.allowed(0), "DataType must support zero"
        return data_type

    def make_shape_compatible_op(self, model):
        shape = self.get_normal_input_shape()
        # create an ONNX Softmax node with the same shape as this one
        return make_node(
            "Softmax",
            inputs=[self.onnx_node.input[0]],
            outputs=[self.onnx_node.output[0]],
            shape=list(shape),
        )

    def infer_node_datatype(self, model):
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            warn_str = "input_data_type changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype()),
                str(idt),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("input_data_type", idt.name)

        # set output datatype from property
        odt = self.get_output_datatype()
        model.set_tensor_datatype(node.output[0], odt)

    def verify_node(self):
        raise NotImplementedError

    def get_instream_width(self, ind=0):
        ibits = self.get_input_datatype().bitwidth()
        simd = self.get_nodeattr("simd")
        return ibits * simd

    def get_outstream_width(self, ind=0):
        obits = self.get_output_datatype().bitwidth()
        simd = self.get_nodeattr("simd")
        return obits * simd

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output."""
        data_type = DataType[self.get_nodeattr("output_data_type")]
        # the hlslib op always pads with zeros, so ensure that the DataType
        # is able to represent zeros
        assert data_type.allowed(0), "DataType must support zero"
        return data_type

    def get_folded_output_shape(self, ind=0):
        return self.get_folded_input_shape()

    def get_folded_input_shape(self, ind=0):
        normal_ishape = list(self.get_normal_input_shape())
        simd = self.get_nodeattr("simd")
        assert normal_ishape[-1] % simd == 0, "SIMD must divide into input dimension"
        fold = int(normal_ishape[-1] / simd)
        folded_ishape = normal_ishape[:-1] + [fold, simd]
        return tuple(folded_ishape)
