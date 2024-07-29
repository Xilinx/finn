
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.data_packing import numpy_to_hls_code, pack_innermost_dim_as_hex_string
from onnx.helper import make_node
import warnings
from qonnx.core.datatype import DataType

class QuantSoftmax(HWCustomOp):
    """Abstraction layer for HW implementation of VectorVectorActivation layers."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            "ifm_dim": ("ints", True, []),
            "simd": ("i", False, 1),
            "channels": ("i", True, 0),
            # FINN DataTypes for inputs, weights, outputs
            "data_type": ("s", True, ""),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_normal_input_shape(self, ind=0):
        h, w = self.get_nodeattr("ifm_dim")
        c = self.get_nodeattr("channels")
        return (1, h, w, c)

    def get_normal_output_shape(self, ind=0):
        return self.get_normal_input_shape()

    def get_number_output_values(self):
        raise NotImplementedError("This function is not yet implemented.")

    def execute_node(self, context, graph):
        raise NotImplementedError

    def get_number_output_values(self):
        raise NotImplementedError


    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        data_type = DataType[self.get_nodeattr("data_type")]
        # the hlslib op always pads with zeros, so ensure that the DataType
        # is able to represent zeros
        assert data_type.allowed(0), "DataType must support zero"
        return data_type

    def make_shape_compatible_op(self, model):
        shape = self.get_normal_input_shape()
        # create an ONNX Softmax node with the same shape as this one
        return make_node("Softmax",
                         inputs=[self.onnx_node.input[0]],
                         outputs=[self.onnx_node.output[0]],
                         shape=list(shape)
                         )
    def infer_node_datatype(self, model):
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            warn_str = "data_type changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype()),
                str(idt),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("data_type", idt.name)
        model.set_tensor_datatype(node.output[0], idt)

    def verify_node(self):
        raise NotImplementedError