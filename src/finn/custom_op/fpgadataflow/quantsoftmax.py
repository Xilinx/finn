
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.data_packing import numpy_to_hls_code, pack_innermost_dim_as_hex_string

class QuantSoftmax(HWCustomOp):
    """Abstraction layer for HW implementation of VectorVectorActivation layers."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            "img_dim": ("i", True, 0),
            "simd": ("i", False, 1),
            "channels": ("i", True, 0),
            # FINN DataTypes for inputs, weights, outputs
            "data_type": ("s", True, ""),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_normal_input_shape(self, ind=0):
        idim_h, idim_w = self.get_nodeattr("img_dim")
        num_ch = self.get_nodeattr("channels")
        ishape = (1, idim_h, idim_w, num_ch)
        return ishape

    def get_normal_output_shape(self, ind=0):
        return self.get_normal_input_shape()

    def get_number_output_values(self):
        raise NotImplementedError("This function is not yet implemented.")

    def execute_node(self, context, graph):
        raise NotImplementedError

    def get_number_output_values(self):
        raise NotImplementedError

    def get_nodeattr_types(self):
        raise NotImplementedError

    def make_shape_compatible_op(self, model):
        raise NotImplementedError

    def infer_node_datatype(self, model):
        raise NotImplementedError

    def verify_node(self):
        raise NotImplementedError