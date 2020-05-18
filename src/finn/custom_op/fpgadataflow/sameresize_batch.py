import numpy as np
from onnx import TensorProto, helper
from finn.core.datatype import DataType
from finn.custom_op.fpgadataflow import HLSCustomOp


class SameResize_Batch(HLSCustomOp):
    """Class that corresponds to finn-hlslib SameResize function.
    Implements 'same' padding on a given input image."""

    def __init__(self, onnx_node):
        super().__init__(onnx_node)

    def get_nodeattr_types(self):
        my_attrs = {
            "ImgDim": ("i", True, 0),
            "KernelDim": ("i", True, 0),
            "Stride": ("i", True, 0),
            "NumChannels": ("i", True, 0),
            # FINN input datatype
            "inputDataType": ("s", True, ""),
            # distribution of added values to achieve "same" padding
            "PaddingStyle": ("i", True, 2),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_normal_input_shape(self):
        idim = self.get_nodeattr("ImgDim")
        num_ch = self.get_nodeattr("NumChannels")

        ishape = (1, idim, idim, num_ch)
        return ishape

    def get_normal_output_shape(self):
        idim = self.get_nodeattr("ImgDim")
        num_ch = self.get_nodeattr("NumChannels")
        kdim = self.get_nodeattr("KernelDim")
        stride = self.get_nodeattr("Stride")
        assert idim % stride == 0, "Stride must divide input dimension."
        # number of "same" windows over the input data
        same_windows = idim // stride
        odim = kdim + stride * (same_windows - 1)

        oshape = (1, odim, odim, num_ch)
        return oshape

    def get_folded_input_shape(self):
        pass

    def get_folded_output_shape(self):
        pass

    def make_shape_compatible_op(self, model):
        exp_ishape = self.get_normal_input_shape()
        oshape = self.get_normal_output_shape()
        ishape = tuple(model.get_tensor_shape(self.onnx_node.input[0]))
        assert ishape == exp_ishape, "Unexpect input shape for SameResize."
        # implement tensor with correct shape
        values = np.random.randn(*oshape).astype(np.float32)
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

    def verify_node(self):
        pass

    def get_input_datatype(self):
        """Returns FINN DataType of input."""
        return DataType[self.get_nodeattr("inputDataType")]

    def get_output_datatype(self):
        """Returns FINN DataType of output. (Same as input datatype)"""
        return DataType[self.get_nodeattr("inputDataType")]

    def get_instream_width(self):
        ibits = self.get_input_datatype().bitwidth()
        num_ch = self.get_nodeattr("NumChannels")

        return ibits * num_ch

    def get_outstream_width(self):
        obits = self.get_output_datatype().bitwidth()
        num_ch = self.get_nodeattr("NumChannels")

        return obits * num_ch

    def get_number_output_values():
        pass

    def global_includes(self):
        pass

    def defines(self, var):
        pass

    def read_npy_data(self):
        pass

    def strm_decl(self):
        pass

    def docompute(self):
        pass

    def dataoutstrm(self):
        pass

    def save_as_npy(self):
        pass

    def blackboxfunction(self):
        pass

    def pragmas(self):
        pass
