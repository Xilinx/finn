import os
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
        # even though there is no folding in the current hlslib op,
        # insert a time multiplexing axis to remain compatible with the
        # shapes produced by the rest of the dataflow pipeline
        ret = list(self.get_normal_input_shape())
        ret.insert(-1, 1)
        return tuple(ret)

    def get_folded_output_shape(self):
        # even though there is no folding in the current hlslib op,
        # insert a time multiplexing axis to remain compatible with the
        # shapes produced by the rest of the dataflow pipeline
        ret = list(self.get_normal_output_shape())
        ret.insert(-1, 1)
        return tuple(ret)

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

    def get_number_output_values(self):
        oshape = self.get_normal_output_shape()
        return np.prod(oshape)

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "streamtools.h"']

    def defines(self, var):
        numReps = 1
        self.code_gen_dict["$DEFINES$"] = [
            """#define ImgDim1 {}\n #define KernelDim1 {}\n
            #define Stride1 {}\n #define NumChannels1 {}\n
            #define PaddingStyle1 {}\n #define numReps {}""".format(
                self.get_nodeattr("ImgDim"),
                self.get_nodeattr("KernelDim"),
                self.get_nodeattr("Stride"),
                self.get_nodeattr("NumChannels"),
                self.get_nodeattr("PaddingStyle"),
                numReps,
            )
        ]

    def read_npy_data(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_input_datatype()
        if dtype == DataType.BIPOLAR:
            # use binary for bipolar storage
            dtype = DataType.BINARY
        elem_bits = dtype.bitwidth()
        packed_bits = self.get_instream_width()
        packed_hls_type = "ap_uint<%d>" % packed_bits
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"
        npy_in = "%s/input_0.npy" % code_gen_dir
        self.code_gen_dict["$READNPYDATA$"] = []
        self.code_gen_dict["$READNPYDATA$"].append(
            'npy2apintstream<%s, %s, %d, %s>("%s", in0);'
            % (packed_hls_type, elem_hls_type, elem_bits, npy_type, npy_in)
        )

    def strm_decl(self):
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> in0 ("in0");'.format(self.get_instream_width())
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> out ("out");'.format(self.get_outstream_width())
        )

    def docompute(self):
        in_t = self.get_input_datatype().get_hls_datatype_str()
        node = self.onnx_node
        self.code_gen_dict["$DOCOMPUTE$"] = [
            """{}<ImgDim1, KernelDim1, Stride1, NumChannels1,
                {}, PaddingStyle1> (in0, out, numReps);""".format(
                node.op_type, in_t
            )
        ]

    def dataoutstrm(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_output_datatype()
        if dtype == DataType.BIPOLAR:
            # use binary for bipolar storage
            dtype = DataType.BINARY
        elem_bits = dtype.bitwidth()
        packed_bits = self.get_outstream_width()
        packed_hls_type = "ap_uint<%d>" % packed_bits
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"
        npy_out = "%s/output.npy" % code_gen_dir
        oshape = self.get_folded_output_shape()
        oshape_cpp_str = str(oshape).replace("(", "{").replace(")", "}")

        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            'apintstream2npy<%s, %s, %d, %s>(out, %s, "%s");'
            % (
                packed_hls_type,
                elem_hls_type,
                elem_bits,
                npy_type,
                oshape_cpp_str,
                npy_out,
            )
        ]

    def save_as_npy(self):
        self.code_gen_dict["$SAVEASCNPY$"] = []

    def blackboxfunction(self):
        pass

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"] = ["#pragma HLS INTERFACE axis port=in0"]
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=out")
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE ap_ctrl_none port=return"
        )

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node
        exp_ishape = self.get_normal_input_shape()
        exp_oshape = self.get_normal_output_shape()
        folded_oshape = self.get_folded_output_shape()

        if mode == "cppsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        elif mode == "rtlsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )

        inp = context[node.input[0]]
        assert str(inp.dtype) == "float32", "Input datatype is not float32"
        assert (
            inp.shape == exp_ishape
        ), """Input shape doesn't
        match expected shape (1, ImgDim, ImgDim, NumChannels)."""
        if self.get_input_datatype() == DataType.BIPOLAR:
            # store bipolar activations as binary
            inp = (inp + 1) / 2
            # export_idt = DataType.BINARY
        # else:
        # export_idt = self.get_input_datatype()

        # no reshaping for input since assuming no folding on input
        # make copy before saving array
        inp = inp.copy()
        np.save(os.path.join(code_gen_dir, "input_0.npy"), inp)

        if mode == "cppsim":
            # execute the precompiled model
            super().exec_precompiled_singlenode_model()
            # load output npy file
            super().npy_to_dynamic_output(context)
            assert (
                context[node.output[0]].shape == folded_oshape
            ), "cppsim \
            did not produce expected ofolded utput shape"
            context[node.output[0]] = context[node.output[0]].reshape(*exp_oshape)
