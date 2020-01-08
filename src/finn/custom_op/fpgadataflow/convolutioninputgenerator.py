import os

import numpy as np

from finn.core.datatype import DataType
from finn.custom_op.fpgadataflow import HLSCustomOp


class ConvolutionInputGenerator(HLSCustomOp):
    def __init__(self, onnx_node):
        super().__init__(onnx_node)

    def get_nodeattr_types(self):
        my_attrs = {
            "ConvKernelDim": ("i", True, 0),
            "IFMChannels": ("i", True, 0),
            "Input_precision": ("i", True, 0),
            "IFMDim": ("i", True, 0),
            "OFMDim": ("i", True, 0),
            "SIMD": ("i", True, 0),
            "Stride": ("i", True, 0),
            # FINN DataTypes for inputs, weights, outputs
            "inputDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def make_shape_compatible_op(self):
        pass

    def infer_node_datatype(self, model):
        pass

    def verify_node(self):
        pass

    def get_input_datatype(self):
        return DataType[self.get_nodeattr("inputDataType")]

    def get_output_datatype(self):
        return DataType[self.get_nodeattr("outputDataType")]

    def get_stream_width(self):
        return self.get_nodeattr("SIMD") * self.get_nodeattr("Input_precision")

    def execute_node(self, context, graph):
        node = self.onnx_node
        k = self.get_nodeattr("ConvKernelDim")
        ifm_dim = self.get_nodeattr("IFMDim")
        ifm_ch = self.get_nodeattr("IFMChannels")
        ofm_dim = self.get_nodeattr("OFMDim")
        out_pix = ofm_dim * ofm_dim
        idt = self.get_input_datatype()
        if idt == DataType.BIPOLAR:
            # use binary for bipolar storage
            idt = DataType.BINARY

        # TODO ensure codegen dir exists
        code_gen_dir = self.get_nodeattr("code_gen_dir_npysim")
        # create a npy file for input of the node

        inp = context[node.input[0]]
        assert str(inp.dtype) == "float32"
        assert inp.shape == (1, ifm_ch, ifm_dim, ifm_dim)
        reshaped_inp = inp.transpose(0, 2, 3, 1)
        np.save(os.path.join(code_gen_dir, "input_0.npy"), reshaped_inp)
        # execute the precompiled model
        super().exec_precompiled_singlenode_model()
        # load output npy file
        super().npy_to_dynamic_output(context)
        if self.get_output_datatype() == DataType.BIPOLAR:
            out = context[node.output[0]]
            out = 2 * out - 1
            context[node.output[0]] = out
        assert context[node.output[0]].shape == (1, out_pix, k * k, ifm_ch)
        # reshape output to have expected shape
        context[node.output[0]] = context[node.output[0]].reshape(
            1, out_pix, k * k * ifm_ch
        )

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "slidingwindow.h"']

    def defines(self):
        numReps = 1
        self.code_gen_dict["$DEFINES$"] = [
            """#define ConvKernelDim1 {}\n #define IFMChannels1 {}
            #define Input_precision1 {}\n #define IFMDim1 {}\n #define OFMDim1 {}
            #define SIMD1 {}\n #define Stride1 {}\n #define numReps {}""".format(
                self.get_nodeattr("ConvKernelDim"),
                self.get_nodeattr("IFMChannels"),
                self.get_nodeattr("Input_precision"),
                self.get_nodeattr("IFMDim"),
                self.get_nodeattr("OFMDim"),
                self.get_nodeattr("SIMD"),
                self.get_nodeattr("Stride"),
                numReps,
            )
        ]

    def read_npy_data(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_npysim")
        dtype = self.get_input_datatype()
        if dtype == DataType.BIPOLAR:
            # use binary for bipolar storage
            dtype = DataType.BINARY
        elem_bits = dtype.bitwidth()
        packed_bits = self.get_stream_width()
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
            'hls::stream<ap_uint<{}>> in0 ("in0");'.format(self.get_stream_width())
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> out ("out");'.format(self.get_stream_width())
        )

    def docompute(self):
        node = self.onnx_node
        self.code_gen_dict["$DOCOMPUTE$"] = [
            """{}<ConvKernelDim1, IFMChannels1, Input_precision1, IFMDim1,
                OFMDim1, SIMD1, Stride1> (in0, out, numReps);""".format(
                node.op_type,
            )
        ]

    def dataoutstrm(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_npysim")
        dtype = self.get_output_datatype()
        if dtype == DataType.BIPOLAR:
            # use binary for bipolar storage
            dtype = DataType.BINARY
        elem_bits = dtype.bitwidth()
        packed_bits = self.get_stream_width()
        packed_hls_type = "ap_uint<%d>" % packed_bits
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"
        npy_out = "%s/output.npy" % code_gen_dir
        ofm_dim = self.get_nodeattr("OFMDim")
        out_pix = ofm_dim * ofm_dim
        k = self.get_nodeattr("ConvKernelDim")
        ifm_ch = self.get_nodeattr("IFMChannels")
        shape = (1, out_pix, k * k, ifm_ch)
        shape_cpp_str = str(shape).replace("(", "{").replace(")", "}")

        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            'apintstream2npy<%s, %s, %d, %s>(out, %s, "%s");'
            % (
                packed_hls_type,
                elem_hls_type,
                elem_bits,
                npy_type,
                shape_cpp_str,
                npy_out,
            )
        ]

    def save_as_npy(self):
        self.code_gen_dict["$SAVEASCNPY$"] = []

    def blackboxfunction(self):
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            """void {}(hls::stream<ap_uint<SIMD1*Input_precision1>> &in0,
                hls::stream<ap_uint<SIMD1*Input_precision1>> &out)""".format(
                self.onnx_node.name
            )
        ]

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"] = ["#pragma HLS INTERFACE axis port=in0"]
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=out")
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE ap_ctrl_none port=return"
        )
