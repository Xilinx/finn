# import os

# import numpy as np

# from finn.backend.fpgadataflow.utils import numpy_to_hls_code
from finn.core.datatype import DataType
# from finn.core.utils import interleave_matrix_outer_dim_from_partitions
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

    def get_instream_width(self):
        return self.get_nodeattr("IFMDim") * self.get_nodeattr("Input_precision")

    def get_outstream_width(self):
        o_bits = self.get_output_datatype().bitwidth()
        return self.get_nodeattr("OFMDim")

    def execute_node(self, context, graph):
        pass

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
        code_gen_dir = self.get_nodeattr("code_gen_dir")
        dtype = self.get_input_datatype()
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
        node = self.onnx_node
        self.code_gen_dict["$DOCOMPUTE$"] = [
            """{}<ConvKernelDim1, IFMChannels1, Input_precision1, IFMDim1,
                OFMDim1, SIMD1, Stride1> (in0, out, numReps);""".format(
                node.op_type,
            )
        ]

    def dataoutstrm(self):
        pass

    def save_as_npy(self):
        self.code_gen_dict["$SAVEASCNPY$"] = []
