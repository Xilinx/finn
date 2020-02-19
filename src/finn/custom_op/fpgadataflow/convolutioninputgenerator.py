import os

import numpy as np
from pyverilator import PyVerilator

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

    def bram_estimation(self):
        pass

    def lut_estimation(self):
        pass

    def get_input_datatype(self):
        return DataType[self.get_nodeattr("inputDataType")]

    def get_output_datatype(self):
        return DataType[self.get_nodeattr("outputDataType")]

    def get_stream_width(self):
        return self.get_nodeattr("SIMD") * self.get_nodeattr("Input_precision")

    def get_number_output_values(self):
        k = self.get_nodeattr("ConvKernelDim")
        ifm_ch = self.get_nodeattr("IFMChannels")
        ofm_dim = self.get_nodeattr("OFMDim")
        out_pix = ofm_dim * ofm_dim

        return out_pix * k * k * ifm_ch

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node
        k = self.get_nodeattr("ConvKernelDim")
        ifm_dim = self.get_nodeattr("IFMDim")
        ifm_ch = self.get_nodeattr("IFMChannels")
        ofm_dim = self.get_nodeattr("OFMDim")
        out_pix = ofm_dim * ofm_dim

        if mode == "npysim":
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
        elif mode == "rtlsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
            # check if needed file exists
            verilog_file = "{}/project_{}/sol1/impl/verilog/{}.v".format(
                code_gen_dir, node.name, node.name
            )
            if os.path.isfile(verilog_file):
                inp = context[node.input[0]]
                inp = inp.transpose(0, 2, 3, 1)
                inp = inp.flatten()

                # TODO: check how to sort inputs for multichannel inputs
                # a = []
                # for i in range(len(inp)):
                #     if (i+1) % 2 == 0:
                #         a.append((int(inp[i-1]) << 1) + int(inp[i]))
                # inp = a
                sim = PyVerilator.build(
                    verilog_file,
                    verilog_path=[
                        "{}/project_{}/sol1/impl/verilog/".format(
                            code_gen_dir, node.name
                        )
                    ],
                )
                super().reset_rtlsim(sim)
                super().toggle_clk(sim)
                output = self.rtlsim(sim, inp)
                output = [int(x) for x in output]
                odt = self.get_output_datatype()
                if odt == DataType.BIPOLAR:
                    output = [2 * x - 1 for x in output]

                # pyverilator interprets int2 as uint2, so output has to be corrected
                elif odt == DataType.INT2:
                    mask = 2 ** (odt.bitwidth() - 1)
                    output = [-(x & mask) + (x & ~mask) for x in output]
                # TODO: check how to sort inputs for multichannel inputs
                # output = [bin(x)[2:].zfill(ifm_ch) for x in output]
                # output_ch1 = [int(x[:1]) for x in output]
                # output_ch2 = [int(x[1:]) for x in output]

                # reshape output
                output = np.asarray([output], dtype=np.float32).reshape(
                    1, out_pix, k * k * ifm_ch
                )
                context[node.output[0]] = output

            else:
                raise Exception(
                    """Found no verilog files for this node,
                    did you run the codegen_ipgen transformation?"""
                )
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("npysim", "rtlsim")""".format(
                    mode
                )
            )

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "slidingwindow.h"']

    def defines(self, var):
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
