# Copyright (c) 2020, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os

import numpy as np
from pyverilator import PyVerilator

from finn.core.datatype import DataType
from finn.custom_op.fpgadataflow import HLSCustomOp
from finn.custom_op.im2col import compute_conv_output_dim
from onnx import TensorProto, helper

# ONNX i/o tensor shape assumptions for ConvolutionInputGenerator:
# input 0 is the input tensor, shape NHWC = (1, IFMDim, IFMDim, IFMChannels)
# output 0 is the output tensor, shape NHWC:
#     = (1, OFMDim, OFMDim, (ConvKernelDim^2)*IFMChannels)


class ConvolutionInputGenerator(HLSCustomOp):
    """Class that corresponds to finn-hlslib ConvolutionInputGenerator
    (sliding window) function."""

    def __init__(self, onnx_node):
        super().__init__(onnx_node)

    def get_nodeattr_types(self):
        my_attrs = {
            "ConvKernelDim": ("i", True, 0),
            "IFMChannels": ("i", True, 0),
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

    def make_shape_compatible_op(self, model):
        k = self.get_nodeattr("ConvKernelDim")
        ifm_dim = self.get_nodeattr("IFMDim")
        ifm_ch = self.get_nodeattr("IFMChannels")
        stride = self.get_nodeattr("Stride")
        pad = 0
        exp_ishape = (1, ifm_dim, ifm_dim, ifm_ch)
        ishape = tuple(model.get_tensor_shape(self.onnx_node.input[0]))
        assert ishape == exp_ishape, "Unexpect input shape for ConvInpGen."
        ofm_dim = compute_conv_output_dim(ifm_dim, k, stride, pad)
        # implement tensor with correct shape
        values = np.random.randn(1, ofm_dim, ofm_dim, k * k * ifm_ch).astype(np.float32)
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

    def bram_estimation(self):
        pass

    def lut_estimation(self):
        pass

    def get_input_datatype(self):
        """Returns FINN DataType of input."""
        return DataType[self.get_nodeattr("inputDataType")]

    def get_output_datatype(self):
        """Returns FINN DataType of output."""
        return DataType[self.get_nodeattr("outputDataType")]

    def get_stream_width(self):
        """Returns stream width, input and output stream width are equal for
        the sliding window function"""
        ibits = self.get_input_datatype().bitwidth()
        return self.get_nodeattr("SIMD") * ibits

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
            assert str(inp.dtype) == "float32", "Input datatype is not float32"
            assert inp.shape == (
                1,
                ifm_ch,
                ifm_dim,
                ifm_dim,
            ), """Input shape doesn't
            match expected shape (1, ifm_ch, ifm_dim, ifm_dim)."""
            reshaped_inp = inp.transpose(0, 2, 3, 1)
            # make copy before saving array
            reshaped_inp = reshaped_inp.copy()
            np.save(os.path.join(code_gen_dir, "input_0.npy"), reshaped_inp)
            # execute the precompiled model
            super().exec_precompiled_singlenode_model()
            # load output npy file
            super().npy_to_dynamic_output(context)
            if self.get_output_datatype() == DataType.BIPOLAR:
                out = context[node.output[0]]
                out = 2 * out - 1
                context[node.output[0]] = out
            assert context[node.output[0]].shape == (
                1,
                out_pix,
                k * k,
                ifm_ch,
            ), """Output
            shape doesn't match expected shape (1, out_pix, k*k, ifm_ch)."""
            # reshape output to have expected shape
            context[node.output[0]] = context[node.output[0]].reshape(
                1, out_pix, k * k * ifm_ch
            )
        elif mode == "rtlsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
            prefixed_top_name = "%s_%s" % (node.name, node.name)
            # check if needed file exists
            verilog_file = "{}/project_{}/sol1/impl/verilog/{}.v".format(
                code_gen_dir, node.name, prefixed_top_name
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
                self.get_input_datatype().bitwidth(),
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
                node.op_type
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
