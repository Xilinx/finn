# Copyright (C) 2024, Advanced Micro Devices, Inc.
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

import numpy as np
import os
from qonnx.core.datatype import DataType
from qonnx.util.basic import interleave_matrix_outer_dim_from_partitions

from finn.custom_op.fpgadataflow.deconvolution import Deconvolution
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.util.data_packing import (
    npy_to_rtlsim_input,
    numpy_to_hls_code1,
    rtlsim_output_to_npy,
)


class Deconvolution_hls(Deconvolution, HLSBackend):
    """Corresponds to finn-hlslib deconv function."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(Deconvolution.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def calc_wmem(self):
        """Calculates and returns WMEM."""
        ifm_ch = self.get_nodeattr("IFMChannels")
        ofm_ch = self.get_nodeattr("OFMChannels")
        kernel_2 = np.prod(self.get_nodeattr("KernelDim"))
        pe = self.get_nodeattr("PE")
        simd = self.get_nodeattr("SIMD")
        assert ofm_ch % pe == 0, "Requirement output channels divisable by PE is violated."
        assert ifm_ch % simd == 0, "Requirement input channels divisable by SIMD is violated."
        wmem = (ofm_ch / pe) * kernel_2 * (ifm_ch / simd)
        return int(wmem)

    def generate_params(self, model, path):
        code_gen_dir = path
        # weights, if not external
        weights = model.get_initializer(self.onnx_node.input[1])
        # save hlslib-compatible weights in params.h
        weight_filename = "{}/params.h".format(code_gen_dir)
        self.make_weight_file(weights, "hls_header", weight_filename)

    def make_weight_file(self, weights, weight_file_mode, weight_file_name):
        """Produce a file containing given weights in appropriate format for this
        layer. This file can be used for either synthesis or run-time reconfig
        of weights.

        Arguments:

        * weights : numpy array with weights to be put into the file
        * weight_file_mode : one of {hls_header, decoupled_verilog_dat,
          decoupled_runtime}
        * weight_file_name : filename for the weight file to be generated

        """
        # convert weights into hlslib-compatible format
        weight_tensor = self.get_hw_compatible_weight_tensor(weights)
        export_wdt = self.get_weight_datatype()
        if weight_file_mode == "hls_header":
            weight_hls_code = numpy_to_hls_code1(weight_tensor, export_wdt, "weights", False, True)
            # write weights into C++ header file as dictated by finn-hlslib
            f_weights = open(weight_file_name, "w")
            f_weights.write(
                "static {} const weights[{}][{}][{}] = ".format(
                    export_wdt.get_hls_datatype_str(),
                    self.calc_wmem(),
                    self.get_nodeattr("PE"),
                    self.get_nodeattr("SIMD"),
                )
            )
            f_weights.write(weight_hls_code)
            f_weights.close()

    def get_hw_compatible_weight_tensor(self, orig_weight_matrix):
        """Convert the original numpy weight matrix orig_weight_matrix into
        a form suitable for passing to the hlslib call:
        * ensure OCH % PE == 0 and ICH % SIMD == 0
        * interleave rows between PEs
        * reshape into (1, PE, WMEM, SIMD) and return
        """
        k_h, k_w = self.get_nodeattr("KernelDim")
        ifm_ch = self.get_nodeattr("IFMChannels")
        ofm_ch = self.get_nodeattr("OFMChannels")
        pe = self.get_nodeattr("PE")
        simd = self.get_nodeattr("SIMD")
        wmem = self.calc_wmem()
        assert orig_weight_matrix.shape == (
            ofm_ch, k_h, k_w, ifm_ch
        ), """Weights matrix doesn't
        have expected shape (ofm_ch, k_h, k_w, ifm_ch)"""
        assert ofm_ch % pe == 0, "Requirement output channels divisable by PE is violated."
        assert ifm_ch % simd == 0, "Requirement input channels divisable by SIMD is violated."
        # interleave rows between PEs and reshape
        # distribute rows between PEs
        ret = orig_weight_matrix
        ret = ret.reshape(ofm_ch, k_h * k_w * ifm_ch)
        ret = interleave_matrix_outer_dim_from_partitions(ret, pe)
        # create SIMD as innermost dimension
        ret = ret.reshape(1, pe, wmem, simd)
        ret = ret.transpose(0, 2, 1, 3)
        return ret

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "deconv.hpp"']

    def defines(self, var):
        ifm_dim = self.get_nodeattr("IFMDim")
        self.code_gen_dict["$DEFINES$"] = [
            """constexpr unsigned Kernel = {};\n constexpr unsigned Stride = {};\n
            constexpr unsigned Padding = {};\n constexpr unsigned IFMH = {};\n
            constexpr unsigned IFMW = {};\n constexpr unsigned ICH = {};\n
            constexpr unsigned OCH = {};\n constexpr unsigned SIMD1 = {};\n
            constexpr unsigned PE1 = {};""".format(
                self.get_nodeattr("KernelDim")[0],
                self.get_nodeattr("Stride")[0],
                self.get_nodeattr("Padding")[0],
                ifm_dim[0],
                ifm_dim[1],
                self.get_nodeattr("IFMChannels"),
                self.get_nodeattr("OFMChannels"),
                self.get_nodeattr("SIMD"),
                self.get_nodeattr("PE"),
            )
        ]

    def read_npy_data(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_input_datatype()
        elem_hls_type = dtype.get_hls_datatype_str()
        simd = self.get_nodeattr("SIMD")
        npy_type = "float"
        npy_in = "%s/input_0.npy" % code_gen_dir
        self.code_gen_dict["$READNPYDATA$"] = []
        # note: the innermost dim is reversed for the input
        self.code_gen_dict["$READNPYDATA$"].append(
            'npy2vectorstream<%s, %s, %d>("%s", in0_%s, false);'
            % (
                elem_hls_type,
                npy_type,
                simd,
                npy_in,
                self.hls_sname(),
            )
        )

    def strm_decl(self):
        idtype = self.get_input_datatype()
        odtype = self.get_output_datatype()
        simd = self.get_nodeattr("SIMD")
        pe = self.get_nodeattr("PE")
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<hls::vector<{},{}>> in0_{} ("in0_{}");'.format(
                idtype.get_hls_datatype_str(), simd, self.hls_sname(), self.hls_sname()
            )
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<hls::vector<{},{}>> out_{} ("out_{}");'.format(
                odtype.get_hls_datatype_str(), pe, self.hls_sname(), self.hls_sname()
            )
        )

    def docompute(self):
        odtype = self.get_output_datatype()
        pe = self.get_nodeattr("PE")
        ishape = self.get_normal_input_shape()
        oshape = self.get_normal_output_shape()
        self.code_gen_dict["$DOCOMPUTE$"] = [
            "hls::stream<hls::vector<{},{}>> strm;".format(odtype.get_hls_datatype_str(), pe)
        ]
        self.code_gen_dict["$DOCOMPUTE$"].append("unsigned  timeout = 0;")
        self.code_gen_dict["$DOCOMPUTE$"].append("while(timeout < %s) {" % (2 * np.prod(oshape)))
        self.code_gen_dict["$DOCOMPUTE$"].append(
            """deconv<Kernel, Stride, Padding, IFMH, IFMW, OCH, ICH, PE1, SIMD1>
            (weights, in0_{}, out_{});""".format(
                self.hls_sname(),
                self.hls_sname(),
            )
        )
        self.code_gen_dict["$DOCOMPUTE$"].append("if(out_V.empty())  timeout++;")
        self.code_gen_dict["$DOCOMPUTE$"].append("else {")
        self.code_gen_dict["$DOCOMPUTE$"].append("strm << out_V.read();")
        self.code_gen_dict["$DOCOMPUTE$"].append("timeout = 0;")
        self.code_gen_dict["$DOCOMPUTE$"].append("}")
        self.code_gen_dict["$DOCOMPUTE$"].append("}")

    def dataoutstrm(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        pe = self.get_nodeattr("PE")
        dtype = self.get_output_datatype()
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"
        npy_out = "%s/output.npy" % code_gen_dir
        shape = self.get_folded_output_shape()
        shape_cpp_str = str(shape).replace("(", "{").replace(")", "}")

        # note: the innermost dim is not reversed for the output
        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            'vectorstream2npy<%s, %s, %d>(strm, %s, "%s", false);'
            % (
                elem_hls_type,
                npy_type,
                pe,
                shape_cpp_str,
                npy_out,
            )
        ]

    def blackboxfunction(self):
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            """void {}(hls::stream<ap_uint<{}>> &in0_{},
                hls::stream<ap_uint<{}>> &out_{}
                )""".format(
                self.onnx_node.name,
                self.get_instream_width(),
                self.hls_sname(),
                self.get_outstream_width(),
                self.hls_sname(),
            )
        ]

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"] = [
            "#pragma HLS INTERFACE axis port=in0_" + self.hls_sname()
        ]
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE axis port=out_" + self.hls_sname()
        )
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE ap_ctrl_none port=return")

        self.code_gen_dict["$PRAGMAS$"].append('#include "params.h"')
        # the weight tensor is ap_uint<simd*prec> [PE][WMEM]
        # partition for parallel access along the PE dimension (dim 1)
        # self.code_gen_dict["$PRAGMAS$"].append(
        #    ("#pragma HLS ARRAY_PARTITION variable=weights.m_weights " "complete dim=1")
        # )

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node

        # TODO ensure codegen dir exists
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

        # create a npy file fore each input of the node (in_ind is input index)
        in_ind = 0
        for inputs in node.input:
            # it is assumed that the first input of the node is the data input
            # the second input are the weights
            # the third input are the thresholds
            if in_ind == 0:
                assert (
                    str(context[inputs].dtype) == "float32"
                ), """Input datatype is
                not float32 as expected."""
                expected_inp_shape = self.get_folded_input_shape()
                reshaped_input = context[inputs].reshape(expected_inp_shape)
                if self.get_input_datatype() == DataType["BIPOLAR"]:
                    # store bipolar activations as binary
                    reshaped_input = (reshaped_input + 1) / 2
                    export_idt = DataType["BINARY"]
                else:
                    export_idt = self.get_input_datatype()
                # make copy before saving the array
                reshaped_input = reshaped_input.copy()
                np.save(
                    os.path.join(code_gen_dir, "input_{}.npy".format(in_ind)),
                    reshaped_input,
                )
            elif in_ind > 2:
                raise Exception("Unexpected input found for MatrixVectorActivation")
            in_ind += 1

        if mode == "cppsim":
            # execute the precompiled model
            super().exec_precompiled_singlenode_model()
            # load output npy file
            super().npy_to_dynamic_output(context)
            # reinterpret binary output as bipolar where needed
            if self.get_output_datatype() == DataType["BIPOLAR"]:
                out = context[node.output[0]]
                out = 2 * out - 1
                context[node.output[0]] = out
            assert (
                context[node.output[0]].shape == self.get_normal_output_shape()
            ), "cppsim did not produce expected output shape"
        elif mode == "rtlsim":
            sim = self.get_rtlsim()
            nbits = self.get_instream_width()
            inp = npy_to_rtlsim_input("{}/input_0.npy".format(code_gen_dir), export_idt, nbits)
            self.reset_rtlsim(sim)
            self.toggle_clk(sim)
            output = self.rtlsim(sim, inp)
            odt = self.get_output_datatype()
            target_bits = odt.bitwidth()
            packed_bits = self.get_outstream_width()
            out_npy_path = "{}/output.npy".format(code_gen_dir)
            out_shape = self.get_folded_output_shape()
            rtlsim_output_to_npy(output, out_npy_path, odt, out_shape, packed_bits, target_bits)

            # load and reshape output
            output = np.load(out_npy_path)
            oshape = self.get_normal_output_shape()
            output = np.asarray([output], dtype=np.float32).reshape(*oshape)
            context[node.output[0]] = output
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )
