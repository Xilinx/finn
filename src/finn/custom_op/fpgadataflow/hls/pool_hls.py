# Copyright (C) 2023, Advanced Micro Devices, Inc.
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

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.pool import Pool
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy


class Pool_hls(Pool, HLSBackend):
    """Class that corresponds to finn-hlslib Pool_batch function.
    Requires ConvolutionInputGenerator(depthwise == 1) to format its input

    Input shape (BatchSize,OutImgDim,OutImgDim,TotalKernelSize*Channels)
    Output shape (BatchSize,OutImgDim,OutImgDim,Channels)

    Notes:

    * The input shape was chosen to be compatible with im2col (only true when there
      is not folding).
    * The actual data layout produced by the hlslib kernels is different
      for depthwise ops.

        * depthwise SWG: (1, OFMDim, OFMDim, IFMChannels/PE, K, K, PE)

    Channels can be folded using PE (SIMD from the input perspective)
    """

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(Pool.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "pool.hpp"']

    def defines(self, var):
        k = int(np.prod(self.get_nodeattr("KernelSize")))
        cf = int(self.get_nodeattr("Channels") / self.get_nodeattr("PE"))
        osz = np.prod(self.get_nodeattr("OutImgDims"))
        self.code_gen_dict["$DEFINES$"] = [
            "constexpr unsigned  ISIZE = {};".format(osz * cf * k),
            "constexpr unsigned  K = {};".format(k)
        ]

    def read_npy_data(self):
        dtype = self.get_input_datatype()
        if dtype == DataType["BIPOLAR"]:
            # use binary for bipolar storage
            dtype = DataType["BINARY"]

        self.code_gen_dict["$READNPYDATA$"] = [
            'npy2vectorstream<%s, float, %d>("%s/input_0.npy", in0_%s, false);'
            % (
                dtype.get_hls_datatype_str(), self.get_nodeattr("PE"),
                self.get_nodeattr("code_gen_dir_cppsim"), self.hls_sname()
            )
        ]

    def docompute(self):
        pe  = self.get_nodeattr("PE")
        fxn = self.get_nodeattr("Function")
        idt = self.get_input_datatype()
        odt = self.get_output_datatype()
        i_hls_dt = 'hls::vector<%s, %d>' % (idt.get_hls_datatype_str(), pe)
        o_hls_dt = 'hls::vector<%s, %d>' % (odt.get_hls_datatype_str(), pe)

        self.code_gen_dict["$DOCOMPUTE$"] = []
        if fxn == "MaxPool":
            self.code_gen_dict["$DOCOMPUTE$"] += [
                "MaxPoolFunction<{}> pool_fxn;".format(o_hls_dt)
            ]
        elif fxn == "QuantAvgPool":
            shift      = self.get_nodeattr("Size")
            accum_bits = self.get_nodeattr("AccumBits")
            act_hls_dt = 'hls::vector<ap_%sint<%d>, %d>' % ('' if idt.signed() else 'u', accum_bits, pe)
            self.code_gen_dict["$DOCOMPUTE$"] += [
                "QuantAvgPoolFunction<{},{},{}> pool_fxn;".format(o_hls_dt, act_hls_dt, shift)
            ]
        else:
            raise Exception("Pool_Batch doesn't currently support " + fxn)

        self.code_gen_dict["$DOCOMPUTE$"] += [
            'Pool_batch<ISIZE, K>(in0_{}, out_{}, pool_fxn);'.format(
                self.hls_sname(), self.hls_sname()
            )
        ]

    def dataoutstrm(self):
        dtype = self.get_output_datatype()
        if dtype == DataType["BIPOLAR"]:
            # use binary for bipolar storage
            dtype = DataType["BINARY"]
        oshape = str(self.get_folded_output_shape()).replace("(", "{").replace(")", "}")

        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            'vectorstream2npy<%s, float, %d>(out_%s, %s, "%s/output.npy", false);'
            % (
                dtype.get_hls_datatype_str(), self.get_nodeattr("PE"),
                self.hls_sname(), oshape, self.get_nodeattr("code_gen_dir_cppsim")
            )
        ]

    def pragmas(self):
        super().pragmas()
        self.code_gen_dict["$PRAGMAS$"].append(
            f"#pragma HLS aggregate variable=in0_{self.hls_sname()} compact=bit"
        )

        # TODO : This will need to be different with latest refactoring changes.
        self.code_gen_dict["$PRAGMAS$"].append(
            f"#pragma HLS aggregate variable=out_{self.hls_sname()} compact=bit"
        )

    def blackboxfunction(self):
        pe  = self.get_nodeattr("PE")
        idt = self.get_input_datatype()
        odt = self.get_output_datatype()
        i_hls_dt = 'hls::vector<%s, %d>' % (idt.get_hls_datatype_str(), pe)
        o_hls_dt = 'hls::vector<%s, %d>' % (odt.get_hls_datatype_str(), pe)

        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            "void %s(hls::stream<%s> &in0_%s, hls::stream<%s> &out_%s)"
            % (
                self.onnx_node.name,
                i_hls_dt, self.hls_sname(),
                o_hls_dt, self.hls_sname()
            )
        ]

    def strm_decl(self):
        """Function to generate the commands for the stream declaration in c++,
        is member function of HLSBackend class but might need to be filled
        by node."""
        dtype = self.get_input_datatype()
        if dtype == DataType["BIPOLAR"]:
            # use binary for bipolar storage
            dtype = DataType["BINARY"]
        elem_input_hls_type = dtype.get_hls_datatype_str()
		
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<hls::vector<{},{}>> in0_{} ("in0_{}");'.format(
               elem_input_hls_type, self.get_folded_input_shape()[-1], self.hls_sname(), self.hls_sname()
            )
        )

        dtype = self.get_output_datatype()
        if dtype == DataType["BIPOLAR"]:
            # use binary for bipolar storage
            dtype = DataType["BINARY"]
        elem_output_hls_type = dtype.get_hls_datatype_str()

        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<hls::vector<{},{}>> out_{} ("out_{}");'.format(
               elem_output_hls_type, self.get_folded_output_shape()[-1], self.hls_sname(), self.hls_sname()
            )
        )


    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node
        exp_ishape = self.get_normal_input_shape()
        folded_ishape = self.get_folded_input_shape()
        exp_oshape = self.get_normal_output_shape()

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

        inp = context[node.input[0]]

        assert str(inp.dtype) == "float32", "Input datatype is not float32"
        assert (
            inp.shape == exp_ishape
        ), """Input shape doesn't
        match expected shape (batch_size,odim,odim,k*k*ifm_ch)."""

        export_idt = self.get_input_datatype()
        reshaped_input = inp.reshape(folded_ishape)

        np.save(os.path.join(code_gen_dir, "input_0.npy"), reshaped_input)

        if mode == "cppsim":
            # execute the precompiled model
            super().exec_precompiled_singlenode_model()
            # load output npy file
            super().npy_to_dynamic_output(context)
            assert (
                context[node.output[0]].shape == exp_oshape
            ), "cppsim did not produce expected output shape"
        elif mode == "rtlsim":
            sim = self.get_rtlsim()
            nbits = self.get_instream_width()
            rtlsim_inp = npy_to_rtlsim_input(
                "{}/input_0.npy".format(code_gen_dir), export_idt, nbits
            )
            super().reset_rtlsim(sim)
            io_dict = {
                "inputs": {"in0": rtlsim_inp},
                "outputs": {"out": []},
            }
            self.rtlsim_multi_io(sim, io_dict)
            super().close_rtlsim(sim)
            rtlsim_output = io_dict["outputs"]["out"]
            odt = self.get_output_datatype()
            target_bits = odt.bitwidth()
            packed_bits = self.get_outstream_width()
            out_npy_path = "{}/output.npy".format(code_gen_dir)
            out_shape = self.get_folded_output_shape()
            rtlsim_output_to_npy(
                rtlsim_output, out_npy_path, odt, out_shape, packed_bits, target_bits
            )
            # load and reshape output
            output = np.load(out_npy_path)
            output = np.asarray([output], dtype=np.float32).reshape(*exp_oshape)
            context[node.output[0]] = output
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )

        assert (
            context[node.output[0]].shape == exp_oshape
        ), """Output
        shape doesn't match expected shape (1, ofm_dim, ofm_dim, k*k*ifm_ch)."""
