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

from finn.custom_op.fpgadataflow import HLSCustomOp
from finn.core.datatype import DataType
from onnx import TensorProto, helper
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy


class Pool_Batch(HLSCustomOp):
    """Class that corresponds to finn-hlslib Pool_batch function.
    Requires ConvolutionInputGenerator(depthwise == 1) to format its input

    Input shape (BatchSize,OutImgDim,OutImgDim,KernelSize^2*Channels)
    Output shape (BatchSize,OutImgDim,OutImgDim,Channels)

    Notes:
    # The input shape was chosen to be compatible with im2col (only true when there
    is not folding).

    # The actual data layout produced by the hlslib kernels is different
    for depthwise ops.
     * depthwise SWG: (1, OFMDim, OFMDim, IFMChannels/PE, K, K, PE)

    Channels can be folded using PE (SIMD from the input perspective)
    """

    def get_nodeattr_types(self):
        my_attrs = {
            "Channels": ("i", True, 0),
            "PE": ("i", True, 1),
            "KernelSize": ("i", True, 0),
            # Function:
            #  - MaxPool
            #  - AvgPool (not yet supported, but HLSLIB does)
            #  - AccPool (not yet supported, but HLSLIB does)
            "Function": ("s", True, ""),
            "OutImgDim": ("i", True, 0),
            # FINN DataTypes for inputs/outputs
            "InputDataType": ("s", True, ""),
            "OutputDataType": ("s", True, ""),
            "AccumBits": ("i", False, 0),
            "Size": ("i", False, 1),
            "BatchSize": ("i", False, 1),
        }

        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_input_datatype(self):
        """Returns FINN DataType of input."""
        return DataType[self.get_nodeattr("InputDataType")]

    def get_output_datatype(self):
        """Returns FINN DataType of output."""
        fxn = self.get_nodeattr("Function")
        odt = DataType[self.get_nodeattr("OutputDataType")]

        if fxn == "MaxPool":
            # Same as input
            idt = DataType[self.get_nodeattr("InputDataType")]
            assert odt == idt, "In datatype must be equal to out datatype for Maxpool"
        elif fxn == "QuantAvgPool":
            idt = DataType[self.get_nodeattr("InputDataType")]
            assert (
                idt.signed() == odt.signed()
            ), """QuantAvgPool: Can't mix signed
            and unsigned datatypes"""
        else:
            raise Exception("Pool_Batch doesn't currently support " + fxn)

        return odt

    def get_normal_input_shape(self):
        ifm_ch = self.get_nodeattr("Channels")
        odim = self.get_nodeattr("OutImgDim")
        batch_size = self.get_nodeattr("BatchSize")
        k = self.get_nodeattr("KernelSize")
        ishape = (batch_size, odim, odim, k * k * ifm_ch)
        return ishape

    def get_folded_input_shape(self):
        normal_ishape = list(self.get_normal_input_shape())
        ifm_ch = self.get_nodeattr("Channels")
        pe = self.get_nodeattr("PE")
        assert ifm_ch % pe == 0, "PE must divide input channels"
        fold = int(normal_ishape[-1] / pe)
        folded_ishape = normal_ishape[:-1] + [fold, pe]
        return tuple(folded_ishape)

    def get_normal_output_shape(self):
        ofm_ch = self.get_nodeattr("Channels")
        odim = self.get_nodeattr("OutImgDim")
        batch_size = self.get_nodeattr("BatchSize")
        oshape = (batch_size, odim, odim, ofm_ch)
        return oshape

    def get_folded_output_shape(self):
        normal_oshape = list(self.get_normal_output_shape())
        ifm_ch = self.get_nodeattr("Channels")
        pe = self.get_nodeattr("PE")
        assert ifm_ch % pe == 0, "PE must divide input channels"
        fold = int(ifm_ch / pe)
        folded_oshape = normal_oshape[:-1] + [fold, pe]
        return tuple(folded_oshape)

    def get_number_output_values(self):
        folded_oshape = self.get_folded_output_shape()
        return np.prod(folded_oshape[1:-1])

    def get_exp_cycles(self):
        # Channels/PE * batch size * odim * odim
        return np.prod(self.get_folded_output_shape()[:-1])

    def get_instream_width(self):
        dt_bits = self.get_input_datatype().bitwidth()
        pe = self.get_nodeattr("PE")
        in_width = int(dt_bits * pe)
        return in_width

    def get_outstream_width(self):
        dt_bits = self.get_output_datatype().bitwidth()
        pe = self.get_nodeattr("PE")
        out_width = int(dt_bits * pe)
        return out_width

    def make_shape_compatible_op(self, model):
        exp_ishape = self.get_normal_input_shape()
        oshape = self.get_normal_output_shape()
        ishape = tuple(model.get_tensor_shape(self.onnx_node.input[0]))
        assert ishape == exp_ishape, "Unexpected input shape for Pool_Batch."
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
        dtype = self.get_output_datatype()
        model.set_tensor_datatype(node.output[0], dtype)

    def verify_node(self):
        info_messages = []

        # verify that "domain" is set to "finn"
        domain_value = self.onnx_node.domain
        if domain_value == "finn":
            info_messages.append("Attribute domain is set correctly")
        else:
            info_messages.append('Attribute domain should be set to "finn"')

        # verify that "backend" is set to "fpgadataflow"
        backend_value = self.get_nodeattr("backend")
        if backend_value == "fpgadataflow":
            info_messages.append("Attribute backend is set correctly")
        else:
            info_messages.append('Attribute backend should be set to "fpgadataflow"')

        # verify the number of inputs
        if len(self.onnx_node.input) == 1:
            info_messages.append("The number of inputs is correct")
        else:
            info_messages.append("""Pool_Batch needs 1 data input""")

        # check supported function
        fnx = self.get_nodeattr("Function")
        if fnx in ["MaxPool", "QuantAvgPool"]:
            info_messages.append(
                "Attribute Function contains a supported pool function"
            )
        else:
            info_messages.append(
                "Attribute Function contains an unsupported pool function"
            )
        return info_messages

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "maxpool.h"']
        self.code_gen_dict["$GLOBALS$"] += ['#include "pool.hpp"']

    def defines(self, var):
        self.code_gen_dict["$DEFINES$"] = []

        ifm_ch = self.get_nodeattr("Channels")
        self.code_gen_dict["$DEFINES$"] += ["#define Channels {}".format(ifm_ch)]

        pe = self.get_nodeattr("PE")
        self.code_gen_dict["$DEFINES$"] += ["#define PE {}".format(pe)]

        k = self.get_nodeattr("KernelSize")
        self.code_gen_dict["$DEFINES$"] += ["#define KernelSize {}".format(k)]

        odim = self.get_nodeattr("OutImgDim")
        self.code_gen_dict["$DEFINES$"] += ["#define OFMDim {}".format(odim)]

        numReps = self.get_nodeattr("BatchSize")
        self.code_gen_dict["$DEFINES$"] += ["#define numReps {}".format(numReps)]

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
            'npy2apintstream<%s, %s, %d, %s>("%s", in0,false);'
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
        idt = self.get_input_datatype()
        i_hls_dt = idt.get_hls_datatype_str()
        odt = self.get_output_datatype()
        o_hls_dt = odt.get_hls_datatype_str()
        size = self.get_nodeattr("Size")
        accum_bits = self.get_nodeattr("AccumBits")
        self.code_gen_dict["$DOCOMPUTE$"] = []

        fxn = self.get_nodeattr("Function")
        if fxn == "MaxPool":
            self.code_gen_dict["$DOCOMPUTE$"] += [
                "MaxPoolFunction<{},KernelSize> pool_fxn;".format(i_hls_dt)
            ]
        elif fxn == "QuantAvgPool":
            if idt.signed():
                act_hls_dt = "ap_int<{}>".format(accum_bits)
            else:
                act_hls_dt = "ap_uint<{}>".format(accum_bits)
            self.code_gen_dict["$DOCOMPUTE$"] += [
                "QuantAvgPoolFunction<{},{},{}> pool_fxn;".format(
                    act_hls_dt, o_hls_dt, size
                )
            ]
        else:
            raise Exception("Pool_Batch doesn't currently support " + fxn)

        self.code_gen_dict["$DOCOMPUTE$"] += [
            """Pool_batch<Channels, PE, KernelSize,Slice<{} >, Slice< {} > >
        (in0,out, pool_fxn, OFMDim*OFMDim*numReps);""".format(
                i_hls_dt, o_hls_dt
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
            'apintstream2npy<%s, %s, %d, %s>(out, %s, "%s",false);'
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
        packed_ibits = self.get_instream_width()
        packed_in_hls_type = "ap_uint<%d>" % packed_ibits

        packed_obits = self.get_outstream_width()
        packed_out_hls_type = "ap_uint<%d>" % packed_obits
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            "void %s(hls::stream<%s > &in0, hls::stream<%s > &out)"
            % (self.onnx_node.name, packed_in_hls_type, packed_out_hls_type)
        ]

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
        folded_ishape = self.get_folded_input_shape()
        exp_oshape = self.get_normal_output_shape()
        folded_oshape = self.get_folded_output_shape()

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
                context[node.output[0]].shape == folded_oshape
            ), "cppsim did not produce expected folded output shape"
            context[node.output[0]] = context[node.output[0]].reshape(*exp_oshape)
        elif mode == "rtlsim":
            sim = self.get_rtlsim()
            nbits = self.get_instream_width()
            rtlsim_inp = npy_to_rtlsim_input(
                "{}/input_0.npy".format(code_gen_dir), export_idt, nbits
            )
            super().reset_rtlsim(sim)
            super().toggle_clk(sim)
            rtlsim_output = self.rtlsim(sim, rtlsim_inp)
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
