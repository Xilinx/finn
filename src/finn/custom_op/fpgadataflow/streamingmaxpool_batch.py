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

import numpy as np
import os
import warnings

from finn.core.datatype import DataType
from finn.custom_op.fpgadataflow.hlscustomop import HLSCustomOp
from finn.custom_op.general.maxpoolnhwc import compute_pool_output_dim
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy


class StreamingMaxPool_Batch(HLSCustomOp):
    """Class that corresponds to finn-hlslib StreamingMaxPool_batch function."""

    def get_nodeattr_types(self):
        my_attrs = {
            "ImgDim": ("ints", True, []),  # [H, W] = [Y, X]
            "PoolDim": ("ints", True, []),  # [H, W] = [Y, X]
            "NumChannels": ("i", True, 0),
            "PE": ("i", True, 0),
            "CeilMode": ("i", False, 0),
            # FINN DataTypes for inputs/outputs
            "dataType": ("s", True, ""),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_input_datatype(self):
        """Returns FINN DataType of input."""
        return DataType[self.get_nodeattr("dataType")]

    def get_output_datatype(self):
        """Returns FINN DataType of output."""
        return DataType[self.get_nodeattr("dataType")]

    def get_1d_attrs_normalized(self):
        # support both (1, D) and (D, 1) cases transparently:
        # assume the dummy ('1') dimension is the Y-dimension, i.e.
        # images and kernels (and their attributes) of dimension
        # [H, W] = [Y, X] = [D, 1] or [1, D] are always mapped to [1, D]
        ifm_dim = self.get_nodeattr("ImgDim")
        k = self.get_nodeattr("PoolDim")
        ifm_ch = self.get_nodeattr("NumChannels")
        if ifm_dim[1] == 1:
            ifm_dim = ifm_dim[::-1]
            k = k[::-1]
        return (ifm_dim, k, ifm_ch)

    def is_1d(self):
        ifm_dim, k, ifm_ch = self.get_1d_attrs_normalized()
        return (ifm_dim[0] == 1) and (k[0] == 1)

    def get_normal_input_shape(self):
        ifm_dim_h, ifm_dim_w = self.get_nodeattr("ImgDim")
        ifm_ch = self.get_nodeattr("NumChannels")
        ishape = (1, ifm_dim_h, ifm_dim_w, ifm_ch)
        return ishape

    def get_folded_input_shape(self):
        ifm_dim_h, ifm_dim_w = self.get_nodeattr("ImgDim")
        ifm_ch = self.get_nodeattr("NumChannels")
        pe = self.get_nodeattr("PE")
        nf = int(ifm_ch / pe)
        if self.is_1d():
            folded_ishape = (1, ifm_dim_h, ifm_dim_w, nf, pe)
        else:
            folded_ishape = (1, ifm_dim_h, ifm_dim_w, 1, ifm_ch)
        return folded_ishape

    def get_normal_output_shape(self):
        ifm_dim_h, ifm_dim_w = self.get_nodeattr("ImgDim")
        k_h, k_w = tuple(self.get_nodeattr("PoolDim"))
        ifm_ch = self.get_nodeattr("NumChannels")
        ceil_mode = self.get_nodeattr("CeilMode")
        if not self.is_1d():
            assert (
                ifm_dim_h % k_h == 0
            ), "StreamingMaxPool needs ImgDim_h % PoolDim_h == 0"
            assert (
                ifm_dim_w % k_w == 0
            ), "StreamingMaxPool needs ImgDim_w % PoolDim_w == 0"
        ofm_dim_h = compute_pool_output_dim(ifm_dim_h, k_h, k_h, 0, ceil_mode)
        ofm_dim_w = compute_pool_output_dim(ifm_dim_w, k_w, k_w, 0, ceil_mode)
        oshape = (1, ofm_dim_h, ofm_dim_w, ifm_ch)
        return oshape

    def get_folded_output_shape(self):
        # even though there is no folding in the current hlslib op,
        # insert a time multiplexing axis to remain compatible with the
        # shapes produced by the rest of the dataflow pipeline
        ifm_ch = self.get_nodeattr("NumChannels")
        pe = self.get_nodeattr("PE")
        nf = int(ifm_ch / pe)
        ret = list(self.get_normal_output_shape())
        if self.is_1d():
            ret[-1] = nf
            ret.append(pe)
        else:
            ret.insert(-1, 1)
        return tuple(ret)

    def get_number_output_values(self):
        folded_oshape = self.get_folded_output_shape()
        return np.prod(folded_oshape[:-1])

    def get_exp_cycles(self):
        # derived from StreamingMaxPool_Batch loop nest
        ifm_dim, k, ifm_ch = self.get_1d_attrs_normalized()
        _, _, ofm_dim_w, nf, _ = self.get_folded_output_shape()

        if self.is_1d():
            exp_cycles = ofm_dim_w * nf * (k[1] + 1)
            return int(exp_cycles)
        else:
            # TODO: adjust inaccurate formula
            return int(ifm_dim[1] * (ifm_dim[1] + (ifm_dim[1] / k[1])))

    def get_instream_width(self):
        dt_bits = self.get_input_datatype().bitwidth()
        pe = self.get_nodeattr("PE")
        ifm_ch = self.get_nodeattr("NumChannels")
        if self.is_1d():
            in_width = int(dt_bits * pe)
        else:
            in_width = int(dt_bits * ifm_ch)
        return in_width

    def get_outstream_width(self):
        """For streaming maxpool out stream width is the same as in stream width"""
        return self.get_instream_width()

    def make_shape_compatible_op(self, model):
        exp_ishape = self.get_normal_input_shape()
        oshape = self.get_normal_output_shape()
        ishape = tuple(model.get_tensor_shape(self.onnx_node.input[0]))
        assert ishape == exp_ishape, "Unexpect input shape for StreamingMaxPool."
        return super().make_const_shape_op(oshape)

    def infer_node_datatype(self, model):
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            warn_str = "inputDataType changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype()),
                str(idt),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("dataType", idt.name)
        # data type stays the same
        model.set_tensor_datatype(node.output[0], idt)

    def verify_node(self):
        info_messages = []
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
            info_messages.append("""StreamingMaxPool_Batch needs 1 data input""")

        return info_messages

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "maxpool.h"']

    def defines(self, var):
        numReps = 1
        ifm_dim, k, ifm_ch = self.get_1d_attrs_normalized()
        ceil_mode = self.get_nodeattr("CeilMode")
        output_size = compute_pool_output_dim(ifm_dim[1], k[1], k[1], 0, ceil_mode)
        remainder_size = ifm_dim[1] - k[1] * output_size
        if remainder_size < 0:
            remainder_size = 0

        if self.is_1d():
            self.code_gen_dict["$DEFINES$"] = [
                """#define ImgDim {}\n #define PoolDim {}\n
                #define NumChannels {}\n #define PE {}\n #define OutputSize {}
                \n #define RemainderSize {}\n #define numReps {}""".format(
                    ifm_dim[1],
                    k[1],
                    self.get_nodeattr("NumChannels"),
                    self.get_nodeattr("PE"),
                    output_size,
                    remainder_size,
                    numReps,
                )
            ]
        else:
            self.code_gen_dict["$DEFINES$"] = [
                """#define ImgDim {}\n #define PoolDim {}\n
                #define NumChannels {}\n #define numReps {}""".format(
                    ifm_dim[1],
                    k[1],
                    self.get_nodeattr("NumChannels"),
                    numReps,
                )
            ]

    def read_npy_data(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_input_datatype()
        if dtype == DataType["BIPOLAR"]:
            # use binary for bipolar storage
            dtype = DataType["BINARY"]
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
        dtype = self.get_input_datatype()
        if dtype.bitwidth() == 1:
            if self.is_1d():
                raise Exception("Binary 1d MaxPool not implemented on HLS backend")
            else:
                op = "StreamingMaxPool"
            self.code_gen_dict["$DOCOMPUTE$"] = [
                "%s<ImgDim, PoolDim, NumChannels>(in0, out);" % (op)
            ]
        else:
            dtype = self.get_input_datatype()
            dtype_hls = dtype.get_hls_datatype_str()
            minval_str = str(int(dtype.min()))
            if self.is_1d():
                op = "StreamingMaxPool_Precision_1d"
                self.code_gen_dict["$DOCOMPUTE$"] = [
                    """%s<ImgDim, PoolDim, NumChannels, PE,
                     OutputSize, RemainderSize, %s, %s>(in0, out);"""
                    % (op, dtype_hls, minval_str)
                ]
            else:
                op = "StreamingMaxPool_Precision"
                self.code_gen_dict["$DOCOMPUTE$"] = [
                    "%s<ImgDim, PoolDim, NumChannels, %s, %s>(in0, out);"
                    % (op, dtype_hls, minval_str)
                ]

    def dataoutstrm(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_output_datatype()
        if dtype == DataType["BIPOLAR"]:
            # use binary for bipolar storage
            dtype = DataType["BINARY"]
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
        packed_bits = self.get_instream_width()
        packed_hls_type = "ap_uint<%d>" % packed_bits
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            "void %s(hls::stream<%s > &in0, hls::stream<%s > &out)"
            % (self.onnx_node.name, packed_hls_type, packed_hls_type)
        ]

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"] = [
            "#pragma HLS INTERFACE axis port=in0 name=in0_" + self.hls_sname()
        ]
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE axis port=out name=out_" + self.hls_sname()
        )
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE ap_ctrl_none port=return"
        )

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node
        exp_ishape = self.get_normal_input_shape()
        exp_oshape = self.get_normal_output_shape()
        folded_ishape = self.get_folded_input_shape()
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
        match expected shape (1, ifm_dim, ifm_dim, ifm_ch)."""
        if self.get_input_datatype() == DataType["BIPOLAR"]:
            # store bipolar activations as binary
            inp = (inp + 1) / 2
            export_idt = DataType["BINARY"]
        else:
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
            ), "cppsim \
            did not produce expected folded output shape"
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
            odt = export_idt
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
        # binary -> bipolar if needed
        if self.get_output_datatype() == DataType["BIPOLAR"]:
            out = context[node.output[0]]
            out = 2 * out - 1
            context[node.output[0]] = out
        assert (
            context[node.output[0]].shape == exp_oshape
        ), """Output
        shape doesn't match expected shape (1, ofm_dim, ofm_dim, ifm_ch)."""
