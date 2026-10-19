# Copyright (c) 2022, Xilinx, Inc.
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
import warnings
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class CheckSum_hls(HWCustomOp, HLSBackend):
    """Class that corresponds to custom_hls checksum function."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            # number of data words in a frame
            "words_per_frame": ("i", True, 0),
            # subword count per data word
            "items_per_word": ("i", True, 0),
            # FINN DataTypes for input
            "inputDataType": ("s", True, ""),
            # folded shape of input/output
            "folded_shape": ("ints", True, []),
        }
        my_attrs.update(HWCustomOp.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def infer_node_datatype(self, model):
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            warn_str = "inputDataType changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype().name),
                str(idt.name),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("inputDataType", idt.name)
        # set output datatype from property
        odt = self.get_output_datatype()
        model.set_tensor_datatype(node.output[0], odt)

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        return DataType[self.get_nodeattr("inputDataType")]

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output."""
        # here same as input data type
        return DataType[self.get_nodeattr("inputDataType")]

    def get_instream_width(self, ind=0):
        dtype = DataType[self.get_nodeattr("inputDataType")]
        folded_shape = self.get_nodeattr("folded_shape")
        in_width = folded_shape[-1] * dtype.bitwidth()
        return in_width

    def get_outstream_width(self, ind=0):
        return self.get_instream_width()

    def get_folded_input_shape(self, ind=0):
        return self.get_nodeattr("folded_shape")

    def get_folded_output_shape(self, ind=0):
        return self.get_nodeattr("folded_shape")

    def get_normal_input_shape(self, ind=0):
        # derive normal shape from folded shape
        # checksum nodes are inserted in between fpgadataflow nodes
        # the folded shape could be for example (1, nf, pe)
        # with nf (neuron folding): mh // pe
        # the normal input shape is in this case (1, mh)
        # so to achieve this the two inner dimensions are multiplied
        # and together with all previous dimensions
        # this gives the normal input shape

        folded_shape = self.get_nodeattr("folded_shape")
        # extract inner dimension
        inner_dim = folded_shape[-1]
        # multiply with the next inner dimension
        folding_factor = folded_shape[-2] * inner_dim
        normal_ishape = []
        # create the normal_ishape
        for i in range(len(folded_shape) - 2):
            normal_ishape.append(folded_shape[i])
        normal_ishape.append(folding_factor)

        return normal_ishape

    def get_ap_int_max_w(self):
        return max(super().get_ap_int_max_w(), 32)

    def get_normal_output_shape(self, ind=0):
        if ind == 0:
            # same shape as input
            return self.get_normal_input_shape()
        # second output is scalar checksum output
        elif ind == 1:
            return tuple([1])
        else:
            raise Exception("Undefined input ind for this layer type")

    def npy_to_dynamic_output(self, context):
        super().npy_to_dynamic_output(context)
        node = self.onnx_node
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        output_checksum = np.load("{}/output_1.npy".format(code_gen_dir))
        context[node.output[1]] = output_checksum

    def execute_node(self, context, graph):
        HLSBackend.execute_node(self, context, graph)

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "checksum.hpp"', '#include "tensor.hpp"']

    def defines(self, var):
        self.code_gen_dict["$DEFINES$"] = [
            """static constexpr unsigned WORDS_PER_FRAME = {};
            static constexpr unsigned ITEMS_PER_WORD = {};
            static constexpr unsigned WORD_SIZE = {};""".format(
                self.get_nodeattr("words_per_frame"),
                self.get_nodeattr("items_per_word"),
                self.get_instream_width(),
            )
        ]

    def read_npy_data(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_input_datatype()
        elem_hls_type = dtype.get_hls_datatype_str()
        n_elems = self.get_folded_input_shape()[-1]
        npy_type = "float"
        npy_in = "%s/input_0.npy" % code_gen_dir
        self.code_gen_dict["$READNPYDATA$"] = []
        self.code_gen_dict["$READNPYDATA$"].append(
            'npy2tensorstream<%s>("%s", in0_V);'
            % (npy_type, npy_in)
        )

    def strm_decl(self):
        dtype = self.get_input_datatype()
        elem_hls_type = dtype.get_hls_datatype_str()
        i_n = self.get_folded_input_shape()[-1]
        odtype = self.get_output_datatype()
        o_elem = odtype.get_hls_datatype_str()
        o_n = self.get_folded_output_shape()[-1]
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<tensor<{},{}>> in0_V ("in0_V");'.format(elem_hls_type, i_n)
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<tensor<{},{}>> out0_V ("out0_V");'.format(o_elem, o_n)
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<tensor<{},{}>> strm0 ("strm0");'.format(o_elem, o_n)
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append("ap_uint<32> chk;")
        # set drain = false for cppsim
        self.code_gen_dict["$STREAMDECLARATIONS$"].append("ap_uint<1> drain = false;")

    def docompute(self):
        self.code_gen_dict["$DOCOMPUTE$"] = [
            """checksum<WORDS_PER_FRAME, ITEMS_PER_WORD>(in0_V, out0_V, chk, drain,
            [](auto const &x, unsigned j) -> ap_uint<WORD_SIZE/ITEMS_PER_WORD> {
                return  x[j];
            });"""
        ]

    def dataoutstrm(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_output_datatype()
        elem_hls_type = dtype.get_hls_datatype_str()
        n_elems = self.get_folded_output_shape()[-1]
        npy_type = "float"
        npy_out = "%s/output_0.npy" % code_gen_dir
        shape = tuple(self.get_folded_output_shape())
        shape_cpp_str = str(shape).replace("(", "{").replace(")", "}")

        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            'tensorstream2npy<%s>(strm0, %s, "%s");'
            % (npy_type, shape_cpp_str, npy_out),
            "std::vector<unsigned int> checksum(1);",
            "checksum[0] = chk;",
            'cnpy::npy_save("%s/output_1.npy",&checksum[0],{1});' % code_gen_dir,
        ]

    def blackboxfunction(self):
        dtype = self.get_input_datatype()
        elem_hls_type = dtype.get_hls_datatype_str()
        n_elems = self.get_folded_input_shape()[-1]
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            "void {}(hls::stream<tensor<{},{}>> &in0_V, "
            "hls::stream<tensor<{},{}>> &out0_V, "
            "ap_uint<32> &chk, ap_uint<1> &drain)".format(
                self.onnx_node.name,
                elem_hls_type, n_elems,
                elem_hls_type, n_elems,
            )
        ]

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"] = ["#pragma HLS interface axis port=in0_V"]
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS interface axis port=out0_V")
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS interface s_axilite port=chk bundle=checksum"
        )
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS interface s_axilite port=drain bundle=checksum"
        )
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS interface ap_ctrl_none port=return")

    def get_verilog_top_module_intf_names(self):
        intf_names = super().get_verilog_top_module_intf_names()
        # expose axilite interface
        intf_names["axilite"] = ["s_axi_checksum"]
        return intf_names
