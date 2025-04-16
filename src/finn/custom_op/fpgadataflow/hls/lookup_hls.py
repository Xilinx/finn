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
from math import ceil, log2
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.lookup import Lookup
from finn.util.data_packing import numpy_to_hls_code, pack_innermost_dim_as_hex_string


class Lookup_hls(Lookup, HLSBackend):
    "Streaming elementwise HLS lookup, mapping indices to values."

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(Lookup.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def global_includes(self):
        mem_mode = self.get_nodeattr("mem_mode")
        global_incls = []
        global_incls.append('#include "lookup.hpp"')
        if mem_mode == "internal_embedded":
            global_incls.append('#include "embeddings.hpp"')
        self.code_gen_dict["$GLOBALS$"] = global_incls

    def defines(self, var):
        n_inputs = np.prod(self.get_folded_input_shape()[:-1])
        dtype = self.get_input_datatype()
        elem_hls_type = dtype.get_hls_datatype_str()
        emb_type = DataType[self.get_nodeattr("EmbeddingType")]
        emb_hls_type = emb_type.get_hls_datatype_str()
        emb_dim = self.get_nodeattr("EmbeddingDim")
        mem_mode = self.get_nodeattr("mem_mode")
        my_defines = []
        my_defines.append("#define NumInputs %d" % n_inputs)
        if mem_mode == "external":
            ext_mem_width = self.get_nodeattr("ext_mem_width")
            ext_mem_emb_size = self.get_folded_output_shape()[-2]
            ext_mem_emb_align = ceil(log2(ext_mem_emb_size))
            my_defines.append("#define MemBits %d" % ext_mem_width)
            my_defines.append("#define EmbeddingSize %d" % ext_mem_emb_size)
            my_defines.append("#define EmbeddingAlign %d" % ext_mem_emb_align)
            my_defines.append("#define T_SRC %s" % elem_hls_type)
            my_defines.append("#define T_DST ap_uint<MemBits>")
        elif mem_mode == "internal_embedded":
            my_defines.append("#define NumEmbeddings %d" % self.get_nodeattr("NumEmbeddings"))
            my_defines.append("#define EmbeddingDim %d" % emb_dim)
            my_defines.append("#define InputType %s" % elem_hls_type)
            my_defines.append("#define EmbeddingType %s" % emb_hls_type)
        self.code_gen_dict["$DEFINES$"] = my_defines

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
        npy_out = "%s/output_0.npy" % code_gen_dir
        oshape = self.get_folded_output_shape()
        oshape_cpp_str = str(oshape).replace("(", "{").replace(")", "}")

        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            'apintstream2npy<%s, %s, %d, %s>(out0_V, %s, "%s", %s);'
            % (
                packed_hls_type,
                elem_hls_type,
                elem_bits,
                npy_type,
                oshape_cpp_str,
                npy_out,
                "false",
            )
        ]

    def docompute(self):
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "internal_embedded":
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """StreamingLookup<NumEmbeddings,  EmbeddingDim, NumInputs,
                InputType, EmbeddingType >(in0_V, out0_V, embeddings);"""
            ]
        elif mem_mode == "external":
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """StreamingLookup_ext<EmbeddingSize>(in0_V, out0_V, mem, size, oob_count,
                oob_irq);"""
            ]

    def blackboxfunction(self):
        mem_mode = self.get_nodeattr("mem_mode")
        ibits = self.get_instream_width()
        packed_input_hls_type = "ap_uint<%d>" % ibits
        obits = self.get_outstream_width()
        packed_output_hls_type = "ap_uint<%d>" % obits
        if mem_mode == "internal_embedded":
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                "void %s(hls::stream<%s > &in0_V, hls::stream<%s > &out0_V)"
                % (
                    self.onnx_node.name,
                    packed_input_hls_type,
                    packed_output_hls_type,
                )
            ]
        elif mem_mode == "external":
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                "void "
                + self.onnx_node.name
                + "(hls::stream<T_SRC> &in0_V, hls::stream<T_DST> &out0_V, "
                + "T_DST const *const  mem, unsigned const size, "
                + "unsigned &oob_count, bool &oob_irq)"
            ]

    def pragmas(self):
        mem_mode = self.get_nodeattr("mem_mode")
        my_pragmas = ["#pragma HLS INTERFACE axis port=in0_V"]
        my_pragmas.append("#pragma HLS INTERFACE axis port=out0_V")
        my_pragmas.append("#pragma HLS INTERFACE ap_ctrl_none port=return")
        if mem_mode == "internal_embedded":
            my_pragmas.append("#pragma HLS BIND_STORAGE variable=embeddings type=ROM_2P impl=BRAM")
        elif mem_mode == "external":
            my_pragmas.append("#pragma HLS INTERFACE m_axi offset=slave port=mem")
            my_pragmas.append("#pragma HLS INTERFACE s_axilite port=mem bundle=control")
            my_pragmas.append("#pragma HLS INTERFACE s_axilite port=size bundle=control")
            my_pragmas.append("#pragma HLS INTERFACE s_axilite port=oob_count bundle=control")
            my_pragmas.append("#pragma HLS INTERFACE ap_none port=oob_irq")
        else:
            raise Exception("Unrecognized mem_mode: " + mem_mode)
        self.code_gen_dict["$PRAGMAS$"] = my_pragmas

    def generate_params(self, model, path):
        mem_mode = self.get_nodeattr("mem_mode")
        embeddings = model.get_initializer(self.onnx_node.input[1])
        if mem_mode == "internal_embedded":
            code_gen_dir = path
            weight_filename = "{}/embeddings.hpp".format(code_gen_dir)
            edt = DataType[self.get_nodeattr("EmbeddingType")]
            # obits = self.get_outstream_width()
            # packed_output_hls_type = "ap_uint<%d>" % obits
            assert np.vectorize(edt.allowed)(
                embeddings
            ).all(), "Embeddings can't be expressed with type %s" % str(edt)
            # reverse innertmost dim in embeddings to remain compatible with
            # how we normally encode the data in FINN
            embeddings_rev = np.flip(embeddings, -1)
            embeddings_hls_code = numpy_to_hls_code(embeddings_rev, edt, "embeddings", True, False)
            f_thresh = open(weight_filename, "w")
            f_thresh.write(embeddings_hls_code)
            f_thresh.close()
        elif mem_mode == "external":
            edt = DataType[self.get_nodeattr("EmbeddingType")]
            ext_mem_width = self.get_nodeattr("ext_mem_width")
            assert edt.bitwidth() == 8, (
                "Lookup with mem_mode=external "
                + "only works with 8-bit embeddings but found "
                + str(edt)
            )
            emb_dim = self.get_nodeattr("EmbeddingDim")
            # need to zero-pad embeddings in external mode for burst alignment
            # compute how much padding we need
            emb_elems_per_ext_mem_width = self.get_folded_output_shape()[-1]
            ext_mem_emb_size = self.get_folded_output_shape()[-2]
            ext_mem_emb_align = ceil(log2(ext_mem_emb_size))
            align_factor = int((ext_mem_width / 8) * 2**ext_mem_emb_align)
            pad_amount = align_factor - emb_dim
            embeddings_padded = np.pad(embeddings, [(0, 0), (0, pad_amount)])
            # reshape for packing the innermost dim
            embeddings_padded = embeddings_padded.reshape(-1, emb_elems_per_ext_mem_width)
            weight_filename = "%s/%s.dat" % (path, self.onnx_node.name)
            ret = pack_innermost_dim_as_hex_string(
                embeddings_padded, edt, ext_mem_width, True, prefix=""
            )
            with open(weight_filename, "w") as f:
                for current_line in ret:
                    f.write(current_line + "\n")
        else:
            raise Exception("Unrecognized mem_mode: " + mem_mode)

    def execute_node(self, context, graph):
        mem_mode = self.get_nodeattr("mem_mode")
        assert (
            mem_mode == "internal_embedded"
        ), "Only mem_mode=internal_embedded is supported for simulation of Lookup layer"
        HLSBackend.execute_node(self, context, graph)

    def get_ap_int_max_w(self):
        parent_max = super().get_ap_int_max_w()
        mem_mode = self.get_nodeattr("mem_mode")
        ext_mem_width = self.get_nodeattr("ext_mem_width")
        if mem_mode == "external":
            return max(ext_mem_width, parent_max)
        else:
            return parent_max
