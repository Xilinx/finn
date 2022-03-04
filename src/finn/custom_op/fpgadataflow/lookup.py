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
from math import ceil, log2

from finn.core.datatype import DataType
from finn.custom_op.fpgadataflow.hlscustomop import HLSCustomOp
from finn.util.data_packing import (
    npy_to_rtlsim_input,
    numpy_to_hls_code,
    rtlsim_output_to_npy,
)


class Lookup(HLSCustomOp):
    "Streaming elementwise HLS lookup, mapping indices to values."

    def __init__(self, onnx_node):
        super().__init__(onnx_node)

    def get_nodeattr_types(self):
        my_attrs = {
            # Number of embeddings ("memory depth")
            "NumEmbeddings": ("i", True, 0),
            # Dimensionality of each embedding (part of "memory width")
            "EmbeddingDim": ("i", True, 0),
            # Datatype for embeddings (part of "memory width")
            "EmbeddingType": ("s", True, ""),
            # Datatype for inputs
            "InputType": ("s", True, ""),
            # Input shape
            "InputShape": ("ints", False, [1]),
            # Memory mode
            # const : parameters baked into bitfile (BRAM)
            # external : lookup performed in external memory over AXI MM
            "mem_mode": ("s", False, "const", ["const", "external"]),
            # Width for AXI-MM interface
            # only relevant when mem_mode="external"
            "ext_mem_width": ("i", False, 32),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_exp_cycles(self):
        n_inputs = np.prod(self.get_nodeattr("InputShape"))
        exp_cycles = int(n_inputs)
        return exp_cycles

    def get_normal_input_shape(self):
        return self.get_nodeattr("InputShape")

    def get_normal_output_shape(self):
        ishape = self.get_normal_input_shape()
        emb_dim = self.get_nodeattr("EmbeddingDim")
        oshape = list(ishape) + [emb_dim]
        return tuple(oshape)

    def get_folded_input_shape(self):
        ishape = self.get_normal_input_shape()
        folded_ishape = list(ishape) + [1]
        return tuple(folded_ishape)

    def get_folded_output_shape(self):
        ishape = self.get_normal_input_shape()
        mem_mode = self.get_nodeattr("mem_mode")
        emb_dim = self.get_nodeattr("EmbeddingDim")
        if mem_mode == "const":
            oshape = list(ishape) + [emb_dim]
        elif mem_mode == "external":
            ext_mem_width = self.get_nodeattr("ext_mem_width")
            bits_per_emb_elem = self.get_output_datatype().bitwidth()
            assert ext_mem_width % bits_per_emb_elem == 0
            emb_elems_per_ext_mem_width = ext_mem_width // bits_per_emb_elem
            oshape = list(ishape) + [
                emb_dim // emb_elems_per_ext_mem_width,
                emb_elems_per_ext_mem_width,
            ]
        else:
            raise Exception("Unrecognized mem_mode:" + mem_mode)
        return tuple(oshape)

    def make_shape_compatible_op(self, model):
        exp_ishape = tuple(self.get_normal_input_shape())
        oshape = tuple(self.get_normal_output_shape())
        ishape = tuple(model.get_tensor_shape(self.onnx_node.input[0]))
        assert ishape == exp_ishape, "Unexpected input shape for Lookup: %s vs %s" % (
            str(exp_ishape),
            str(ishape),
        )
        return super().make_const_shape_op(oshape)

    def infer_node_datatype(self, model):
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            warn_str = "InputType changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype()),
                str(idt),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("InputType", idt.name)
        odt = DataType[self.get_nodeattr("EmbeddingType")]
        model.set_tensor_datatype(node.output[0], odt)

    def verify_node(self):
        pass

    def get_input_datatype(self):
        ret = DataType[self.get_nodeattr("InputType")]
        return ret

    def get_output_datatype(self):
        ret = DataType[self.get_nodeattr("EmbeddingType")]
        return ret

    def get_instream_width(self):
        ibits = self.get_input_datatype().bitwidth()
        return ibits

    def get_outstream_width(self):
        folded_oshape = self.get_folded_output_shape()
        obits = self.get_output_datatype().bitwidth()
        return obits * folded_oshape[-1]

    def get_number_output_values(self):
        folded_oshape = self.get_folded_output_shape()
        return np.prod(folded_oshape[:-1])

    def global_includes(self):
        mem_mode = self.get_nodeattr("mem_mode")
        global_incls = []
        if mem_mode == "const":
            global_incls.append('#include "lookup.hpp"')
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
        elif mem_mode == "const":
            my_defines.append(
                "#define NumEmbeddings %d" % self.get_nodeattr("NumEmbeddings")
            )
            my_defines.append("#define EmbeddingDim %d" % emb_dim)
            my_defines.append("#define InputType %s" % elem_hls_type)
            my_defines.append("#define EmbeddingType %s" % emb_hls_type)
        self.code_gen_dict["$DEFINES$"] = my_defines

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
        npy_type = "int64_t"
        npy_in = "%s/input_0.npy" % code_gen_dir
        self.code_gen_dict["$READNPYDATA$"] = []
        self.code_gen_dict["$READNPYDATA$"].append(
            'npy2apintstream<%s, %s, %d, %s>("%s", in0);'
            % (packed_hls_type, elem_hls_type, elem_bits, npy_type, npy_in)
        )

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
            'apintstream2npy<%s, %s, %d, %s>(out, %s, "%s", %s);'
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

    def save_as_npy(self):
        self.code_gen_dict["$SAVEASCNPY$"] = []

    def strm_decl(self):
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> in0 ("in0");'.format(self.get_instream_width())
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> out ("out");'.format(self.get_outstream_width())
        )

    def docompute(self):
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "const":
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """StreamingLookup<NumEmbeddings,  EmbeddingDim, NumInputs,
                InputType, EmbeddingType >(in0, out, embeddings);"""
            ]
        elif mem_mode == "external":
            hls_impl = """
    for(unsigned  i = 0; i < NumInputs; i++) {
        ap_uint<T_SRC::width+EmbeddingAlign> const  base =
            (in0.read(), ap_uint<EmbeddingAlign>(0));
        for(unsigned  j = 0; j < EmbeddingSize; j++) {
#pragma HLS PIPELINE II=1
            out.write(mem[base+j]);
        }
    }
            """
            self.code_gen_dict["$DOCOMPUTE$"] = [hls_impl]

    def blackboxfunction(self):
        mem_mode = self.get_nodeattr("mem_mode")
        ibits = self.get_instream_width()
        packed_input_hls_type = "ap_uint<%d>" % ibits
        obits = self.get_outstream_width()
        packed_output_hls_type = "ap_uint<%d>" % obits
        if mem_mode == "const":
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                "void %s(hls::stream<%s > &in0, hls::stream<%s > &out)"
                % (self.onnx_node.name, packed_input_hls_type, packed_output_hls_type)
            ]
        elif mem_mode == "external":
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                "void "
                + self.onnx_node.name
                + "(hls::stream<T_SRC> &in0, hls::stream<T_DST> &out, "
                + "T_DST const *const  mem)"
            ]

    def pragmas(self):
        mem_mode = self.get_nodeattr("mem_mode")
        my_pragmas = [
            "#pragma HLS INTERFACE axis port=in0 name=in0_" + self.hls_sname()
        ]
        my_pragmas.append(
            "#pragma HLS INTERFACE axis port=out name=out_" + self.hls_sname()
        )
        my_pragmas.append("#pragma HLS INTERFACE ap_ctrl_none port=return")
        if mem_mode == "const":
            my_pragmas.append(
                "#pragma HLS BIND_STORAGE variable=embeddings type=ROM_2P impl=BRAM"
            )
        elif mem_mode == "external":
            my_pragmas.append("#pragma HLS INTERFACE m_axi offset=slave port=mem")
            my_pragmas.append("#pragma HLS INTERFACE s_axilite port=mem bundle=control")
        else:
            raise Exception("Unrecognized mem_mode: " + mem_mode)
        self.code_gen_dict["$PRAGMAS$"] = my_pragmas

    def generate_params(self, model, path):
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "const":
            code_gen_dir = path
            embeddings = model.get_initializer(self.onnx_node.input[1])
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
            embeddings_hls_code = numpy_to_hls_code(
                embeddings_rev, edt, "embeddings", True, False
            )
            f_thresh = open(weight_filename, "w")
            f_thresh.write(embeddings_hls_code)
            f_thresh.close()

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node
        exp_ishape = tuple(self.get_normal_input_shape())
        exp_oshape = tuple(self.get_normal_output_shape())
        folded_ishape = tuple(self.get_folded_input_shape())
        folded_oshape = tuple(self.get_folded_output_shape())
        mem_mode = self.get_nodeattr("mem_mode")
        assert (
            mem_mode == "const"
        ), "Only mem_mode=const is supported for simulation of Lookup layer"

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
        assert inp.dtype == np.int64, "Inputs must be contained in int64 ndarray"
        assert inp.shape == exp_ishape, """Input shape doesn't match expected shape."""
        export_idt = self.get_input_datatype()
        odt = self.get_output_datatype()

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
            target_bits = odt.bitwidth()
            packed_bits = self.get_outstream_width()
            out_npy_path = "{}/output.npy".format(code_gen_dir)
            out_shape = self.get_folded_output_shape()
            rtlsim_output_to_npy(
                rtlsim_output,
                out_npy_path,
                odt,
                out_shape,
                packed_bits,
                target_bits,
                reverse_inner=True,
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
        ), """Output shape doesn't match expected shape."""

    def bram_estimation(self):
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "const":
            # current calculation assumes embeddings always stored in BRAM_18Ks
            # when mem_mode is const
            width_factor = ceil(self.get_outstream_width() / 16)
            depth_factor = ceil(self.get_nodeattr("NumEmbeddings") / 1024)
            return width_factor * depth_factor
        else:
            # TODO can we estimate BRAMs for the DMA engine?
            return 0

    def bram_efficiency_estimation(self):
        bram16_est = self.bram_estimation()
        if bram16_est == 0:
            return 1
        ebits = self.get_outstream_width() * self.get_nodeattr("NumEmbeddings")
        bram16_est_capacity = bram16_est * 18 * 1024
        return ebits / bram16_est_capacity

    def get_ap_int_max_w(self):
        parent_max = super().get_ap_int_max_w()
        mem_mode = self.get_nodeattr("mem_mode")
        ext_mem_width = self.get_nodeattr("ext_mem_width")
        if mem_mode == "external":
            return max(ext_mem_width, parent_max)
        else:
            return parent_max
