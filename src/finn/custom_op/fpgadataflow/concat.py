# Copyright (c) 2021, Xilinx
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

from finn.core.datatype import DataType
from finn.custom_op.fpgadataflow.hlscustomop import HLSCustomOp
from finn.util.basic import roundup_to_integer_multiple
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy


class StreamingConcat(HLSCustomOp):
    """Streaming concatenation node with dynamically generated HLS.
    Only supports concatenating along the last axis."""

    def __init__(self, onnx_node):
        super().__init__(onnx_node)

    def get_nodeattr_types(self):
        my_attrs = {
            # number of elements from each stream to concat
            "ElemsPerStream": ("ints", True, []),
            # FINN DataTypes for inputs; output datatype inferred from input
            "inputDataType": ("s", True, ""),
            # number of input vectors for non-concat axes, examples:
            # [1] is a single vector (like a FC layer with batch=1)
            # [4] is four vectors (like a FC layer with batch=4)
            # [1, 4, 4] is four * four vectors (like a conv layer with batch=1)
            "numInputVectors": ("ints", False, [1]),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_n_inputs(self):
        return len(self.get_nodeattr("ElemsPerStream"))

    def get_total_elems(self):
        elems_per_stream = self.get_nodeattr("ElemsPerStream")
        return int(np.sum(elems_per_stream))

    def get_normal_input_shape(self, ind=0):
        elems_per_stream = self.get_nodeattr("ElemsPerStream")
        elems = elems_per_stream[ind]
        vecs = list(self.get_nodeattr("numInputVectors"))
        ishape = tuple(vecs + [elems])
        return ishape

    def get_folded_input_shape(self, ind=0):
        return self.get_normal_input_shape(ind)

    def get_normal_output_shape(self):
        total_elems = self.get_total_elems()
        vecs = list(self.get_nodeattr("numInputVectors"))
        return tuple(vecs + [total_elems])

    def get_folded_output_shape(self):
        return self.get_normal_output_shape()

    def make_shape_compatible_op(self, model):
        # check all input shapes
        for i, inp in enumerate(self.onnx_node.input):
            exp_ishape = self.get_normal_input_shape(i)
            ishape = tuple(model.get_tensor_shape(inp))
            assert ishape == exp_ishape, "Unexpected shape for " + inp
        oshape = self.get_normal_output_shape()
        return super().make_const_shape_op(oshape)

    def infer_node_datatype(self, model):
        # check all input datatypes
        for i, inp in enumerate(self.onnx_node.input):
            idt = model.get_tensor_datatype(inp)
            assert idt == self.get_input_datatype()
        odt = self.get_output_datatype()
        model.set_tensor_datatype(self.onnx_node.output[0], odt)

    def verify_node(self):
        pass

    def get_input_datatype(self, ind=0):
        # input dt identical for all inputs
        return DataType[self.get_nodeattr("inputDataType")]

    def get_output_datatype(self):
        return self.get_input_datatype()

    def get_instream_width(self, ind=0):
        elems_per_stream = self.get_nodeattr("ElemsPerStream")
        elems = elems_per_stream[ind]
        ibits = self.get_input_datatype().bitwidth()
        return elems * ibits

    def get_outstream_width(self):
        obits = self.get_output_datatype().bitwidth()
        total_elems = self.get_total_elems()
        out_width = total_elems * obits
        return out_width

    def get_number_output_values(self):
        return np.prod(self.get_folded_output_shape()[:-1])

    def get_exp_cycles(self):
        return np.prod(self.get_folded_output_shape()[:-1])

    def generate_params(self, model, path):
        elems_per_stream = self.get_nodeattr("ElemsPerStream")
        inp_streams = []
        commands = []
        idt = self.get_input_datatype()
        total_elems = self.get_total_elems()
        total_bw = idt.bitwidth() * total_elems
        for (i, elems) in enumerate(elems_per_stream):
            bw = idt.bitwidth() * elems
            inp_stream = "hls::stream<ap_uint<%d> > &in%d" % (bw, i)
            inp_streams.append(inp_stream)
            cmd = "in%d.read()" % i
            commands.append(cmd)
        out_stream = "hls::stream<ap_uint<%d> > &out" % (total_bw)
        inp_streams.append(out_stream)

        impl_hls_code = []
        impl_hls_code.append("void StreamingConcat(")
        impl_hls_code.append(",".join(inp_streams))
        impl_hls_code.append(", unsigned int numReps) {")
        impl_hls_code.append("for(unsigned int i = 0; i < numReps; i++) {")
        impl_hls_code.append("#pragma HLS PIPELINE II=1")
        impl_hls_code.append("ap_uint<%d> out_elem;" % total_bw)
        impl_hls_code.append("out_elem = (" + ",".join(commands) + ");")
        impl_hls_code.append("out.write(out_elem);")
        impl_hls_code.append("}")
        impl_hls_code.append("}")
        impl_hls_code = "\n".join(impl_hls_code)

        impl_filename = "{}/concat_impl.hpp".format(path)
        f_impl = open(impl_filename, "w")
        f_impl.write(impl_hls_code)
        f_impl.close()

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node
        n_inps = len(self.onnx_node.input)
        ishapes = [self.get_normal_input_shape(x) for x in range(n_inps)]
        folded_ishapes = [self.get_folded_input_shape(x) for x in range(n_inps)]
        exp_oshape = self.get_normal_output_shape()
        folded_oshape = self.get_folded_output_shape()
        export_idt = self.get_input_datatype()

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

        for i in range(n_inps):
            inp = context[node.input[i]]
            assert str(inp.dtype) == "float32", "Input datatype is not float32"
            assert inp.shape == ishapes[i], "Input shape mismatch for " + node.input[i]
            # reshape input into folded form
            inp = inp.reshape(folded_ishapes[i])
            # make copy before saving array
            reshaped_input = inp.copy()
            np.save(os.path.join(code_gen_dir, "input_%d.npy" % i), reshaped_input)

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
            io_dict = {"inputs": {}, "outputs": {"out": []}}
            for i in range(n_inps):
                nbits = self.get_instream_width(i)
                rtlsim_inp = npy_to_rtlsim_input(
                    "%s/input_%d.npy" % (code_gen_dir, i),
                    export_idt,
                    nbits,
                    reverse_inner=False,
                )
                io_dict["inputs"]["in%d" % i] = rtlsim_inp
            super().reset_rtlsim(sim)
            super().toggle_clk(sim)

            self.rtlsim_multi_io(sim, io_dict)
            rtlsim_output = io_dict["outputs"]["out"]
            odt = self.get_output_datatype()
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
                reverse_inner=False,
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

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "concat_impl.hpp"']

    def defines(self, var):
        num_reps = self.get_nodeattr("numInputVectors")
        num_reps = np.prod(num_reps)
        self.code_gen_dict["$DEFINES$"] = ["#define NumReps %d" % num_reps]

    def read_npy_data(self):
        n_inputs = self.get_n_inputs()
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        npy_type = "float"
        self.code_gen_dict["$READNPYDATA$"] = []
        idt = self.get_input_datatype()
        idt_bw = idt.bitwidth()
        elem_hls_type = idt.get_hls_datatype_str()
        elem_bits = idt_bw
        for i in range(n_inputs):
            packed_bits = self.get_instream_width(i)
            packed_hls_type = "ap_uint<%d>" % packed_bits
            npy_in = "%s/input_%d.npy" % (code_gen_dir, i)
            self.code_gen_dict["$READNPYDATA$"].append(
                'npy2apintstream<%s, %s, %d, %s>("%s", in%d);'
                % (packed_hls_type, elem_hls_type, elem_bits, npy_type, npy_in, i)
            )

    def strm_decl(self):
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        n_inputs = self.get_n_inputs()
        for i in range(n_inputs):
            packed_bits = self.get_instream_width(i)
            packed_hls_type = "ap_uint<%d>" % packed_bits
            stream_name = "in%d" % i
            self.code_gen_dict["$STREAMDECLARATIONS$"].append(
                'hls::stream<%s> %s ("%s");'
                % (packed_hls_type, stream_name, stream_name)
            )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> out ("out");'.format(self.get_outstream_width())
        )

    def docompute(self):
        self.code_gen_dict["$DOCOMPUTE$"] = []
        n_inputs = self.get_n_inputs()
        in_stream_names = ["in%d" % x for x in range(n_inputs)]
        in_stream_names = ",".join(in_stream_names)
        comp_call = "StreamingConcat(%s, out, NumReps);" % (in_stream_names)
        self.code_gen_dict["$DOCOMPUTE$"] = [comp_call]

    def dataoutstrm(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_output_datatype()
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
        n_inputs = self.get_n_inputs()
        in_streams = []
        for i in range(n_inputs):
            iwidth = self.get_instream_width(i)
            in_streams.append("hls::stream<ap_uint<%d>> &in%d" % (iwidth, i))
        in_streams = ",".join(in_streams)
        total_width = self.get_input_datatype().bitwidth() * self.get_total_elems()
        out_stream = "hls::stream<ap_uint<%d>> &out" % (total_width)
        blackbox_hls = "void %s(%s, %s)" % (self.onnx_node.name, in_streams, out_stream)
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [blackbox_hls]

    def pragmas(self):
        n_inputs = self.get_n_inputs()
        pragmas = []
        for i in range(n_inputs):
            pragmas.append("#pragma HLS INTERFACE axis port=in%d" % i)
        self.code_gen_dict["$PRAGMAS$"] = pragmas
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=out")
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE ap_ctrl_none port=return"
        )

    def get_instream_width_padded(self, ind=0):
        in_width = self.get_instream_width(ind)
        return roundup_to_integer_multiple(in_width, 8)

    def get_verilog_top_module_intf_names(self):
        intf_names = super().get_verilog_top_module_intf_names()
        n_inputs = self.get_n_inputs()
        intf_names["s_axis"] = []
        for i in range(n_inputs):
            intf_names["s_axis"].append(
                ("in%d_V_V" % i, self.get_instream_width_padded(i))
            )
        return intf_names
