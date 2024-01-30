# Copyright (c) 2021, Xilinx
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

from finn.custom_op.fpgadataflow.concat import StreamingConcat
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy


class StreamingConcat_hls(StreamingConcat, HLSBackend):
    """Streaming concatenation node with dynamically generated HLS.
    Only supports concatenating along the last axis."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(StreamingConcat.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def generate_params(self, model, path):
        elems_per_stream = self.get_nodeattr("ElemsPerStream")
        inp_streams = []
        commands = []
        idt = self.get_input_datatype()
        total_elems = self.get_total_elems()
        total_bw = idt.bitwidth() * total_elems
        for i, elems in enumerate(elems_per_stream):
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
        # FIXME: the order of streams for concatenation works out differently
        # for cppsim vs rtlsim, addressed via reversing the order of commands
        # for now
        impl_hls_code.append("#ifdef __SYNTHESIS__")
        impl_hls_code.append("out_elem = (" + ",".join(commands[::-1]) + ");")
        impl_hls_code.append("#else")
        impl_hls_code.append("out_elem = (" + ",".join(commands) + ");")
        impl_hls_code.append("#endif")
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
                    reverse_inner=True,
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
                'npy2apintstream<%s, %s, %d, %s>("%s", in%d_%s);'
                % (
                    packed_hls_type,
                    elem_hls_type,
                    elem_bits,
                    npy_type,
                    npy_in,
                    i,
                    self.hls_sname(),
                )
            )

    def strm_decl(self):
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        n_inputs = self.get_n_inputs()
        for i in range(n_inputs):
            packed_bits = self.get_instream_width(i)
            packed_hls_type = "ap_uint<%d>" % packed_bits
            stream_name = "in%d_%s" % (i, self.hls_sname())
            self.code_gen_dict["$STREAMDECLARATIONS$"].append(
                'hls::stream<%s> %s ("%s");' % (packed_hls_type, stream_name, stream_name)
            )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> out_{} ("out_{}");'.format(
                self.get_outstream_width(), self.hls_sname(), self.hls_sname()
            )
        )

    def docompute(self):
        self.code_gen_dict["$DOCOMPUTE$"] = []
        n_inputs = self.get_n_inputs()
        in_streams = []
        for i in range(n_inputs):
            in_streams.append("in%d_%s" % (i, self.hls_sname()))
        in_stream_names = ",".join(in_streams)
        comp_call = "StreamingConcat(%s, out_%s, NumReps);" % (
            in_stream_names,
            self.hls_sname(),
        )
        self.code_gen_dict["$DOCOMPUTE$"] = [comp_call]

    def blackboxfunction(self):
        n_inputs = self.get_n_inputs()
        in_streams = []
        for i in range(n_inputs):
            iwidth = self.get_instream_width(i)
            in_streams.append("hls::stream<ap_uint<%d>> &in%d_%s" % (iwidth, i, self.hls_sname()))
        in_streams = ",".join(in_streams)
        total_width = self.get_input_datatype().bitwidth() * self.get_total_elems()
        out_stream = "hls::stream<ap_uint<%d>> &out_%s" % (
            total_width,
            self.hls_sname(),
        )
        blackbox_hls = "void %s(%s, %s)" % (self.onnx_node.name, in_streams, out_stream)
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [blackbox_hls]

    def pragmas(self):
        n_inputs = self.get_n_inputs()
        pragmas = []
        for i in range(n_inputs):
            pragmas.append("#pragma HLS INTERFACE axis port=in%d_%s" % (i, self.hls_sname()))
        self.code_gen_dict["$PRAGMAS$"] = pragmas
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE axis port=out_" + self.hls_sname()
        )
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE ap_ctrl_none port=return")
