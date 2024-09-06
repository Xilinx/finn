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
from finn.custom_op.fpgadataflow import templates
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

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node
        n_inps = len(self.onnx_node.input)
        ishapes = [self.get_normal_input_shape(x) for x in range(n_inps)]
        folded_ishapes = [self.get_folded_input_shape(x) for x in range(n_inps)]
        exp_oshape = self.get_normal_output_shape()
        export_idts = [self.get_input_datatype(i) for i in range(n_inps)]

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
                context[node.output[0]].shape == exp_oshape
            ), "cppsim did not produce expected folded output shape. Got: {}, expected: {}".format(context[node.output[0]].shape, exp_oshape)
            context[node.output[0]] = context[node.output[0]].reshape(*exp_oshape)
        elif mode == "rtlsim":
            sim = self.get_rtlsim()
            io_dict = {"inputs": {}, "outputs": {"out": []}}
            for i in range(n_inps):
                nbits = self.get_instream_width(i)
                rtlsim_inp = npy_to_rtlsim_input(
                    "%s/input_%d.npy" % (code_gen_dir, i),
                    export_idts[i],
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

    def code_generation_cppsim(self, model):
        """Generates c++ code for simulation (cppsim)."""
        node = self.onnx_node
        path = self.get_nodeattr("code_gen_dir_cppsim")
        self.code_gen_dict["$AP_INT_MAX_W$"] = [str(self.get_ap_int_max_w())]
        self.generate_params(model, path)
        self.global_includes()
        self.defines("cppsim")
        self.read_npy_data()
        self.strm_decl()
        self.pragmas()
        self.docompute()
        self.dataoutstrm()
        self.save_as_npy()
        self.timeout_value()
        self.timeout_condition()
        self.timeout_read_stream()

        template = templates.docompute_template_timeout

        for key in self.code_gen_dict:
            # transform list into long string separated by '\n'
            code_gen_line = "\n".join(self.code_gen_dict[key])
            template = template.replace(key, code_gen_line)
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        f = open(os.path.join(code_gen_dir, "execute_{}.cpp".format(node.op_type)), "w")
        f.write(template)
        f.close()
        self.code_gen_dict.clear()

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "concat.hpp"']

    def defines(self, var):
        self.code_gen_dict["$DEFINES$"] = ["#define SIMD {}".format(self.get_nodeattr("SIMD"))]

    def read_npy_data(self):
        n_inputs = self.get_n_inputs()
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        npy_type = "float"
        self.code_gen_dict["$READNPYDATA$"] = []
        simd = self.get_nodeattr("SIMD")
        for i in range(n_inputs):
            input_elem_hls_type = self.get_input_datatype(i).get_hls_datatype_str()
            npy_in = "%s/input_%d.npy" % (code_gen_dir, i)
            self.code_gen_dict["$READNPYDATA$"].append(
                'npy2vectorstream<%s, %s, SIMD>("%s", in%d_%s);'
                % (
                    input_elem_hls_type,
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
            input_elem_hls_type = self.get_input_datatype(i).get_hls_datatype_str()
            stream_name = "in%d_%s" % (i, self.hls_sname())
            self.code_gen_dict["$STREAMDECLARATIONS$"].append(
                'hls::stream<hls::vector<%s, SIMD>> %s ("%s");' %
                (input_elem_hls_type, stream_name, stream_name)
            )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<hls::vector<{}, SIMD>> out_{} ("out_{}");'.format(
                self.get_output_datatype().get_hls_datatype_str(),
                self.hls_sname(),
                self.hls_sname()
            )
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<hls::vector<{}, SIMD>> debug_out_{} ("debug_out_{}");'.format(
                self.get_output_datatype().get_hls_datatype_str(),
                self.hls_sname(),
                self.hls_sname()
            )
        )

    def docompute(self):
        self.code_gen_dict["$DOCOMPUTE$"] = []
        n_inputs = self.get_n_inputs()
        input_folds = [str(self.get_folded_input_shape(i)[-2]) for i in range(n_inputs)]
        in_streams = []
        for i in range(n_inputs):
            in_streams.append("in%d_%s" % (i, self.hls_sname()))
        in_stream_names = ", ".join(in_streams)
        in_stream_folds = ", ".join(input_folds)
        comp_call = "StreamingConcat<{}>(out_{}, {});".format(in_stream_folds, self.hls_sname(), in_stream_names)
        self.code_gen_dict["$DOCOMPUTE$"] = [comp_call]

    def dataoutstrm(self):
        npy_type = "float"
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        oshape = self.get_folded_output_shape()
        oshape_cpp_str = str(oshape).replace("(", "{").replace(")", "}")
        npy_out = "%s/output.npy" % code_gen_dir
        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            'vectorstream2npy<%s, %s, SIMD>(debug_out_%s, %s, "%s");'
            % (
                self.get_output_datatype().get_hls_datatype_str(),
                npy_type,
                self.hls_sname(),
                oshape_cpp_str,
                npy_out,
            )
        ]

    def blackboxfunction(self):
        n_inputs = self.get_n_inputs()
        in_streams = []
        for i in range(n_inputs):
            input_elem_hls_type = self.get_input_datatype(i).get_hls_datatype_str()
            in_streams.append("hls::stream<hls::vector<%s, SIMD>> &in%d_%s" % (input_elem_hls_type, i, self.hls_sname()))
        in_streams = ", ".join(in_streams)
        out_stream = "hls::stream<hls::vector<%s, SIMD>> &out_%s" % (
            self.get_output_datatype().get_hls_datatype_str(),
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
        for i in range(n_inputs):
            pragmas.append("#pragma HLS aggregate variable=in%d_%s compact=bit" % (i, self.hls_sname()))
        pragmas.append("#pragma HLS aggregate variable=out_%s compact=bit" % self.hls_sname())
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE ap_ctrl_none port=return")
