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

from finn.custom_op.fpgadataflow import templates
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.split import StreamingSplit
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy


class StreamingSplit_hls(StreamingSplit, HLSBackend):
    """Streaming split node with dynamically generated HLS.
    Only supports splitting along the last axis."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(StreamingSplit.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node
        ishape = self.get_normal_input_shape()
        folded_ishape = self.get_folded_input_shape()
        n_outputs = self.get_n_outputs()
        exp_oshapes = [self.get_normal_output_shape(i) for i in range(len(node.output))]
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

        inp = context[node.input[0]]
        assert str(inp.dtype) == "float32", "Input datatype is not float32"
        assert inp.shape == ishape, "Input shape mismatch for " + node.input[0]
        # reshape input into folded form
        inp = inp.reshape(folded_ishape)
        # make copy before saving array
        reshaped_input = inp.copy()
        np.save(os.path.join(code_gen_dir, "input_0.npy"), reshaped_input)

        if mode == "cppsim":
            # execute the precompiled model
            super().exec_precompiled_singlenode_model()
            # load output npy file
            super().npy_to_dynamic_outputs(context, ["output_%d.npy" % i for i in range(n_outputs)])
            for i in range(n_outputs):
                assert (
                    context[node.output[i]].shape == exp_oshapes[i]
                ), "cppsim did not produce expected folded output shape: {}, expected: {}".format(
                    context[node.output[i]].shape, exp_oshapes[i]
                )
        elif mode == "rtlsim":
            sim = self.get_rtlsim()
            io_dict = {"inputs": {}, "outputs": {}}

            nbits = self.get_instream_width()
            rtlsim_inp = npy_to_rtlsim_input(
                "%s/input_0.npy" % code_gen_dir,
                export_idt,
                nbits,
                # reverse_inner=True,
            )
            io_dict["inputs"]["in0"] = rtlsim_inp
            super().reset_rtlsim(sim)
            super().toggle_clk(sim)

            for i in range(n_outputs):
                io_dict["outputs"]["out_arr_%d" % i] = []
            self.rtlsim_multi_io(sim, io_dict, sname="_")
            odt = self.get_output_datatype()
            target_bits = odt.bitwidth()
            packed_bits = self.get_outstream_width()
            for i in range(n_outputs):
                out_npy_path = "%s/output_%d.npy" % (code_gen_dir, i)
                out_shape = self.get_folded_output_shape(i)
                rtlsim_output_to_npy(
                    io_dict["outputs"]["out_arr_%d" % i],
                    out_npy_path,
                    odt,
                    out_shape,
                    packed_bits,
                    target_bits,
                    # reverse_inner=True,
                )
                # load and reshape output
                output = np.load(out_npy_path)
                output = np.asarray([output], dtype=np.float32).reshape(*exp_oshapes[i])
                context[node.output[i]] = output
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )

        for i in range(n_outputs):
            assert (
                context[node.output[i]].shape == exp_oshapes[i]
            ), "cppsim did not produce expected folded output shape. Got: {}, expected: {}".format(
                context[node.output[i]].shape, exp_oshapes[i]
            )

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
        self.code_gen_dict["$GLOBALS$"] = ['#include "split.hpp"']

    def defines(self, var):
        self.code_gen_dict["$DEFINES$"] = ["#define NUM_OUTPUTS " + str(self.get_n_outputs())]

    def read_npy_data(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        npy_type = "float"
        self.code_gen_dict["$READNPYDATA$"] = []
        simd = self.get_nodeattr("SIMD")
        input_elem_hls_type = self.get_input_datatype().get_hls_datatype_str()
        npy_in = "%s/input_0.npy" % code_gen_dir
        self.code_gen_dict["$READNPYDATA$"].append(
            'npy2vectorstream<%s, %s, %d>("%s", in0);'
            % (input_elem_hls_type, npy_type, simd, npy_in)
        )

    def strm_decl(self):
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        simd = self.get_nodeattr("SIMD")
        input_elem_hls_type = self.get_input_datatype().get_hls_datatype_str()
        stream_name = "in0"
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<hls::vector<%s, %d>> %s ("%s");'
            % (input_elem_hls_type, simd, stream_name, stream_name)
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            "hls::stream<hls::vector<{}, {}>> out_arr[NUM_OUTPUTS];".format(
                self.get_output_datatype().get_hls_datatype_str(), simd
            )
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            "hls::stream<hls::vector<{}, {}>> debug_out_arr[NUM_OUTPUTS];".format(
                self.get_output_datatype().get_hls_datatype_str(), simd
            )
        )

    def docompute(self):
        self.code_gen_dict["$DOCOMPUTE$"] = []
        n_outputs = self.get_n_outputs()
        output_folds = [str(self.get_folded_output_shape(i)[-2]) for i in range(n_outputs)]
        out_stream_folds = ", ".join(output_folds)
        comp_call = "StreamingSplit<{}>(in0, out_arr);".format(out_stream_folds)
        self.code_gen_dict["$DOCOMPUTE$"] = [comp_call]

    def dataoutstrm(self):
        npy_type = "float"
        simd = self.get_nodeattr("SIMD")
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        n_outputs = self.get_n_outputs()
        self.code_gen_dict["$DATAOUTSTREAM$"] = []
        for i in range(n_outputs):
            oshape = self.get_folded_output_shape(i)
            oshape_cpp_str = str(oshape).replace("(", "{").replace(")", "}")
            npy_out = "%s/output_%d.npy" % (code_gen_dir, i)
            self.code_gen_dict["$DATAOUTSTREAM$"].append(
                'vectorstream2npy<%s, %s, %d>(debug_out_arr[%d], %s, "%s");'
                % (
                    self.get_output_datatype(i).get_hls_datatype_str(),
                    npy_type,
                    simd,
                    i,
                    oshape_cpp_str,
                    npy_out,
                )
            )

    def blackboxfunction(self):
        input_elem_hls_type = self.get_input_datatype().get_hls_datatype_str()
        simd = self.get_nodeattr("SIMD")
        in_stream = "hls::stream<hls::vector<%s, %d>> &in0" % (input_elem_hls_type, simd)
        out_streams = "hls::stream<hls::vector<%s, %d>> (&out_arr)[NUM_OUTPUTS]" % (
            input_elem_hls_type,
            simd,
        )
        blackbox_hls = "void %s(%s, %s)" % (self.onnx_node.name, in_stream, out_streams)
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [blackbox_hls]

    def pragmas(self):
        pragmas = []
        pragmas.append("#pragma HLS INTERFACE axis port=in0")
        for i in range(self.get_n_outputs()):
            pragmas.append("#pragma HLS INTERFACE axis port=out_arr[%d]" % i)
        pragmas.append("#pragma HLS INTERFACE ap_ctrl_none port=return")
        pragmas.append("#pragma HLS aggregate variable=in0 compact=bit")
        for i in range(self.get_n_outputs()):
            pragmas.append("#pragma HLS aggregate variable=out_arr[%d] compact=bit" % i)
        self.code_gen_dict["$PRAGMAS$"] = pragmas

    def timeout_condition(self):
        condition = []
        for i in range(self.get_n_outputs()):
            condition.append("out_arr[{}].empty()".format(i))
        condition = " && ".join(condition)
        self.code_gen_dict["$TIMEOUT_CONDITION$"] = [condition]

    def timeout_read_stream(self):
        read_stream_command = """
for(int i = 0; i < NUM_OUTPUTS; i++){
    if(!out_arr[i].empty())
         debug_out_arr[i] << out_arr[i].read();
}
"""
        self.code_gen_dict["$TIMEOUT_READ_STREAM$"] = [read_stream_command]
