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
import warnings
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.split import StreamingSplit
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy


class StreamingSplit_hls(StreamingSplit, HLSBackend):
    """Streaming split node with dynamically generated HLS.
    Only supports splitting along the last axis."""

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(StreamingSplit.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node

        if mode == "cppsim":
            HLSBackend.execute_node(self, context, graph)
            return
        elif mode == "rtlsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )
        inputs = {}
        for i, inp in enumerate(node.input):
            exp_ishape = tuple(self.get_normal_input_shape(i))
            folded_ishape = self.get_folded_input_shape(i)
            inp_val = context[inp]
            # Make sure the input has the right container datatype
            if inp_val.dtype is not np.float32:
                # Issue a warning to make the user aware of this type-cast
                warnings.warn(
                    f"{node.name}: Changing input container datatype from "
                    f"{inp_val.dtype} to {np.float32}"
                )
                # Convert the input to floating point representation as the
                # container datatype
                inp_val = inp_val.astype(np.float32)
            assert inp_val.shape == exp_ishape, "Input shape doesn't match expected shape."
            export_idt = self.get_input_datatype(i)

            if export_idt == DataType["BIPOLAR"]:
                # store bipolar activations as binary
                inp_val = (inp_val + 1) / 2
                export_idt = DataType["BINARY"]

            reshaped_input = inp_val.reshape(folded_ishape)
            reshaped_input = reshaped_input.copy()
            np.save(os.path.join(code_gen_dir, "input_%s.npy" % i), reshaped_input)
            nbits = self.get_instream_width(i)
            # if the stream is not exposed, it has 0 width and no npy file will be created
            if nbits == 0:
                continue
            rtlsim_inp = npy_to_rtlsim_input(
                "{}/input_{}.npy".format(code_gen_dir, i), export_idt, nbits
            )
            inputs["in%s" % i] = rtlsim_inp

        if mode == "rtlsim":
            outputs = {}
            for o, outp in enumerate(node.output):
                outputs["out_%d" % o] = []
            # assembled execution context
            io_dict = {"inputs": inputs, "outputs": outputs}

            sim = self.get_rtlsim()
            self.reset_rtlsim(sim)
            self.rtlsim_multi_io(sim, io_dict, sname="")
            self.close_rtlsim(sim)
            for o, outp in enumerate(node.output):
                rtlsim_output = io_dict["outputs"]["out_%s" % o]
                odt = self.get_output_datatype(o)
                target_bits = odt.bitwidth()
                packed_bits = self.get_outstream_width(o)
                out_npy_path = "{}/output_{}.npy".format(code_gen_dir, o)
                out_shape = self.get_folded_output_shape(o)
                rtlsim_output_to_npy(
                    rtlsim_output, out_npy_path, odt, out_shape, packed_bits, target_bits
                )
                # load and reshape output
                exp_oshape = tuple(self.get_normal_output_shape(o))
                output = np.load(out_npy_path)
                output = np.asarray([output], dtype=np.float32).reshape(*exp_oshape)
                context[outp] = output

                assert (
                    context[outp].shape == exp_oshape
                ), "Output shape doesn't match expected shape."

        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )

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

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "split.hpp"']

    def defines(self, var):
        self.code_gen_dict["$DEFINES$"] = ["#define NUM_OUTPUTS " + str(self.get_n_outputs())]

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
            "hls::stream<hls::vector<{}, {}>> out_[NUM_OUTPUTS];".format(
                self.get_output_datatype().get_hls_datatype_str(), simd
            )
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            "hls::stream<hls::vector<{}, {}>> debug_out[NUM_OUTPUTS];".format(
                self.get_output_datatype().get_hls_datatype_str(), simd
            )
        )

    def docompute(self):
        self.code_gen_dict["$DOCOMPUTE$"] = []
        n_outputs = self.get_n_outputs()
        output_folds = [str(self.get_folded_output_shape(i)[-2]) for i in range(n_outputs)]
        out_stream_folds = ", ".join(output_folds)
        comp_call = "StreamingSplit<{}>(in0, out_);".format(out_stream_folds)
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
                'vectorstream2npy<%s, %s, %d>(debug_out[%d], %s, "%s");'
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
        out_streams = "hls::stream<hls::vector<%s, %d>> (&out_)[NUM_OUTPUTS]" % (
            input_elem_hls_type,
            simd,
        )
        blackbox_hls = "void %s(%s, %s)" % (self.onnx_node.name, in_stream, out_streams)
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [blackbox_hls]

    def pragmas(self):
        pragmas = []
        pragmas.append("#pragma HLS INTERFACE axis port=in0")
        for i in range(self.get_n_outputs()):
            pragmas.append("#pragma HLS INTERFACE axis port=out_[%d]" % i)
        pragmas.append("#pragma HLS INTERFACE ap_ctrl_none port=return")
        pragmas.append("#pragma HLS aggregate variable=in0 compact=bit")
        for i in range(self.get_n_outputs()):
            pragmas.append("#pragma HLS aggregate variable=out_[%d] compact=bit" % i)
        self.code_gen_dict["$PRAGMAS$"] = pragmas

    def timeout_condition(self):
        condition = []
        for i in range(self.get_n_outputs()):
            condition.append("out_[{}].empty()".format(i))
        condition = " && ".join(condition)
        self.code_gen_dict["$TIMEOUT_CONDITION$"] = [condition]

    def timeout_read_stream(self):
        read_stream_command = """
for(int i = 0; i < NUM_OUTPUTS; i++){
    if(!out_[i].empty())
         debug_out[i] << out_[i].read();
}
"""
        self.code_gen_dict["$TIMEOUT_READ_STREAM$"] = [read_stream_command]
