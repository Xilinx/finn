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

import os
import numpy as np
from finn.custom_op.fpgadataflow.quantsoftmax import QuantSoftmax
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow import templates
from finn.util.basic import CppBuilder
class QuantSoftmax_hls(QuantSoftmax, HLSBackend):
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(QuantSoftmax.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = [
            '#include <hls_vector.h>',
            '#include "softmax.hpp"',
            '#include "utils.hpp"',
            ]

    def defines(self, var):
        simd = self.get_nodeattr("simd")
        dtype = self.get_input_datatype()
        channels = self.get_nodeattr("channels")
        self.code_gen_dict["$DEFINES$"] = [
           f"""
            constexpr unsigned  SIMD = {simd};
            constexpr unsigned  W = {channels};
            using  T = {dtype.get_hls_datatype_str()};
            using  F = float;
           """
        ]

    def docompute(self):
        self.code_gen_dict["$DOCOMPUTE$"] = [
            f'''
                static hls::stream<hls::vector<T,SIMD>>  src0;
                static hls::stream<hls::vector<T,SIMD>>  dst0;

                move(in0_{self.hls_sname()}, src0);
                smaxquant<W,SIMD,T>(src0, dst0);
                move(dst0, out_{self.hls_sname()});
        '''
        ]

    def blackboxfunction(self):
        self.code_gen_dict["$BLACKBOXFUNCTION$"]  = [
            f'''
            void {self.onnx_node.name}(
                hls::stream<hls::vector<T,SIMD>> &in0_{self.hls_sname()},
                hls::stream<hls::vector<T,SIMD>> &out_{self.hls_sname()}
                )
            '''
        ]

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"]  = [
            f'''
            #pragma HLS interface AXIS port=in0_{self.hls_sname()}
            #pragma HLS interface AXIS port=out_{self.hls_sname()}
            #pragma HLS aggregate  variable=in0_{self.hls_sname()} compact=bit
            #pragma HLS aggregate  variable=out_{self.hls_sname()} compact=bit

            #pragma HLS interface ap_ctrl_none port=return
            #pragma HLS dataflow disable_start_propagation
            '''
        ]

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node
        folded_ishape = self.get_folded_input_shape()

        if mode == "cppsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
            inp = context[node.input[0]]
            export_idt = self.get_input_datatype()
            inp = inp.reshape(folded_ishape)
            np.save(os.path.join(code_gen_dir, "input_0.npy"), inp)
            # # execute the precompiled model
            super().exec_precompiled_singlenode_model()
            # # load output npy file
            super().npy_to_dynamic_output(context)
        else:
            raise Exception(f"Unsupported execution mode: {mode}")

    def compile_singlenode_code(self):
        """Builds the bash script for compilation using the CppBuilder from
        finn.util.basic and executes the script to produce the executable."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        builder = CppBuilder()
        # to enable additional debug features please uncommand the next line
        # builder.append_includes("-DDEBUG")
        builder.append_includes("-I$FINN_ROOT/src/finn/qnn-data/cpp")
        builder.append_includes("-I$FINN_ROOT/deps/cnpy/")
        builder.append_includes("-I$FINN_ROOT/deps/finn-hlslib")
        builder.append_includes("-I$FINN_ROOT/custom_hls")
        builder.append_includes("-I{}/include".format(os.environ["HLS_PATH"]))
        builder.append_includes("--std=c++14")
        builder.append_includes("-O3")
        builder.append_sources(code_gen_dir + "/*.cpp")
        builder.append_sources("$FINN_ROOT/deps/cnpy/cnpy.cpp")
        builder.append_includes("-lz")
        builder.append_includes("-fno-builtin -fno-inline -Wl,-rpath,\"$HLS_PATH/lnx64/lib/csim\" -L$HLS_PATH/lnx64/lib/csim -lhlsmc++-GCC46")
        builder.append_includes("-L$HLS_PATH/lnx64/tools/fpo_v7_1 -lgmp -lmpfr -lIp_floating_point_v7_1_bitacc_cmodel")
        builder.set_executable_path(code_gen_dir + "/node_model")
        builder.build(code_gen_dir)
        self.set_nodeattr("executable_path", builder.executable_path)

    def code_generation_cppsim(self, model):
        """Generates c++ code for simulation (cppsim)."""
        self.code_gen_dict["$READNPYDATA$"] = [""]
        self.code_gen_dict["$DATAOUTSTREAM$"] = [""]
        self.code_gen_dict["$STREAMDECLARATIONS$"] = [""]
        node = self.onnx_node
        path = self.get_nodeattr("code_gen_dir_cppsim")
        self.code_gen_dict["$AP_INT_MAX_W$"] = [str(self.get_ap_int_max_w())]
        self.generate_params(model, path)
        self.global_includes()
        self.defines("cppsim")
        self.pragmas()
        oshape = self.get_folded_output_shape()
        oshape_str = str(oshape).replace("(", "{").replace(")", "}")
        self.code_gen_dict["$DOCOMPUTE$"] = [
            f'''
            static hls::stream<hls::vector<T,SIMD>>  in0_V;
            static hls::stream<hls::vector<T,SIMD>>  out_V;

            npy2vectorstream<T, float, SIMD>("{path}/input_0.npy", in0_V);
            int stream_size = in0_V.size();

            while(out_V.size() != stream_size){{
                smaxquant<W, SIMD, T>(in0_V, out_V);
            }}

            vectorstream2npy<T, float, SIMD>(out_V,{oshape_str}, "{path}/output.npy");
            '''
        ]
        self.save_as_npy()

        template = templates.docompute_template

        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim") + f"/execute_{node.op_type}.cpp"
        with open(code_gen_dir, "w") as f:
            for key in self.code_gen_dict:
                # transform list into long string separated by '\n'
                code_gen_line = "\n".join(self.code_gen_dict[key])
                template = template.replace(key, code_gen_line)
            f.write(template)

    def prepare_rtlsim(self):
        # this node currently does not support rtlsim
        raise NotImplementedError("QuantSoftmax_hls does not support rtlsim")