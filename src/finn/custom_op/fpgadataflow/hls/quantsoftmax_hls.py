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

from finn.custom_op.fpgadataflow.quantsoftmax import QuantSoftmax
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend

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
            '#include "softmax.hpp"',
            '#include "utils.hpp"'
            ]

    def defines(self, var):
        simd = self.get_nodeattr("simd")
        ibits = self.get_input_datatype().bitwidth()
        channels = self.get_nodeattr("channels")
        self.code_gen_dict["$DEFINES$"] = [
           f"""
            constexpr unsigned  SIMD = {simd};
            constexpr unsigned  W = {channels};
            using  T = ap_uint<{ibits}>;
            using  F = float;
           """
        ]

    def docompute(self):
        self.code_gen_dict["$DOCOMPUTE$"] = [
            f'''
                static hls::stream<hls::vector<T,SIMD>>  src0;
                static hls::stream<hls::vector<T,SIMD>>  dst0;

                move(in0_{self.hls_sname()}, src0);
                smaxquant<W,SIMD,T,F>(src0, dst0);
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