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

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.pool import Pool


class Pool_hls(Pool, HLSBackend):
    """Class that corresponds to finn-hlslib Pool_batch function.
    Requires ConvolutionInputGenerator(depthwise == 1) to format its input

    Input shape (BatchSize,OutImgDim,OutImgDim,TotalKernelSize*Channels)
    Output shape (BatchSize,OutImgDim,OutImgDim,Channels)

    Notes:

    * The input shape was chosen to be compatible with im2col (only true when there
      is not folding).
    * The actual data layout produced by the hlslib kernels is different
      for depthwise ops.

        * depthwise SWG: (1, OFMDim, OFMDim, IFMChannels/PE, K, K, PE)

    Channels can be folded using PE (SIMD from the input perspective)
    """

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(Pool.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "pool.hpp"']

    def defines(self, var):
        k = int(np.prod(self.get_nodeattr("KernelSize")))
        self.code_gen_dict["$DEFINES$"] = [
            "static constexpr unsigned K = {};".format(k),
        ]

    def docompute(self):
        pe = self.get_nodeattr("PE")
        fxn = self.get_nodeattr("Function")
        idt = self.get_input_datatype()
        odt = self.get_output_datatype()
        o_tensor_dt = "tensor<%s, %d>" % (odt.get_hls_datatype_str(), pe)

        self.code_gen_dict["$DOCOMPUTE$"] = []
        if fxn == "MaxPool":
            self.code_gen_dict["$DOCOMPUTE$"] += ["MaxPoolFunction<{}> pool_fxn;".format(o_tensor_dt)]
        elif fxn == "QuantAvgPool":
            shift = self.get_nodeattr("Size")
            accum_bits = self.get_nodeattr("AccumBits")
            act_tensor_dt = "tensor<ap_%sint<%d>, %d>" % (
                "" if idt.signed() else "u",
                accum_bits,
                pe,
            )
            self.code_gen_dict["$DOCOMPUTE$"] += [
                "QuantAvgPoolFunction<{},{},{}> pool_fxn;".format(o_tensor_dt, act_tensor_dt, shift)
            ]
        else:
            raise Exception("Pool_Batch doesn't currently support " + fxn)

        self.code_gen_dict["$DOCOMPUTE$"] += ["Pool_batch<K>(in0_V, out0_V, pool_fxn);"]

    def pragmas(self):
        super().pragmas()
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS dataflow disable_start_propagation")
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS aggregate variable=in0_V compact=bit")
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS aggregate variable=out0_V compact=bit")

    def ipgen_extra_directives(self):
        "Return a list of extra tcl directives for HLS synthesis."
        return ["config_compile -pipeline_style flp"]

    def blackboxfunction(self):
        pe = self.get_nodeattr("PE")
        idt = self.get_input_datatype()
        odt = self.get_output_datatype()
        i_tensor_dt = "tensor<%s, %d>" % (idt.get_hls_datatype_str(), pe)
        o_tensor_dt = "tensor<%s, %d>" % (odt.get_hls_datatype_str(), pe)

        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            "void %s(hls::stream<%s> &in0_V, hls::stream<%s> &out0_V)"
            % (self.onnx_node.name, i_tensor_dt, o_tensor_dt)
        ]

    def execute_node(self, context, graph):
        HLSBackend.execute_node(self, context, graph)
