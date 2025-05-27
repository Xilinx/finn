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
        self.code_gen_dict["$GLOBALS$"] = ['#include "activations.hpp"']
        self.code_gen_dict["$GLOBALS$"] += ['#include "maxpool.h"']
        self.code_gen_dict["$GLOBALS$"] += ['#include "pool.hpp"']

    def defines(self, var):
        self.code_gen_dict["$DEFINES$"] = []

        ifm_ch = self.get_nodeattr("Channels")
        self.code_gen_dict["$DEFINES$"] += ["#define Channels {}".format(ifm_ch)]

        pe = self.get_nodeattr("PE")
        self.code_gen_dict["$DEFINES$"] += ["#define PE {}".format(pe)]

        k = self.get_nodeattr("KernelSize")
        k_prod = int(np.prod(k))
        self.code_gen_dict["$DEFINES$"] += ["#define KernelSize {}".format(k_prod)]

        odims = self.get_nodeattr("OutImgDims")
        total_odim = np.prod(odims)
        self.code_gen_dict["$DEFINES$"] += ["#define OFMDimTotal {}".format(total_odim)]

        numReps = self.get_nodeattr("BatchSize")
        self.code_gen_dict["$DEFINES$"] += ["#define numReps {}".format(numReps)]

    def docompute(self):
        idt = self.get_input_datatype()
        i_hls_dt = idt.get_hls_datatype_str()
        odt = self.get_output_datatype()
        o_hls_dt = odt.get_hls_datatype_str()
        size = self.get_nodeattr("Size")
        accum_bits = self.get_nodeattr("AccumBits")
        self.code_gen_dict["$DOCOMPUTE$"] = []

        fxn = self.get_nodeattr("Function")
        if fxn == "MaxPool":
            self.code_gen_dict["$DOCOMPUTE$"] += [
                "MaxPoolFunction<{},KernelSize> pool_fxn;".format(i_hls_dt)
            ]
        elif fxn == "QuantAvgPool":
            if idt.signed():
                act_hls_dt = "ap_int<{}>".format(accum_bits)
            else:
                act_hls_dt = "ap_uint<{}>".format(accum_bits)
            self.code_gen_dict["$DOCOMPUTE$"] += [
                "QuantAvgPoolFunction<{},{},{}> pool_fxn;".format(act_hls_dt, o_hls_dt, size)
            ]
        else:
            raise Exception("Pool_Batch doesn't currently support " + fxn)

        self.code_gen_dict["$DOCOMPUTE$"] += [
            """Pool_batch<Channels, PE, KernelSize,Slice<{} >, Slice< {} > >
        (in0_V, out0_V, pool_fxn, OFMDimTotal*numReps);""".format(
                i_hls_dt, o_hls_dt
            )
        ]

    def blackboxfunction(self):
        packed_ibits = self.get_instream_width()
        packed_in_hls_type = "ap_uint<%d>" % packed_ibits

        packed_obits = self.get_outstream_width()
        packed_out_hls_type = "ap_uint<%d>" % packed_obits
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            "void %s(hls::stream<%s > &in0_V, hls::stream<%s > &out0_V)"
            % (
                self.onnx_node.name,
                packed_in_hls_type,
                packed_out_hls_type,
            )
        ]

    def execute_node(self, context, graph):
        HLSBackend.execute_node(self, context, graph)
