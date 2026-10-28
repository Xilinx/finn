# Copyright (C) 2023-2024, Advanced Micro Devices, Inc.
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
from finn.util.data_packing import hls_datatype_str


class Pool_hls(Pool, HLSBackend):
    """Consolidated HLS pooling: fuses pad<> + input_gen<> + Pool_batch<K>()
    into a single HLS dataflow kernel.

    Input shape (BatchSize, IFMDim_H, IFMDim_W, Channels)  — raw feature map
    Output shape (BatchSize, OutImgDim_H, OutImgDim_W, Channels)

    Channels can be folded using PE.
    """

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(Pool.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def global_includes(self):
        includes = ['#include "pool.hpp"', '#include "input_gen.hpp"']
        pad = self.get_nodeattr("Padding")
        if any(p != 0 for p in pad):
            includes.append('#include "pad.hpp"')
        self.code_gen_dict["$GLOBALS$"] = includes

    def defines(self, var):
        pe = self.get_nodeattr("PE")
        ch = self.get_nodeattr("Channels")
        cf = ch // pe
        kh, kw = self.get_nodeattr("KernelSize")
        k = kh * kw
        sh, sw = self.get_nodeattr("Stride")
        ifm_h, ifm_w = self.get_nodeattr("IFMDim")
        ofm_h, ofm_w = self.get_nodeattr("OutImgDims")
        pad = self.get_nodeattr("Padding")
        has_padding = any(p != 0 for p in pad)

        # Padded dimensions (what input_gen sees)
        ph = ifm_h + pad[0] + pad[2]
        pw = ifm_w + pad[1] + pad[3]

        defines = [
            "static constexpr unsigned K = {};".format(k),
            "static constexpr unsigned PE = {};".format(pe),
            "static constexpr unsigned CF = {};".format(cf),
            "static constexpr unsigned H = {};".format(ph),
            "static constexpr unsigned W = {};".format(pw),
            "static constexpr unsigned KH = {};".format(kh),
            "static constexpr unsigned KW = {};".format(kw),
            "static constexpr unsigned SH = {};".format(sh),
            "static constexpr unsigned SW = {};".format(sw),
            "static constexpr unsigned OH = {};".format(ofm_h),
            "static constexpr unsigned OW = {};".format(ofm_w),
        ]
        if has_padding:
            defines += [
                "static constexpr unsigned IFM_H = {};".format(ifm_h),
                "static constexpr unsigned IFM_W = {};".format(ifm_w),
                "static constexpr unsigned PAD_N = {};".format(pad[0]),
                "static constexpr unsigned PAD_W = {};".format(pad[1]),
                "static constexpr unsigned PAD_S = {};".format(pad[2]),
                "static constexpr unsigned PAD_E = {};".format(pad[3]),
            ]
        self.code_gen_dict["$DEFINES$"] = defines

    def docompute(self):
        pe = self.get_nodeattr("PE")
        fxn = self.get_nodeattr("Function")
        idt = self.get_input_datatype()
        odt = self.get_output_datatype()
        i_tensor_dt = "tensor<%s, %d>" % (hls_datatype_str(idt), pe)
        o_tensor_dt = "tensor<%s, %d>" % (hls_datatype_str(odt), pe)

        pad = self.get_nodeattr("Padding")
        has_padding = any(p != 0 for p in pad)

        lines = []

        # Pool functor instantiation
        if fxn == "MaxPool":
            lines.append("MaxPoolFunction<%s> pool_fxn;" % o_tensor_dt)
        elif fxn == "QuantAvgPool":
            shift = self.get_nodeattr("Size")
            accum_bits = self.get_nodeattr("AccumBits")
            act_tensor_dt = "tensor<ap_%sint<%d>, %d>" % (
                "" if idt.signed() else "u",
                accum_bits,
                pe,
            )
            lines.append(
                "QuantAvgPoolFunction<%s,%s,%d> pool_fxn;" % (o_tensor_dt, act_tensor_dt, shift)
            )
        else:
            raise Exception("Pool doesn't currently support " + fxn)

        # input_gen loop nest for depthwise 2D pooling with stride:
        #   for oh in OH: for ow in OW: for kh in KH: for kw in KW: for cf in CF:
        #     addr = (oh*SH+kh)*W*CF + (ow*SW+kw)*CF + cf
        ig_call = (
            "input_gen<-1, H*W*CF, OH, SH*W*CF, OW, SW*CF, CF, 1, KH, W*CF, KW, CF>"
        )

        if has_padding:
            # Pad value expression
            pad_val = self.get_nodeattr("PadValue")
            lines.append("static hls::stream<%s>  padded(\"padded\");" % i_tensor_dt)
            lines.append("#pragma HLS stream depth=2 variable=padded")
            lines.append("static hls::stream<%s>  feed(\"feed\");" % i_tensor_dt)
            lines.append("#pragma HLS stream depth=2 variable=feed")
            # Build pad value initializer
            lines.append("{")
            lines.append("  %s pad_val;" % i_tensor_dt)
            lines.append(
                "  for(unsigned i = 0; i < PE; i++) pad_val[i] = %s(%d);"
                % (hls_datatype_str(idt), pad_val)
            )
            lines.append(
                "  pad<IFM_H, IFM_W, CF, PAD_N, PAD_E, PAD_S, PAD_W>(in0_V, padded, pad_val);"
            )
            lines.append("}")
            lines.append("%s(padded, feed);" % ig_call)
            lines.append("Pool_batch<K>(feed, out0_V, pool_fxn);")
        else:
            lines.append("static hls::stream<%s>  feed(\"feed\");" % i_tensor_dt)
            lines.append("#pragma HLS stream depth=2 variable=feed")
            lines.append("%s(in0_V, feed);" % ig_call)
            lines.append("Pool_batch<K>(feed, out0_V, pool_fxn);")

        self.code_gen_dict["$DOCOMPUTE$"] = lines

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
        i_tensor_dt = "tensor<%s, %d>" % (hls_datatype_str(idt), pe)
        o_tensor_dt = "tensor<%s, %d>" % (hls_datatype_str(odt), pe)

        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            "void %s(hls::stream<%s> &in0_V, hls::stream<%s> &out0_V)"
            % (self.onnx_node.name, i_tensor_dt, o_tensor_dt)
        ]

    def execute_node(self, context, graph):
        HLSBackend.execute_node(self, context, graph)
