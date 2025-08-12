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
from qonnx.util.basic import interleave_matrix_outer_dim_from_partitions

from finn.custom_op.fpgadataflow.deconvolution import Deconvolution
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.util.data_packing import numpy_to_hls_code1


class Deconvolution_hls(Deconvolution, HLSBackend):
    """Corresponds to finn-hlslib deconv function."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(Deconvolution.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def calc_wmem(self):
        """Calculates and returns WMEM."""
        ifm_ch = self.get_nodeattr("IFMChannels")
        ofm_ch = self.get_nodeattr("OFMChannels")
        kernel_2 = np.prod(self.get_nodeattr("KernelDim"))
        pe = self.get_nodeattr("PE")
        simd = self.get_nodeattr("SIMD")
        assert ofm_ch % pe == 0, "Requirement output channels divisable by PE is violated."
        assert ifm_ch % simd == 0, "Requirement input channels divisable by SIMD is violated."
        wmem = (ofm_ch / pe) * kernel_2 * (ifm_ch / simd)
        return int(wmem)

    def generate_params(self, model, path):
        code_gen_dir = path
        # weights, if not external
        weights = model.get_initializer(self.onnx_node.input[1])
        # save hlslib-compatible weights in params.h
        weight_filename = "{}/params.h".format(code_gen_dir)
        self.make_weight_file(weights, "hls_header", weight_filename)

    def make_weight_file(self, weights, weight_file_mode, weight_file_name):
        """Produce a file containing given weights in appropriate format for this
        layer. This file can be used for either synthesis or run-time reconfig
        of weights.

        Arguments:

        * weights : numpy array with weights to be put into the file
        * weight_file_mode : one of {hls_header, decoupled_verilog_dat,
          decoupled_runtime}
        * weight_file_name : filename for the weight file to be generated

        """
        # convert weights into hlslib-compatible format
        weight_tensor = self.get_hw_compatible_weight_tensor(weights)
        export_wdt = self.get_weight_datatype()
        if weight_file_mode == "hls_header":
            weight_hls_code = numpy_to_hls_code1(weight_tensor, export_wdt, "weights", False, True)
            # write weights into C++ header file as dictated by finn-hlslib
            f_weights = open(weight_file_name, "w")
            f_weights.write(
                "static {} const weights[{}][{}][{}] = ".format(
                    export_wdt.get_hls_datatype_str(),
                    self.calc_wmem(),
                    self.get_nodeattr("PE"),
                    self.get_nodeattr("SIMD"),
                )
            )
            f_weights.write(weight_hls_code)
            f_weights.close()

    def get_hw_compatible_weight_tensor(self, orig_weight_matrix):
        """Convert the original numpy weight matrix orig_weight_matrix into
        a form suitable for passing to the hlslib call:
        * ensure OCH % PE == 0 and ICH % SIMD == 0
        * interleave rows between PEs
        * reshape into (1, PE, WMEM, SIMD) and return
        """
        k_h, k_w = self.get_nodeattr("KernelDim")
        ifm_ch = self.get_nodeattr("IFMChannels")
        ofm_ch = self.get_nodeattr("OFMChannels")
        pe = self.get_nodeattr("PE")
        simd = self.get_nodeattr("SIMD")
        wmem = self.calc_wmem()
        assert orig_weight_matrix.shape == (
            ofm_ch,
            k_h,
            k_w,
            ifm_ch,
        ), """Weights matrix doesn't
        have expected shape (ofm_ch, k_h, k_w, ifm_ch)"""
        assert ofm_ch % pe == 0, "Requirement output channels divisable by PE is violated."
        assert ifm_ch % simd == 0, "Requirement input channels divisable by SIMD is violated."
        # interleave rows between PEs and reshape
        # distribute rows between PEs
        ret = orig_weight_matrix
        ret = ret.reshape(ofm_ch, k_h * k_w * ifm_ch)
        ret = interleave_matrix_outer_dim_from_partitions(ret, pe)
        # create SIMD as innermost dimension
        ret = ret.reshape(1, pe, wmem, simd)
        ret = ret.transpose(0, 2, 1, 3)
        return ret

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "deconv.hpp"']

    def defines(self, var):
        ifm_dim = self.get_nodeattr("IFMDim")
        self.code_gen_dict["$DEFINES$"] = [
            """constexpr unsigned Kernel = {};\n constexpr unsigned Stride = {};\n
            constexpr unsigned Padding = {};\n constexpr unsigned IFMH = {};\n
            constexpr unsigned IFMW = {};\n constexpr unsigned ICH = {};\n
            constexpr unsigned OCH = {};\n constexpr unsigned SIMD1 = {};\n
            constexpr unsigned PE1 = {};""".format(
                self.get_nodeattr("KernelDim")[0],
                self.get_nodeattr("Stride")[0],
                self.get_nodeattr("Padding")[0],
                ifm_dim[0],
                ifm_dim[1],
                self.get_nodeattr("IFMChannels"),
                self.get_nodeattr("OFMChannels"),
                self.get_nodeattr("SIMD"),
                self.get_nodeattr("PE"),
            )
        ]

    def docompute(self):
        self.code_gen_dict["$DOCOMPUTE$"] = []
        self.code_gen_dict["$DOCOMPUTE$"].append(
            """deconv<Kernel, Stride, Padding, IFMH, IFMW, OCH, ICH, PE1, SIMD1>
            (weights, in0_V, out0_V);"""
        )

    def blackboxfunction(self):
        input_elem_hls_type = self.get_input_datatype().get_hls_datatype_str()
        output_elem_hls_type = self.get_output_datatype().get_hls_datatype_str()
        simd = self.get_nodeattr("SIMD")
        pe = self.get_nodeattr("PE")
        in_stream = "hls::stream<hls::vector<%s, %d>> &in0_V" % (
            input_elem_hls_type,
            simd,
        )
        out_stream = "hls::stream<hls::vector<%s, %d>> &out0_V" % (
            output_elem_hls_type,
            pe,
        )
        blackbox_hls = "void %s(%s, %s)" % (self.onnx_node.name, in_stream, out_stream)
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [blackbox_hls]

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"] = ["#pragma HLS INTERFACE axis port=in0_V"]
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=out0_V")
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE ap_ctrl_none port=return")

        self.code_gen_dict["$PRAGMAS$"].append('#include "params.h"')
        # the weight tensor is ap_uint<simd*prec> [PE][WMEM]
        # partition for parallel access along the PE dimension (dim 1)
        # self.code_gen_dict["$PRAGMAS$"].append(
        #    ("#pragma HLS ARRAY_PARTITION variable=weights.m_weights " "complete dim=1")
        # )

    def execute_node(self, context, graph):
        HLSBackend.execute_node(self, context, graph)

    def timeout_value(self):
        """Set timeout value for HLS functions defined for one clock cycle"""
        simd = self.get_nodeattr("SIMD")
        i_ch = self.get_nodeattr("IFMChannels")
        k_h, k_w = self.get_nodeattr("KernelDim")
        s_h, s_w = self.get_nodeattr("Stride")
        i_h, i_w = self.get_nodeattr("IFMDim")
        p_h, p_w = self.get_nodeattr("Padding")
        if p_w >= k_w - s_w:
            padup = 0
        else:
            padup = (k_w - p_w - 1) / s_w
        crop = s_w * padup - ((k_w - s_w) - p_w)
        sf = i_ch / simd
        w_eff = padup + i_w + padup
        wo_eff = (w_eff - 1) * s_w + k_w
        self.code_gen_dict["$TIMEOUT_VALUE$"] = [
            "%s" % (wo_eff * (crop + 1) * ((k_w / s_w) ** 2) * 4 * sf + 50)
        ]
