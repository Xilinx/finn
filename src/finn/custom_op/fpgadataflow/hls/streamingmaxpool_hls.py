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
from qonnx.core.datatype import DataType
from qonnx.custom_op.general.maxpoolnhwc import compute_pool_output_dim

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.streamingmaxpool import StreamingMaxPool
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy


class StreamingMaxPool_hls(StreamingMaxPool, HLSBackend):
    """Class that corresponds to finn-hlslib StreamingMaxPool_batch function."""

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(StreamingMaxPool.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def verify_node(self):
        info_messages = []
        # verify that "backend" is set to "fpgadataflow"
        backend_value = self.get_nodeattr("backend")
        if backend_value == "fpgadataflow":
            info_messages.append("Attribute backend is set correctly")
        else:
            info_messages.append('Attribute backend should be set to "fpgadataflow"')

        # verify the number of inputs
        if len(self.onnx_node.input) == 1:
            info_messages.append("The number of inputs is correct")
        else:
            info_messages.append("""StreamingMaxPool_Batch needs 1 data input""")

        return info_messages

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "maxpool.h"']

    def defines(self, var):
        numReps = 1
        ifm_dim, k, ifm_ch = self.get_1d_attrs_normalized()
        ceil_mode = self.get_nodeattr("CeilMode")
        output_size = compute_pool_output_dim(ifm_dim[1], k[1], k[1], 0, ceil_mode)

        if self.is_1d():
            self.code_gen_dict["$DEFINES$"] = [
                """#define ImgDim {}\n #define PoolDim {}\n
                #define NumChannels {}\n #define PE {}\n #define OutputSize {}
                \n #define numReps {}""".format(
                    ifm_dim[1],
                    k[1],
                    self.get_nodeattr("NumChannels"),
                    self.get_nodeattr("PE"),
                    output_size,
                    numReps,
                )
            ]
        else:
            self.code_gen_dict["$DEFINES$"] = [
                """#define ImgDim {}\n #define PoolDim {}\n
                #define NumChannels {}\n #define numReps {}""".format(
                    ifm_dim[1],
                    k[1],
                    self.get_nodeattr("NumChannels"),
                    numReps,
                )
            ]

    def docompute(self):
        dtype = self.get_input_datatype()
        if dtype.bitwidth() == 1:
            if self.is_1d():
                raise Exception("Binary 1d MaxPool not implemented on HLS backend")
            else:
                op = "StreamingMaxPool"
            self.code_gen_dict["$DOCOMPUTE$"] = [
                "%s<ImgDim, PoolDim, NumChannels>(in0_%s, out_%s);"
                % (op, self.hls_sname(), self.hls_sname())
            ]
        else:
            dtype = self.get_input_datatype()
            dtype_hls = dtype.get_hls_datatype_str()
            minval_str = str(int(dtype.min()))
            if self.is_1d():
                op = "StreamingMaxPool_Precision_1d"
                self.code_gen_dict["$DOCOMPUTE$"] = [
                    """%s<ImgDim, PoolDim, NumChannels, PE,
                     OutputSize, %s, %s>(in0_%s, out_%s);"""
                    % (op, dtype_hls, minval_str, self.hls_sname(), self.hls_sname())
                ]
            else:
                op = "StreamingMaxPool_Precision"
                self.code_gen_dict["$DOCOMPUTE$"] = [
                    "%s<ImgDim, PoolDim, NumChannels, %s, %s>(in0_%s, out_%s);"
                    % (op, dtype_hls, minval_str, self.hls_sname(), self.hls_sname())
                ]

    def blackboxfunction(self):
        packed_bits = self.get_instream_width()
        packed_hls_type = "ap_uint<%d>" % packed_bits
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            "void %s(hls::stream<%s > &in0_%s, hls::stream<%s > &out_%s)"
            % (
                self.onnx_node.name,
                packed_hls_type,
                self.hls_sname(),
                packed_hls_type,
                self.hls_sname(),
            )
        ]

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node
        exp_ishape = self.get_normal_input_shape()
        exp_oshape = self.get_normal_output_shape()
        folded_ishape = self.get_folded_input_shape()

        # TODO ensure codegen dir exists
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
        assert (
            inp.shape == exp_ishape
        ), """Input shape doesn't
        match expected shape (1, ifm_dim, ifm_dim, ifm_ch)."""
        if self.get_input_datatype() == DataType["BIPOLAR"]:
            # store bipolar activations as binary
            inp = (inp + 1) / 2
            export_idt = DataType["BINARY"]
        else:
            export_idt = self.get_input_datatype()

        reshaped_input = inp.reshape(folded_ishape)
        np.save(os.path.join(code_gen_dir, "input_0.npy"), reshaped_input)

        if mode == "cppsim":
            # execute the precompiled model
            super().exec_precompiled_singlenode_model()
            # load output npy file
            super().npy_to_dynamic_output(context)
            assert (
                context[node.output[0]].shape == exp_oshape
            ), "cppsim \
            did not produce expected output shape"
        elif mode == "rtlsim":
            sim = self.get_rtlsim()
            nbits = self.get_instream_width()
            rtlsim_inp = npy_to_rtlsim_input(
                "{}/input_0.npy".format(code_gen_dir), export_idt, nbits
            )
            super().reset_rtlsim(sim)
            super().toggle_clk(sim)
            rtlsim_output = self.rtlsim(sim, rtlsim_inp)
            odt = export_idt
            target_bits = odt.bitwidth()
            packed_bits = self.get_outstream_width()
            out_npy_path = "{}/output.npy".format(code_gen_dir)
            out_shape = self.get_folded_output_shape()
            rtlsim_output_to_npy(
                rtlsim_output, out_npy_path, odt, out_shape, packed_bits, target_bits
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
        # binary -> bipolar if needed
        if self.get_output_datatype() == DataType["BIPOLAR"]:
            out = context[node.output[0]]
            out = 2 * out - 1
            context[node.output[0]] = out
        assert (
            context[node.output[0]].shape == exp_oshape
        ), """Output
        shape doesn't match expected shape (1, ofm_dim, ofm_dim, ifm_ch)."""
