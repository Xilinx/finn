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
from finn.custom_op.fpgadataflow.labelselect import LabelSelect


class LabelSelect_hls(LabelSelect, HLSBackend):
    """Class that corresponds to finn-hlslib LabelSelect_Batch function."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(LabelSelect.get_nodeattr_types(self))
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

        # verify that all necessary attributes exist
        try:
            self.get_nodeattr("code_gen_dir_cppsim")
            self.get_nodeattr("executable_path")
            self.get_nodeattr("Labels")
            self.get_nodeattr("PE")
            self.get_nodeattr("K")
            self.get_nodeattr("inputDataType")
            self.get_nodeattr("outputDataType")
            info_messages.append("All necessary attributes exist")
        except Exception:
            info_messages.append("""The required LabelSelect_Batch attributes do not exist.""")

        # verify that input data is 1D
        if len(self.get_nodeattr("numInputVectors")) > 1:
            info_messages.append("""LabelSelect_Batch requires 1D data input.""")
            raise Exception

        return info_messages

    def execute_node(self, context, graph):
        HLSBackend.execute_node(self, context, graph)
        # TopK ind output normally uses TensorProto.INT64, which
        # can cause issues for the node-by-node simulation in FINN
        # (as the custom DataType system always assumes float containers)
        # so cast the output to int64
        outp = self.onnx_node.output[0]
        ret = context[outp]
        context[outp] = ret.astype(np.int64)

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "topk.hpp"']

    def defines(self, var):
        self.code_gen_dict["$DEFINES$"] = []

    def read_npy_data(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        npy_type = "float"
        npy_in = "%s/input_0.npy" % code_gen_dir
        self.code_gen_dict["$READNPYDATA$"] = []

        self.code_gen_dict["$READNPYDATA$"].append(
            'npy2tensorstream<%s>("%s", in0_V);'
            % (
                npy_type,
                npy_in,
            )
        )

    def docompute(self):
        self.code_gen_dict["$DOCOMPUTE$"] = [
            """LabelSelect_Batch<{}, {}, {}, {}, {} > (in0_V, out0_V);""".format(
                self.get_nodeattr("Labels"),
                self.get_nodeattr("PE"),
                self.get_nodeattr("K"),
                self.get_input_datatype().get_hls_datatype_str(),
                self.get_output_datatype().get_hls_datatype_str(),
            )
        ]

    def ipgen_extra_directives(self):
        "Return a list of extra tcl directives for HLS synthesis."
        return ["config_compile -pipeline_style frp"]

    def blackboxfunction(self):
        pe = self.get_nodeattr("PE")
        idt = self.get_input_datatype().get_hls_datatype_str()
        odt = self.get_output_datatype().get_hls_datatype_str()
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            "void {}(hls::stream<tensor<{}, {}>> &in0_V, "
            "hls::stream<tensor<{}, 1>> &out0_V)".format(self.onnx_node.name, idt, pe, odt)
        ]
