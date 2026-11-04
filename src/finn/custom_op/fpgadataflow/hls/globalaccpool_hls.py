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

from finn.custom_op.fpgadataflow.globalaccpool import GlobalAccPool
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.util.data_packing import hls_datatype_str


class GlobalAccPool_hls(GlobalAccPool, HLSBackend):
    """Class that corresponds to finn-hlslib AccPool_Batch function."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(GlobalAccPool.get_nodeattr_types(self))
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
            self.get_nodeattr("NumChannels")
            self.get_nodeattr("PE")
            self.get_nodeattr("inputDataType")
            info_messages.append("All necessary attributes exist")
        except Exception:
            info_messages.append("""The required GlobalAccPool_Batch attributes do not exist.""")

        # verify that input data is 2D
        if len(self.get_nodeattr("numInputVectors")) != 3:
            info_messages.append("""GlobalAccPool_Batch requires 2D data input.""")
            raise Exception

        return info_messages

    def execute_node(self, context, graph):
        HLSBackend.execute_node(self, context, graph)

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "pool.hpp"']

    def defines(self, var):
        self.code_gen_dict["$DEFINES$"] = []

    def docompute(self):
        self.code_gen_dict["$DOCOMPUTE$"] = [
            """AccPool_Batch<{}, {}, {}, {}, {}> (in0_V, out0_V);""".format(
                self.get_normal_input_shape()[1],
                self.get_nodeattr("NumChannels"),
                hls_datatype_str(self.get_input_datatype()),
                self.get_nodeattr("PE"),
                hls_datatype_str(self.get_output_datatype()),
            )
        ]

    def timeout_value(self):
        """AccPool_Batch is function-pipelined and processes one input per call.
        It needs NumChannels/PE * ImgDim^2 calls before the first output, so the
        default timeout of 1000 is insufficient for large configurations."""
        img_dim = self.get_normal_input_shape()[1]
        num_ch = self.get_nodeattr("NumChannels")
        pe = self.get_nodeattr("PE")
        total_inputs = (num_ch // pe) * img_dim * img_dim
        self.code_gen_dict["$TIMEOUT_VALUE$"] = [str(total_inputs + 1000)]

    def ipgen_extra_directives(self):
        "Return a list of extra tcl directives for HLS synthesis."
        return ["config_compile -pipeline_style flp"]

    def blackboxfunction(self):
        pe = self.get_nodeattr("PE")
        idt = hls_datatype_str(self.get_input_datatype())
        odt = hls_datatype_str(self.get_output_datatype())
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            "void {}(hls::stream<tensor<{}, {}>> &in0_V, "
            "hls::stream<tensor<{}, {}>> &out0_V)".format(self.onnx_node.name, idt, pe, odt, pe)
        ]
