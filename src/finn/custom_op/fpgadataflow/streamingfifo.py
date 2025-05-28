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
import math
import numpy as np
import warnings
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class StreamingFIFO(HWCustomOp):
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = super().get_nodeattr_types()
        my_attrs.update(
            {
                # FIFO depth
                "depth": ("i", True, 0),
                # folded shape of input/output
                "folded_shape": ("ints", True, []),
                # normal shape of input/output
                "normal_shape": ("ints", True, []),
                # FINN DataTypes for inputs/outputs
                "dataType": ("s", True, ""),
                # FPGA resource type for FIFOs when impl_style is vivado
                # auto -- let Vivado decide
                # block -- use BRAM
                # distributed -- use LUTRAM
                # ultra -- use URAM (on UltraScale+)
                "ram_style": (
                    "s",
                    False,
                    "auto",
                    {"auto", "block", "distributed", "ultra"},
                ),
                # whether depth monitoring is enabled (impl_style=rtl only)
                "depth_monitor": ("i", False, 0),
                # the FIFO does not need its own FIFOs
                "inFIFODepths": ("ints", False, [0]),
                "outFIFODepths": ("ints", False, [0]),
            }
        )

        return my_attrs

    def infer_node_datatype(self, model):
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            warn_str = "inputDataType changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype()),
                str(idt),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("dataType", idt.name)
        # data type stays the same
        model.set_tensor_datatype(node.output[0], idt)

    def get_verilog_top_module_intf_names(self):
        ret = super().get_verilog_top_module_intf_names()
        is_rtl = self.get_nodeattr("impl_style") == "rtl"
        is_depth_monitor = self.get_nodeattr("depth_monitor") == 1
        if is_rtl and is_depth_monitor:
            ret["ap_none"] = ["maxcount"]
        return ret

    def get_normal_input_shape(self, ind=0):
        depth = self.get_adjusted_depth()
        assert depth >= 1, """Depth is too low"""
        if depth > 256 and self.get_nodeattr("impl_style") == "rtl":
            warnings.warn("Depth is high, set between 2 and 256 for efficient SRL implementation")
        return self.get_nodeattr("normal_shape")

    def get_normal_output_shape(self, ind=0):
        return self.get_normal_input_shape()

    def get_folded_input_shape(self, ind=0):
        return self.get_nodeattr("folded_shape")

    def get_folded_output_shape(self, ind=0):
        return self.get_nodeattr("folded_shape")

    def get_instream_width(self, ind=0):
        dtype = DataType[self.get_nodeattr("dataType")]
        folded_shape = self.get_nodeattr("folded_shape")
        in_width = folded_shape[-1] * dtype.bitwidth()
        return in_width

    def get_outstream_width(self, ind=0):
        dtype = DataType[self.get_nodeattr("dataType")]
        folded_shape = self.get_nodeattr("folded_shape")
        in_width = folded_shape[-1] * dtype.bitwidth()
        return in_width

    def get_input_datatype(self, ind=0):
        return DataType[self.get_nodeattr("dataType")]

    def get_output_datatype(self, ind=0):
        return DataType[self.get_nodeattr("dataType")]

    def execute_node(self, context, graph):
        node = self.onnx_node
        context[node.output[0]] = context[node.input[0]]

    def get_number_output_values(self):
        folded_oshape = self.get_folded_output_shape()
        return np.prod(folded_oshape[:-1])

    def bram_estimation(self):
        """Calculates resource estimation for BRAM"""
        impl = self.get_nodeattr("impl_style")
        ram_type = self.get_nodeattr("ram_style")
        depth = self.get_adjusted_depth()
        W = self.get_instream_width()

        if impl == "rtl" or (impl == "vivado" and ram_type != "block"):
            # Non-BRAM based implementation
            return 0

        if W == 1:
            return math.ceil(depth / 16384)
        elif W == 2:
            return math.ceil(depth / 8192)
        elif W <= 4:
            return (math.ceil(depth / 4096)) * (math.ceil(W / 4))
        elif W <= 9:
            return (math.ceil(depth / 2048)) * (math.ceil(W / 9))
        elif W <= 18 or depth > 512:
            return (math.ceil(depth / 1024)) * (math.ceil(W / 18))
        else:
            return (math.ceil(depth / 512)) * (math.ceil(W / 36))

    def uram_estimation(self):
        """Calculates resource estimation for URAM"""

        impl = self.get_nodeattr("impl_style")
        ram_type = self.get_nodeattr("ram_style")
        depth = self.get_adjusted_depth()
        W = self.get_instream_width()

        if impl == "rtl" or (impl == "vivado" and ram_type != "ultra"):
            # Non-BRAM based implementation
            return 0
        else:
            return (math.ceil(depth / 4096)) * (math.ceil(W / 72))

    def bram_efficiency_estimation(self):
        depth = self.get_adjusted_depth()
        W = self.get_instream_width()
        bram16_est = self.bram_estimation()
        if bram16_est == 0:
            return 1
        wbits = W * depth
        bram16_est_capacity = bram16_est * 36 * 512
        return wbits / bram16_est_capacity

    def lut_estimation(self):
        """Calculates resource estimations for LUTs"""
        impl = self.get_nodeattr("impl_style")
        ram_type = self.get_nodeattr("ram_style")
        depth = self.get_adjusted_depth()
        W = self.get_instream_width()

        address_luts = 2 * math.ceil(math.log(depth, 2))

        if impl == "rtl" or (impl == "vivado" and ram_type == "distributed"):
            ram_luts = (math.ceil(depth / 32)) * (math.ceil(W / 2))
        else:
            ram_luts = 0

        return int(address_luts + ram_luts)
