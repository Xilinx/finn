# Copyright (c) 2020-2022, Xilinx, Inc.
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
import warnings
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.kernels.kernel_registry import gkr
from finn.util.kernel_util import get_node_attr

# the IODMA inerfaces a memory-mapped AXI interface and an AXI stream
# direction "in": pulls data from AXI-MM to AXI stream
# direction "out": pushes data from AXI stream to AXI-MM

# DMA Addressing
# - burst mode can be "wrap" or "increment"
# - "increment" bursts will increment the address when moving to the next image
# - "wrap" bursts will reinitialize the address to the start address,
#   and are useful for e.g. streaming weights, where the same buffer is
#   repeatedly read into the FPGA
# - no additional alignment restrictions beyond anything specified in the AXI spec

# Interfaces
# - AXI-MM name specified by intfName unless this is set to "" (empty, the default)
#   in which case output AXI-MM are named "out0_V" and input AXI-MM are named "in0_V"
# - AXI-MM interface width (in bits) is specified by intfWidth
# - AXI-Stream interface width (in bits) is specified by streamWidth
# - If inftWidth and streamWidth are not equal, the DMA core performs
#   width conversion by going up to the least common multiple of bitwidths
#   e.g. intfWidth=32b -> 96b -> sreamWidth=24b
# - transfers occur in multiples of the AXI-MM interface width, therefore
#   the total number of bits in the tensor must be a multiple of intfWidth
# - transfers occur in multiples of the AXI-Stream interface width, therefore
#   the total number of bits in the tensor must be a multiple of streamWidth
# - both interface widths must be a multiple of 8b (AXI protocol requirement)
# - in most systems, intfWidth is also restricted to a power of 2 (e.g. Vitis)
#   but this is not universal so we don't check here explicitly

# Input/output tensor sizes shapes
# - The data being moved is a tensor of shape numInputVectors+[NumChannels]
# - The data type of the tensor elements is specified by dataType
# - on the stream side
#       -the normal shape is the same as the ONNX tensor attached to it
#       -the folded shape is computed from the stream width and normal shape
# - on the AXI-MM side
#       -the normal shape is the same as the one on the stream side
#       -the folded shape is not defined


class IODMA(HWCustomOp):
    """Class that corresponds to finn-hlslib DMA function(s)."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            "NumChannels": ("i", True, 0),
            # FINN input datatype
            "dataType": ("s", True, ""),
            # Width of input or output stream
            "streamWidth": ("i", False, 32),
            # DMA-specific parameters
            # width of axi-mm interface
            "intfWidth": ("i", False, 32),
            # burst mode for axi-mm interface (wrap used for DRAM weights)
            "burstMode": ("s", False, "increment", {"wrap", "increment"}),
            # IODMA direction: in = read from DRAM, out = write to DRAM
            "direction": ("s", False, "in", {"in", "out"}),
            # shape describing input vecs per execution
            "numInputVectors": ("ints", False, [1]),
            # name of axi-mm interface
            "intfName": ("s", False, ""),
        }
        my_attrs.update(HWCustomOp.get_nodeattr_types(self))
        return my_attrs

    def infer_node_datatype(self, model):
        node = self.onnx_node
        kernel = gkr.kernel(node.op_type, get_node_attr(node, model))
        idt = model.get_tensor_datatype(node.input[0])
        if idt != kernel.get_input_datatype():
            warn_str = "inputDataType changing for %s: %s -> %s " % (
                node.name,
                str(kernel.get_input_datatype()),
                str(idt),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("dataType", idt.name)
        model.set_tensor_datatype(node.output[0], idt)

    def execute_node(self, context, graph):
        pass
