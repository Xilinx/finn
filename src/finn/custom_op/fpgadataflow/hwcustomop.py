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

try:
    import finn_xsi.adapter as finnxsi
except ModuleNotFoundError:
    finnxsi = None

import numpy as np
import os
import warnings
from abc import abstractmethod
from qonnx.custom_op.base import CustomOp
from qonnx.util.basic import roundup_to_integer_multiple

from finn.util.basic import get_liveness_threshold_cycles, is_versal
from finn.kernels.kernel_registry import gkr
from finn.util.kernel_util import get_node_attr


class HWCustomOp(CustomOp):
    """HWCustomOp class all custom ops that can be implemented with either
    HLS or RTL backend are based on. Contains different functions every fpgadataflow
    custom node should have. Some as abstract methods, these have to be filled
    when writing a new fpgadataflow custom op node."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)
        self.code_gen_dict = {}

    def get_nodeattr_types(self):
        return {
            "backend": ("s", True, "fpgadataflow"),
            "preferred_impl_style": ("s", False, "", {"", "hls", "rtl"}),
            "code_gen_dir_ipgen": ("s", False, ""),
            "ipgen_path": ("s", False, ""),
            "ip_path": ("s", False, ""),
            "ip_vlnv": ("s", False, ""),
            "exec_mode": ("s", False, "", {"", "rtlsim", "cppsim"}),
            "cycles_rtlsim": ("i", False, 0),
            "cycles_estimate": ("i", False, 0),
            "rtlsim_trace": ("s", False, ""),
            "res_estimate": ("s", False, ""),
            "res_synth": ("s", False, ""),
            "rtlsim_so": ("s", False, ""),
            # partitioning info
            # ID of SLR to which the Op is attached in Vitis builds
            # Set to -1 as 'don't care'
            "slr": ("i", False, -1),
            # Vitis memory port to which any AXI-MM interface
            # of this Op should be attached in Vitis builds
            # E.g.: "DDR[0]", "HBM[0]", "PLRAM[0]"
            "mem_port": ("s", False, ""),
            # Partition to which the Op belongs; all Ops with the
            # same partition_id are stitched together
            # Users should avoid setting this attribute manually
            # and instead use the floorplan transform to set
            # partition IDs from Vitis design rules and SLR IDs
            "partition_id": ("i", False, 0),
            # ID of FPGA device to which this Op is allocated, in
            # a multi-FPGA setting
            "device_id": ("i", False, 0),
            # input and output FIFO depths for multi-I/O nodes
            "inFIFODepths": ("ints", False, [2]),
            "outFIFODepths": ("ints", False, [2]),
            "output_hook": ("s", False, ""),
            # accumulated characteristic function over two periods
            "io_chrc_in": ("t", False, np.asarray([], dtype=np.int32)),
            "io_chrc_out": ("t", False, np.asarray([], dtype=np.int32)),
            # the period for which the characterization was run
            "io_chrc_period": ("i", False, 0),
            # amount of zero padding inserted during chrc.
            "io_chrc_pads_in": ("ints", False, []),
            "io_chrc_pads_out": ("ints", False, []),
        }

    def make_shape_compatible_op(self, model):
        try:
            oshape = self.get_normal_output_shape()
        except:
            try:
                kernel = gkr.kernel(self.onnx_node.op_type, get_node_attr(self.onnx_node, model))
                oshape = kernel.get_normal_output_shape()
            except:
                raise RuntimeError(f"Neither {self.__class__.__name__}.get_normal_output_shape nor {kernel.__class__.__name__}.get_normal_output_shape exist for HWCustomOp.make_shape_compatible_op.")
        # implement tensor with correct shape
        return super().make_const_shape_op(oshape)

    def verify_node(self):
        """Can be implemented to verify that all attributes the node needs
        are there and that particular attributes are set correctly. Can also
        check if the number of inputs is equal to the expected number."""
        pass
