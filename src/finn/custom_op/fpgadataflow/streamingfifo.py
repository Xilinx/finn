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
import warnings
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.basic import is_versal
from finn.util.resource_models import _fifo_cost, _resolve


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
                # requested FPGA resource for the storage, passed to fifo.sv (with srl
                # mapped to the RTL's "shift" token at the codegen boundary):
                # auto (its RAM_STYLE_EFF ladder decides), srl (SRL shift register),
                # block (BRAM), distributed (LUTRAM) or ultra (URAM, on UltraScale+/Versal)
                "ram_style": (
                    "s",
                    False,
                    "auto",
                    {"auto", "srl", "block", "distributed", "ultra"},
                ),
                # whether the maxcount occupancy output is exposed on the wrapper
                "depth_monitor": ("i", False, 0),
                # the FIFO does not need its own FIFOs
                "inFIFODepths": ("ints", False, [0]),
                "outFIFODepths": ("ints", False, [0]),
                "debug_log_path": ("s", False, ""),
            }
        )

        return my_attrs

    def resolve_ram_style(self):
        """Predicts which of srl/distributed/block/ultra fifo.sv will elaborate.

        generate_hdl() forwards ram_style to the RTL untouched, so this decides
        nothing: it reproduces the RAM_STYLE_EFF ladder so that resource estimation,
        the build report and the folding config describe what actually gets built.
        The ladder is a function of depth, width and the request alone, so unlike the
        estimators this needs no fpgapart."""
        requested = self.get_nodeattr("ram_style")
        depth = self.get_nodeattr("depth")
        W = self.get_instream_width_padded()
        style = _resolve(depth, W, requested)
        # fifo.sv checks depth<=33 before it checks RAM_STYLE!=auto (fifo.sv:85-86), so
        # at this depth any explicit memory request is forced to a shift register
        # regardless. Warn for every explicit style (block/ultra/distributed alike) so
        # the override is not silent
        if depth <= 33 and requested not in ("auto", "srl"):
            warnings.warn(
                "%s: ram_style=%s requested but depth %d <= 33 is built as a shift "
                "register regardless" % (self.onnx_node.name, requested, depth)
            )
        # srl is never auto-selected past 257, so a deeper one is an explicit
        # ram_style=srl request, whose name does not suggest the LUT cost it carries.
        if style == "srl" and depth > 257:
            warnings.warn(
                "%s: ram_style=srl at depth %d costs roughly %d LUTs of shift "
                "register; consider distributed/block/ultra instead"
                % (self.onnx_node.name, depth, _fifo_cost(depth, W, "srl").lut)
            )
        return style

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

    def get_normal_input_shape(self, ind=0):
        assert self.get_nodeattr("depth") >= 1, """Depth is too low"""
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

    def get_ram_style(self):
        """Returns the storage style this FIFO is built with.

        Derived on demand rather than stored: the style depends only on attributes this
        node already carries, so recomputing cannot go stale the way a recorded value
        would when set_fifo_depths changes the depth."""
        return self.resolve_ram_style()

    def get_fifo_cost(self, fpgapart):
        """Returns the FifoCost of this node as fifo.sv implements it.

        Needs the fpgapart because BRAM/URAM aspects differ on Versal."""
        return _fifo_cost(
            self.get_nodeattr("depth"),
            self.get_instream_width_padded(),
            self.get_ram_style(),
            is_versal(fpgapart),
        )

    def bram_estimation(self, fpgapart):
        """Calculates resource estimation for BRAM"""
        return self.get_fifo_cost(fpgapart).bram

    def uram_estimation(self, fpgapart):
        """Calculates resource estimation for URAM"""
        return self.get_fifo_cost(fpgapart).uram

    def lut_estimation(self, fpgapart):
        """Calculates resource estimations for LUTs"""
        return self.get_fifo_cost(fpgapart).lut

    def bram_efficiency_estimation(self, fpgapart):
        bram_est = self.bram_estimation(fpgapart)
        if bram_est == 0:
            return 1
        wbits = self.get_instream_width_padded() * self.get_nodeattr("depth")
        return wbits / (bram_est * 18 * 1024)

    def uram_efficiency_estimation(self, fpgapart):
        # every URAM288 aspect holds 288 Kib, so this capacity is correct on the
        # Versal ladder too; narrow words show up as a smaller uram_estimation()
        uram_est = self.uram_estimation(fpgapart)
        if uram_est == 0:
            return 1
        wbits = self.get_instream_width_padded() * self.get_nodeattr("depth")
        return wbits / (uram_est * 72 * 4096)
