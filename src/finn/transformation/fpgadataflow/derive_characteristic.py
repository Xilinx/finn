# Copyright (C) 2022, Xilinx, Inc.
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


import qonnx.custom_op.registry as registry
import warnings
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import NodeLocalTransformation

from finn.util.fpgadataflow import is_hls_node, is_rtl_node


class DeriveCharacteristic(NodeLocalTransformation):
    """For each node in the graph, run rtlsim to obtain the i/o
    characteristic function for FIFO sizing and set the attribute.
    It is assumed that the PrepareRTLSim transformation was already
    called on the graph.

    This transformation performs rtlsim for each node, so it will run for
    some time (minutes to hours depending on configuration).

    * period (int) desired period over which the characteristic function
      will be derived.

    * num_workers (int or None) number of parallel workers, see documentation in
      NodeLocalTransformation for more details.
    """

    def __init__(
        self, model, period, strategy, fpga_part, clk_period, num_workers=None
    ):
        super().__init__(num_workers=num_workers)
        self.model = model
        self.period = period
        self.strategy = strategy
        self.fpga_part = fpga_part
        self.clk_period = clk_period

    def applyNodeLocal(self, node):
        op_type = node.op_type
        if is_hls_node(node) or is_rtl_node(node):
            try:
                # lookup op_type in registry of CustomOps
                inst = registry.getCustomOp(node)

                inst.derive_characteristic_fxns(
                    model=self.model,
                    period=self.period,
                    strategy=self.strategy,
                    fpga_part=self.fpga_part,
                    clk_period=self.clk_period,
                    op_type=op_type,
                )
            except KeyError:
                # exception if op_type is not supported
                raise Exception("Custom op_type %s is currently not supported." % op_type)
        return (node, False)

    def apply(self, model: ModelWrapper):
        (model, run_again) = super().apply(model)
        return (model, run_again)


class DeriveFIFOSizes(NodeLocalTransformation):
    """Prerequisite: DeriveCharacteristic already called on graph.
    For each node in the graph, use the accumulated I/O characteristic function
    to perform FIFO sizing, setting the in/outFIFODepths attributes of HLSCustomOp
    nodes.

    * num_workers (int or None) number of parallel workers, see documentation in
      NodeLocalTransformation for more details.
    """

    def __init__(self, num_workers=None, io_fifo_depth=32):
        super().__init__(num_workers=num_workers)
        self.io_fifo_depth = io_fifo_depth

    def applyNodeLocal(self, node):
        op_type = node.op_type
        if is_hls_node(node) or is_rtl_node(node):
            try:
                # lookup op_type in registry of CustomOps
                prod = registry.getCustomOp(node)
                assert not (op_type.startswith("StreamingFIFO")), "Found existing FIFOs"
                period = prod.get_nodeattr("io_chrc_period")
                prod_chrc = prod.get_nodeattr("io_chrc_out")[0]
                assert len(prod_chrc) == 2 * period, "Found unexpected characterization attribute"
                if any([x > 2 for x in prod.get_nodeattr("outFIFODepths")]):
                    # FIFO depth already set, can skip this node
                    return (node, False)

                # find consumers
                model = self.ref_input_model
                out_fifo_depths = []
                for output_name in node.output:
                    cons_node = model.find_consumer(output_name)
                    if cons_node is None:
                        # could be final node, will be overridden if so
                        # need an entry in the list anyway
                        out_fifo_depths.append(self.io_fifo_depth)
                        continue
                    cons = registry.getCustomOp(cons_node)
                    cons_chrc = cons.get_nodeattr("io_chrc_in")[0]
                    # find minimum phase shift satisfying the constraint
                    pshift_min = period - 1
                    for pshift_cand in range(period):
                        prod_chrc_part = prod_chrc[pshift_cand:period]
                        cons_chrc_part = cons_chrc[: period - pshift_cand]
                        if (prod_chrc_part >= cons_chrc_part).all():
                            pshift_min = pshift_cand
                            break
                    prod_chrc_part = prod_chrc[pshift_min : (pshift_min + period)]
                    cons_chrc_part = cons_chrc[:period]
                    fifo_depth = int((prod_chrc_part - cons_chrc_part).max())
                    out_fifo_depths.append(fifo_depth)
                # set output FIFO depth for this (producing) node
                # InsertFIFO looks at the max of (outFIFODepths, inFIFODepths)
                # for each tensor
                prod.set_nodeattr("outFIFODepths", out_fifo_depths)

                # finally, check node inputs to ensure FIFOs are added to
                # any top-level inputs (at least self.io_fifo_depth deep)
                in_fifo_depths = prod.get_nodeattr("inFIFODepths")
                for i, input_name in enumerate(node.input):
                    if input_name in [x.name for x in model.graph.input]:
                        in_fifo_depths[i] = max(self.io_fifo_depth, in_fifo_depths[i])
                prod.set_nodeattr("inFIFODepths", in_fifo_depths)

            except KeyError:
                # exception if op_type is not supported
                raise Exception("Custom op_type %s is currently not supported." % op_type)
        return (node, False)
