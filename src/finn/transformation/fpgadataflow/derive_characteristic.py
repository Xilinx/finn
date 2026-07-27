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
from qonnx.transformation.base import NodeLocalTransformation

from finn.util.fpgadataflow import is_hls_node, is_rtl_node


def _find_minimum_phase_shift(prod_chrc, cons_chrc, period):
    """Find the first phase shift where production covers consumption."""

    def is_valid(pshift):
        return (prod_chrc[pshift:period] >= cons_chrc[: period - pshift]).all()

    # Validity is monotonic because the characteristics are cumulative: once a
    # shift is valid, every larger shift compares later production values over
    # a shorter interval. Binary search avoids a linear scan over long periods.
    low = 0
    high = period - 1
    if not is_valid(high):
        return high
    while low < high:
        candidate = (low + high) // 2
        if is_valid(candidate):
            high = candidate
        else:
            low = candidate + 1
    return low


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

    * skip_node_names (iterable of str or None) nodes which already have an
      explicit FIFO sizing policy and therefore do not require RTL simulation.
    """

    def __init__(self, period, num_workers=None, skip_node_names=None):
        super().__init__(num_workers=num_workers)
        self.period = period
        self.skip_node_names = set(skip_node_names or [])

    def applyNodeLocal(self, node):
        if node.name in self.skip_node_names:
            return (node, False)
        op_type = node.op_type
        if is_hls_node(node) or is_rtl_node(node):
            try:
                # lookup op_type in registry of CustomOps
                inst = registry.getCustomOp(node)
                inst.derive_characteristic_fxns(period=self.period)
            except KeyError:
                # exception if op_type is not supported
                raise Exception("Custom op_type %s is currently not supported." % op_type)
        return (node, False)


class DeriveFIFOSizes(NodeLocalTransformation):
    """Prerequisite: DeriveCharacteristic already called on graph.
    For each node in the graph, use the accumulated I/O characteristic function
    to perform FIFO sizing, setting the in/outFIFODepths attributes of HLSCustomOp
    nodes.

    * num_workers (int or None) number of parallel workers, see documentation in
      NodeLocalTransformation for more details.

    * output_fifo_depth_overrides (dict or None) maps node names to output-index
      and FIFO-depth mappings. Overridden edges do not require consumer
      characteristics.
    """

    def __init__(
        self,
        num_workers=None,
        io_fifo_depth=32,
        output_fifo_depth_overrides=None,
    ):
        super().__init__(num_workers=num_workers)
        self.io_fifo_depth = io_fifo_depth
        self.output_fifo_depth_overrides = output_fifo_depth_overrides or {}

    def applyNodeLocal(self, node):
        op_type = node.op_type
        if is_hls_node(node) or is_rtl_node(node):
            try:
                # lookup op_type in registry of CustomOps
                prod = registry.getCustomOp(node)
                assert not (op_type.startswith("StreamingFIFO")), "Found existing FIFOs"
                node_overrides = self.output_fifo_depth_overrides.get(node.name, {})
                model = self.ref_input_model
                if len(node_overrides) == len(node.output):
                    out_fifo_depths = [node_overrides[i] for i in range(len(node.output))]
                else:
                    period = prod.get_nodeattr("io_chrc_period")
                    if any([x > 2 for x in prod.get_nodeattr("outFIFODepths")]):
                        # FIFO depth already set, can skip this node
                        return (node, False)

                    out_fifo_depths = []
                    prod_chrcs = prod.get_io_chrc_out()
                    for output_index, output_name in enumerate(node.output):
                        if output_index in node_overrides:
                            out_fifo_depths.append(node_overrides[output_index])
                            continue
                        prod_chrc_index = (
                            output_index if len(prod_chrcs) == len(node.output) else 0
                        )
                        prod_chrc = prod_chrcs[prod_chrc_index]
                        assert (
                            len(prod_chrc) == 2 * period
                        ), "Found unexpected characterization attribute"
                        cons_node = model.find_consumer(output_name)
                        if cons_node is None:
                            # could be final node, will be overridden if so
                            # need an entry in the list anyway
                            out_fifo_depths.append(self.io_fifo_depth)
                            continue
                        cons = registry.getCustomOp(cons_node)
                        cons_chrcs = cons.get_io_chrc_in()
                        cons_input_index = list(cons_node.input).index(output_name)
                        cons_chrc_index = (
                            cons_input_index if len(cons_chrcs) == len(cons_node.input) else 0
                        )
                        cons_chrc = cons_chrcs[cons_chrc_index]
                        # find minimum phase shift satisfying the constraint
                        pshift_min = _find_minimum_phase_shift(prod_chrc, cons_chrc, period)
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
