# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Resolution of the memstream implementation of the specialized layers."""

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from typing import Optional

from finn.util.fpgadataflow import is_hls_node, is_rtl_node


class ResolveMemHighPerformance(Transformation):
    """Resolve the mem_high_performance attribute of all layers with memstream
    support that are still set to "auto". A given force value selects the
    high-performance (True) or lean (False) memstream throughout. Otherwise, the
    choice follows the memstream clock derived from the target clock period
    clk_ns (see HWCustomOp.get_mem_high_performance()). Layers with an explicit
    "on" or "off" setting are left untouched."""

    def __init__(self, clk_ns: float, force: Optional[bool] = None):
        super().__init__()
        self.clk_ns = clk_ns
        self.force = force

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        for node in model.graph.node:
            if not (is_hls_node(node) or is_rtl_node(node)):
                continue
            inst = getCustomOp(node)
            if not inst.supports_memstream():
                continue
            if inst.get_nodeattr("mem_high_performance") != "auto":
                continue
            if self.force is None:
                pumped = 0
                if "pumpedMemory" in inst.get_nodeattr_types():
                    pumped = inst.get_nodeattr("pumpedMemory")
                high_performance = inst.get_mem_high_performance(self.clk_ns, pumped)
            else:
                high_performance = self.force
            inst.set_nodeattr("mem_high_performance", "on" if high_performance else "off")
        return (model, False)
