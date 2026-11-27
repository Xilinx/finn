# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Tree model of the piecewise polynomial activation (``finn-rtllib/pwpolyf``)."""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from finn.transformation.fpgadataflow.fifo_sizing.tav import (
    CharacteristicNode,
    passthrough_characteristic,
)
from finn.transformation.fpgadataflow.fifo_sizing.tree_models.base import (
    TAV_PIPELINE_LATENCY,
    TreeModel,
    register,
)

if TYPE_CHECKING:
    # hwcustomop imports the tree models
    from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


@register("PWPolyF")
class PWPolyFModel(TreeModel):
    #: cycles of one DSPFP32 FMA stage of the Horner chain (pwpolyf.sv DSP_LAT)
    DSP_LATENCY = 4
    #: input sidestep register plus the output queue read
    IO_LATENCY = 2

    def schedule(self, op: HWCustomOp) -> CharacteristicNode | None:
        """One word in and one out per cycle, for the whole frame.

        ``pwpolyf.sv`` is a fully pipelined Horner chain: a word enters the
        sidestep register, runs through ``degree`` FMA stages and leaves through
        a credit-backed output queue, so with the output ready it takes a word
        every cycle and nothing in it depends on the previous word. Consecutive
        frames run straight into each other, so the period is the frame's
        ``prod(folded_output_shape[:-1])`` words, the same as ``get_exp_cycles``.

        The pipeline depth (``extra_latency``) is not part of the period: it is
        paid once, when the stream starts. rtlsim's period carries a fifth of it
        (five frames, one fill), so it is ``latency // 5`` cycles longer.

        Checked against node rtlsim for PE 1, 4 and 8 and degree 2 and 8.
        """
        dim = int(np.prod(op.get_folded_output_shape()[:-1]))
        if dim < 1:
            return None
        return passthrough_characteristic(dim, "PWPolyF word")

    def latency(self, op: HWCustomOp) -> int:
        """Cycles from accepting an input word to emitting its result:
        ``degree`` DSP stages plus the input and output registers (node rtlsim:
        10 cycles at degree 2, 34 at degree 8)."""
        return op.get_nodeattr("degree") * self.DSP_LATENCY + self.IO_LATENCY

    def extra_latency(self, op: HWCustomOp) -> int:
        """The pipeline depth beyond ``TAV_PIPELINE_LATENCY``, which a high
        polynomial degree exceeds."""
        return max(0, self.latency(op) - TAV_PIPELINE_LATENCY)
