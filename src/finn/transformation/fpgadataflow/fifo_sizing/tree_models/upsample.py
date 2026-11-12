# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Tree model of the nearest-neighbour upsampler."""

import numpy as np
from math import gcd

from finn.transformation.fpgadataflow.fifo_sizing.tav import flat_characteristic_leaf
from finn.transformation.fpgadataflow.fifo_sizing.tree_models.base import (
    TreeModel,
    register,
)


class Stepper:
    """One dimension's input index generator, as the upsampling loop nest runs it.

    A dimension that is not upsampled steps its input index every output step; one
    upsampled from a single input element holds that element for the whole output
    extent; any other ratio walks the input index with Bresenham's line algorithm.
    ``replays()`` answers whether the current input element is needed again, which
    is what decides when its buffer slot may be refilled.
    """

    def __init__(self, size_in, size_out):
        self.size_out = size_out
        self.pos = 0
        if size_in == size_out:
            self.kind = "copy"
        elif size_in == 1:
            self.kind = "hold"
        else:
            self.kind = "bresenham"
            common = gcd(size_in, size_out)
            self.num, self.den = size_in // common, size_out // common
            self.err = (3 * self.num - 2 * self.den) >> 1

    def replays(self):
        if self.kind == "copy":
            return False
        if self.kind == "hold":
            return self.pos != self.size_out - 1
        return self.err < 0

    def step(self):
        """Advance one output position; returns True when the dimension wraps."""
        if self.kind == "bresenham":
            self.err += self.num if self.err < 0 else self.num - self.den
        self.pos += 1
        wrapped = self.pos == self.size_out
        if wrapped:
            self.pos = 0
        return wrapped


def release_mask(dims):
    """Which output words release the input word they were built from.

    ``dims`` lists ``(input extent, output extent)`` from the slowest- to the
    fastest-changing dimension of the output raster. An input word may be
    overwritten once no dimension will replay it, so the mask is one exactly at the
    output positions where every dimension is done with its current input index. It
    holds ``prod(input extents)`` ones, one per input word of a frame.
    """
    steppers = [Stepper(size_in, size_out) for size_in, size_out in dims]
    num_out = int(np.prod([size_out for _, size_out in dims]))
    mask = np.zeros(num_out, dtype=np.int8)
    for pos in range(num_out):
        mask[pos] = 0 if any(s.replays() for s in steppers) else 1
        wrapped = True
        for stepper in reversed(steppers):
            if not wrapped:
                break
            wrapped = stepper.step()
    return mask


@register("UpsampleNearestNeighbour")
class UpsampleModel(TreeModel):
    def input_need(self, op, ind=0):
        """Output pixel (y, x) needs the input up to its nearest neighbour."""
        hi, wi = op.get_nodeattr("HI"), op.get_nodeattr("WI")
        ho, wo = op.get_nodeattr("HO"), op.get_nodeattr("WO")
        words = op.get_nodeattr("NumChannels") // op.get_nodeattr("SIMD")
        b, rest = np.divmod(np.arange(op.get_nodeattr("batchSize") * ho * wo), ho * wo)
        y, x = np.divmod(rest, wo)
        pixels = b * hi * wi + (y * hi // ho) * wi + x * wi // wo + 1
        return np.repeat(pixels * words, words)


@register("UpsampleNearestNeighbour_hls")
class UpsampleHlsModel(UpsampleModel):
    #: cycles between output words: finn-hlslib's ``upsample_nn`` is
    #: free-running (``pipeline II=1 style=frp``), one output word per call
    UPSAMPLE_II = 1
    #: output words between the one that releases a buffer slot and the read of
    #: the input word that refills it: ``upsample_nn`` gates reads on a write
    #: pointer delayed by ``WP_DELAY = 4`` calls. A TAV comparison cannot check
    #: this value: the output is written every cycle, so any constant read offset
    #: is absorbed by the window's phase.
    UPSAMPLE_READ_LAG = 4

    def schedule(self, op):
        """Writes the output raster at one word per cycle and refills the buffer
        behind it.

        The layer holds an input frame in a buffer and walks the output raster over
        it, emitting one folded word per cycle (``UPSAMPLE_II``). Consecutive frames
        run back to back, so the period is the output frame ``get_exp_cycles``
        counts.

        An input word is read once its buffer slot comes free, which happens on the
        output word that used it for the last time: the last output row that
        replicates its row and, within that row, the last output column that
        replicates its column. That gives exactly one read per input word of a
        frame, placed ``UPSAMPLE_READ_LAG`` words after the output word that
        released it (wrapping into the next frame at the end of one). Reads
        therefore inherit the bunching of the replication pattern, which is what
        makes them worth stating rather than spreading evenly.

        The buffer is filled ahead once out of reset; that head start is not
        modelled -- it shifts which input word a read carries, not when reads
        happen.

        Valid for the HLS implementation, for any output extent at least as large
        as its input extent, any SIMD dividing NumChannels, and any batch size.
        Checked against node rtlsim (exact up to the window's phase) for the
        configurations of ``test_fpgadataflow_analytical_characterization_upsampler``.
        """
        simd = op.get_nodeattr("SIMD")
        num_channels = op.get_nodeattr("NumChannels")
        batch = op.get_nodeattr("batchSize")
        hi, wi = op.get_nodeattr("HI"), op.get_nodeattr("WI")
        ho, wo = op.get_nodeattr("HO"), op.get_nodeattr("WO")
        if simd < 1 or num_channels % simd != 0:
            return None
        folds = num_channels // simd
        if min(hi, wi, ho, wo, folds, batch) < 1 or ho < hi or wo < wi:
            return None
        # the loop nest walks (row, column, channel fold), and the channel folds of
        # one pixel are neither replicated nor reordered
        released = np.tile(release_mask([(hi, ho), (wi, wo), (folds, folds)]), batch)
        num_out = released.size
        ii = self.UPSAMPLE_II
        period = ii * num_out
        wr = np.zeros(period, dtype=np.int8)
        wr[ii * np.arange(num_out)] = 1
        rd = np.zeros(period, dtype=np.int8)
        rd[(ii * np.flatnonzero(released) + self.UPSAMPLE_READ_LAG) % period] = 1
        return flat_characteristic_leaf(rd, wr, "Upsample raster")
