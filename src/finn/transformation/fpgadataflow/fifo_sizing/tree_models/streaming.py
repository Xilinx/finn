# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Tree models of the streaming operators: elementwise and other one-to-one
maps, padding and cropping, pooling, width conversion, split and concat."""

import numpy as np

from finn.transformation.fpgadataflow.fifo_sizing.tav import (
    CharacteristicNode,
    flat_characteristic_leaf,
    passthrough_characteristic,
)
from finn.transformation.fpgadataflow.fifo_sizing.tree_models.base import (
    TreeModel,
    register,
)


@register("ElementwiseBinaryOperation")
class ElementwiseBinaryModel(TreeModel):
    def schedule(self, op):
        """Token access vector model for every elementwise binary operation.

        The simplest shape a tree model takes: one flat phase of
        ``prod(folded_output_shape[:-1])`` cycles reading and writing one token
        each. That is not an approximation of this operator, it is its
        definition -- the op's ``get_exp_cycles()`` is the same product, i.e. the
        operator is II=1 with no wind-up phase to model, because an elementwise
        operation needs nothing more than the element it is handed. Every
        specialization (add, mul, bitshift, the comparisons, ...) differs
        only in the arithmetic applied to a token, never in when tokens move, so
        the model is registered for the base class rather than per op.

        Both inputs are described by the single row a tree model emits. For a
        two-stream join that is exact rather than a simplification: an
        elementwise operation consumes one token from each input to produce one
        output, so both input streams carry the identical schedule. When the
        right-hand side is a constant (``rhs_style == "const"``) there is only
        one stream to describe in the first place.
        """
        dim = int(np.prod(op.get_folded_output_shape()[:-1]))
        read_write = CharacteristicNode("passing elementwise layer", [(dim, [1, 1])], True)
        return CharacteristicNode("compute elementwise", [(1, read_write)], False)


@register("Requant")
class RequantModel(TreeModel):
    def schedule(self, op):
        """One token in and one out per cycle, for as many tokens as a frame has.

        Requantization is a scale, a round and a clamp on the token in hand, so
        there is nothing to wait for -- the same shape as the elementwise ops,
        and the same product ``get_exp_cycles`` returns. Both backends behave
        this way, and both run consecutive frames straight into each other:
        there is no gap at a frame boundary, so the two rows are one solid run.

        The period is the frame and does not include the pipeline fill, which
        costs a fixed handful of cycles once -- negligible against a frame of
        any size, but the length of a frame of only a few words.

        The rate assumes the loop pipelines at II=1, as ``get_exp_cycles`` does.
        A design that schedules slower stretches every token by that factor and
        takes proportionally longer.

        Valid for PE 1..16 on both backends.
        """
        dim = int(np.prod(op.get_folded_output_shape()[:-1]))
        if dim < 1:
            return None
        return passthrough_characteristic(dim, "requantize a token")


@register("Thresholding")
class ThresholdingModel(TreeModel):
    def schedule(self, op):
        """One folded word in and one out every cycle, for the whole frame.

        Thresholding is a pure map: ``ImgDim * NumChannels/PE`` folded words per
        frame, each read and written in the same pipelined iteration, with no
        accumulation and no rate change. So the schedule has no free parameters
        at all -- a solid run of read-and-write cycles as long as the frame -- and
        ``output_delay`` is 0 on both backends.

        The two branches exist only so that a hypothetical non-zero
        ``output_delay`` (a pipeline latency shifting writes behind reads) would
        still produce a well-formed schedule when it exceeds the frame itself.
        They are otherwise the same shape.

        The period is deliberately the folded word count and not the slightly
        longer one rtlsim records; the comment below says why.
        """
        reps = list(op.get_nodeattr("numInputVectors"))[0]

        NumChannels = op.get_nodeattr("NumChannels")
        PE = op.get_nodeattr("PE")
        ImgDim = np.prod(list(op.get_nodeattr("numInputVectors"))) // reps

        NF = NumChannels // PE
        total_iterations = ImgDim * NF

        output_delay = 0  # 4 with Vivado 2023.1

        # A note on the period, which is deliberately *not* the recorded one.
        # rtlsim reports a period `e` cycles longer than the folded word count,
        # with e = max(1, ceil(log2(numSteps + 1)) // 2) -- the depth of the
        # pipelined binary search over the threshold table. But its window is
        # *saturated*: it moves one token in every one of those `n + e` cycles, so
        # it carries n + e tokens per period where the node physically has n.
        # Matching the length therefore costs `e` on the token count and on every
        # row-0 value, and the occupancy sum reads those. The shape below --
        # period n, one token per cycle -- is exact on both, at the price of a
        # period short by at most 4 cycles out of 1024 to 401408.
        if total_iterations > output_delay:
            read = CharacteristicNode("read", [(output_delay, [1, 0])], True)

            read_write = CharacteristicNode(
                "Compute", [(total_iterations - output_delay, [1, 1])], True
            )

            write = CharacteristicNode("write", [(output_delay, [0, 1])], True)

            threshold_top = CharacteristicNode(
                "Thresholding Top", [(1, read), (1, read_write), (1, write)], False
            )

        else:
            read = CharacteristicNode("Rush-in", [(total_iterations, [1, 0])], True)
            idle = CharacteristicNode("Idle", [(output_delay - total_iterations, [0, 0])], True)

            write = CharacteristicNode("Compute", [(total_iterations, [0, 1])], True)

            threshold_top = CharacteristicNode(
                "Thresholding Top", [(1, read), (1, idle), (1, write)], False
            )

        return threshold_top  # top level phase of this node


@register("LayerNorm")
class LayerNormModel(TreeModel):
    def schedule(self, op):
        """One beat in and one out per cycle.

        The layer streams: the statistics stage runs a beat behind the data and
        the normalization follows it, so in steady state a beat is taken and a
        beat is emitted every cycle and there is nothing else to describe.
        Consecutive frames run into each other, so the two rows are one solid
        run and the period carries no wind-up.

        The schedule assumes the pipeline is scheduled at II=1, as
        ``get_exp_cycles`` does. The RTL backend is II=1 by construction. The
        HLS backend with a floating-point input is not: both of its statistics
        stages accumulate, and in floating point that accumulation is a
        loop-carried dependency the tool does not close in one cycle, so every
        beat is stretched by the same factor and the node runs several times
        longer than this says.

        TODO: that case returns None for now, which costs one rtlsim per node to
        measure the interval the tool actually reached. The shape of the
        schedule is right, only its rate is not, so a model that knew the II
        would cover it -- but the II is a synthesis outcome and nothing at this
        level can predict it. Reinstate the model here if the II ever becomes
        available, e.g. read back from the synthesis report.

        The HLS backend has one regime this shape does not cover. Its second
        statistics stage feeds a reciprocal square root whose result the
        normalization waits on, and with an integer input -- where nothing else
        holds the pipeline back -- a vector shorter than that path's latency
        cannot hide it. The stream then becomes a burst of ``N / SIMD`` beats
        per vector followed by a wait, which is not a stretch of this schedule
        but a different one, so that case returns None.

        Valid for SIMD 1..8, and on the HLS backend for an integer input with
        ``N / SIMD`` of 40 upwards (see ``LayerNormHlsModel``).
        """
        dim = int(np.prod(op.get_folded_input_shape()[:-1]))
        simd = op.get_nodeattr("SIMD")
        if dim < 1 or simd < 1:
            return None
        return passthrough_characteristic(dim, "normalize a beat")


@register("LayerNorm_hls")
class LayerNormHlsModel(LayerNormModel):
    #: the latency of the reciprocal square root the second statistics
    #: stage feeds, in cycles a vector has to be at least as long as
    RSQRT_PATH = 40

    def schedule(self, op):
        """``LayerNormModel``, declined for the two HLS regimes it does not
        describe: a floating-point input and a vector shorter than
        ``RSQRT_PATH`` beats."""
        if not op.get_input_datatype().is_integer():
            return None  # see the TODO in LayerNormModel: II > 1, and by how
            # much is a synthesis outcome
        simd = op.get_nodeattr("SIMD")
        if simd >= 1 and op.get_normal_input_shape()[-1] // simd < self.RSQRT_PATH:
            return None
        return super().schedule(op)


@register("HWSoftmax")
class HWSoftmaxModel(TreeModel):
    def input_window(self, op, ind=0):
        """A vector is normalized by its maximum and sum, none of its outputs
        leaves before all of it is in."""
        fold = op.get_folded_input_shape()[-2]
        return fold, fold


@register("HWSoftmax_hls")
class HWSoftmaxHlsModel(HWSoftmaxModel):
    def schedule(self, op):
        """One beat in and one out per cycle, with a short stall between vectors.

        The HLS design is three dataflow stages -- maximum, exponentiate and
        accumulate, divide -- with a FIFO between each. Inside a vector the beats
        stream straight through: one beat is read and one written every cycle.
        The stall is at the vector boundary, where the middle stage spends a
        cycle taking the next vector's maximum before it will touch data again;
        from SIMD 4 upwards it spends one more, the reduction tree over the SIMD
        exponentials having to settle before the vector's sum can be handed on.
        So a vector costs ``N / SIMD + stall`` cycles and a frame is one vector
        per position of the feature map.

        The period carries no pipeline fill. The stages buffer whole vectors
        between them and consecutive frames run into each other, so a frame
        boundary is just another inter-vector stall: what the schedule has to
        get right is the rate, and the recorded period is longer than a frame by
        the node's share of the fill rather than by anything periodic.

        Within a vector the stages are assumed to run at II=1, as
        ``get_exp_cycles`` does. A design that schedules slower stretches every
        beat by that factor and takes proportionally longer.

        The RTL design has no per-vector stall and is not modelled here; it falls
        back to rtlsim.

        Valid for SIMD 1..16 and N / SIMD 4..96.
        """
        n_beats = int(np.prod(op.get_folded_input_shape()[:-1]))
        n = op.get_normal_input_shape()[-1]
        simd = op.get_nodeattr("SIMD")
        beats_per_vec = max(1, n // simd)
        if n_beats < 1 or n_beats % beats_per_vec:
            return None
        stall = 2 if simd < 4 else 3
        stream = CharacteristicNode("stream a vector", [(beats_per_vec, [1, 1])], True)
        gap = CharacteristicNode("wait on the next maximum and sum", [(stall, [0, 0])], True)
        vector = CharacteristicNode("softmax one vector", [(1, stream), (1, gap)], False)
        return CharacteristicNode("softmax frame", [(n_beats // beats_per_vec, vector)], False)


@register("Lookup_hls")
class LookupHlsModel(TreeModel):
    def schedule(self, op):
        """An embedding lookup with the embedding dimension unfolded is a wire.

        With the embedding unfolded onto a single output word, one index in
        produces exactly one folded word out, and the node moves one token per
        cycle on both streams with no wind-up and no rate change. So the schedule
        is a plain pass-through over the folded input count.

        Guarded to ``internal_embedded``. ``external`` performs the lookup over
        AXI-MM, whose arrival times this shape does not describe, so that mode
        falls back to rtlsim.
        """
        if op.get_nodeattr("mem_mode") != "internal_embedded":
            return None
        n_in = int(np.prod(op.get_folded_input_shape()[:-1]))
        n_out = int(np.prod(op.get_folded_output_shape()[:-1]))
        if n_in < 1 or n_in != n_out:
            return None
        return passthrough_characteristic(n_in, "Lookup_hls")


@register("LabelSelect_hls")
class LabelSelectHlsModel(TreeModel):
    def schedule(self, op):
        """The HLS ``LabelSelect_Batch`` schedule: read the frame, then write K.

        The structure is that of ``finn-hlslib/maxpool.h``: per frame the top-K
        array is initialised, then one ``II=1`` loop reads ``Labels / PE`` folded
        words back to back, then a second loop writes the K labels, one per
        cycle. So a frame is three runs -- wind-up, a solid read run, a solid
        write run -- and the only part of it that scales with ``Labels`` is the
        length of the read run. The comparison network is over the K top entries
        and is unrolled inside the read loop, so it costs no cycles that grow
        with the label count.

        As offsets into one frame:

            reads   [3, 3 + NF)
            writes  [3 + NF + gap, ... + K)
            period  3 + NF + gap + K + tail

        ``gap`` is the flush of the unrolled shift chain over the K top entries,
        about ``K - 2`` stages after the last read; ``tail`` is the cost of
        re-entering the outer ``reps`` loop. Both are the fitted part, and are
        interpolated in K between K = 1 and K = 5. Valid for Labels 10/12/24 at
        K = 1 and Labels 1000 at K = 5, PE = 1.
        """
        PE = op.get_nodeattr("PE")
        K = op.get_nodeattr("K")
        NF = op.get_nodeattr("Labels") // PE

        if NF < 1 or K < 1:
            return None

        WIND_UP = 3
        gap = max(0, K - 2)
        tail = (K - 1) // 2
        frame = CharacteristicNode(
            "read the frame, then write the top K",
            [(WIND_UP, [0, 0]), (NF, [1, 0]), (gap, [0, 0]), (K, [0, 1]), (tail, [0, 0])],
            True,
        )
        return CharacteristicNode("LabelSelect frame", [(1, frame)], False)


@register("GlobalAccPool_hls")
class GlobalAccPoolHlsModel(TreeModel):
    def schedule(self, op):
        """Read a whole frame, pause, then emit the accumulators.

        The HLS pool is two loops that do not overlap. The first reads the
        frame's ``N`` folded words back to back at one per cycle, accumulating
        each into the bank entry for its channel tile and writing nothing. The
        second walks the bank and writes its ``NumChannels / PE`` words, again
        one per cycle. So a frame is a wind-up, the read run, a gap while the
        last accumulation settles and the write loop is entered, the write run,
        and a tail before the next frame's first read.

        The wind-up and the tail are the fitted part, and step with the width of
        the accumulator word: more than eight accumulators side by side take two
        more cycles to reach and give one back at the end. The gap does not
        move.

        ``NumChannels / PE == 1`` is a different schedule -- with one word to
        emit there is no write loop for HLS to enter and the write lands in the
        read loop's epilogue -- and is not modelled here.

        The read run assumes the accumulate loop pipelines at II=1, as
        ``get_exp_cycles`` does. A design that schedules slower stretches every
        read by that factor and takes proportionally longer.

        Valid for PE 1..64 and NumChannels / PE 2..64.
        """
        pe = op.get_nodeattr("PE")
        n_in = int(np.prod(op.get_folded_input_shape()[:-1]))
        n_out = int(np.prod(op.get_folded_output_shape()[:-1]))
        if n_in < 1 or n_out < 2 or pe < 1:
            return None
        wind_up = 3 if pe <= 8 else 5
        tail = 1 if pe <= 8 else 0
        gap = 5
        frame = CharacteristicNode(
            "accumulate then emit",
            [(wind_up, [0, 0]), (n_in, [1, 0]), (gap, [0, 0]), (n_out, [0, 1]), (tail, [0, 0])],
            True,
        )
        return CharacteristicNode("GlobalAccPool frame", [(1, frame)], False)


@register("Pool_hls")
class PoolHlsModel(TreeModel):
    def folding(self, op):
        """``(NF, SF, reps)``: ``SF = prod(KernelSize)`` folded words per channel
        tile of a pooling window, ``NF = Channels / PE`` tiles per window, and
        ``reps`` windows per frame."""
        PE = op.get_nodeattr("PE")
        Channels = op.get_nodeattr("Channels")
        KernelSize = op.get_nodeattr("KernelSize")
        OutImgDims = op.get_nodeattr("OutImgDims")
        BatchSize = op.get_nodeattr("BatchSize")

        # Derived parameters
        NF = Channels // PE  # neuron folding
        func = op.get_nodeattr("Function")
        if func == "MaxPool":
            SF = KernelSize[1] ** 2  # spatial folding per pooling window
            if KernelSize[0] == 1 or KernelSize[1] == 1:
                if KernelSize[0] == 1:
                    SF = KernelSize[1] ** 2
                else:
                    SF = KernelSize[0] ** 2
                SF = np.prod(KernelSize)
            reps = BatchSize * np.prod(OutImgDims)  # number of pooling windows to process
        else:
            SF = np.prod(KernelSize)  # spatial folding per pooling window
            reps = BatchSize * np.prod(OutImgDims)  # number of pooling windows to process
        return NF, SF, reps

    def schedule(self, op):
        """The HLS pool reads one folded word every cycle for the whole frame and
        emits one every SF, with the first write at cycle 1 rather than SF - 1:
        the writes fall at ``1, 1 + SF, 1 + 2*SF, ...``, i.e. the write for a
        window lands on the *second* cycle of it, not the last.

        A flat leaf because the nested form (``nested_schedule``) can only place
        the write at the *end* of its SF-cycle window. Degenerate foldings fall
        back to that form.

        The modulo on the write positions matters when ``SF == 1``: the last write
        of a period then lands at cycle 0 of the next one, and dropping it instead
        of wrapping it costs one token per period, which the sizer's occupancy sum
        accumulates over every frame.
        """
        NF, SF, reps = self.folding(op)
        if SF >= 1 and NF >= 1 and reps >= 1:
            n_in = int(reps) * int(NF) * int(SF)
            n_out = int(reps) * int(NF)
            if n_in > 2:
                rd = np.ones(n_in, dtype=np.int8)
                wr = np.zeros(n_in, dtype=np.int8)
                wr[(1 + int(SF) * np.arange(n_out)) % n_in] = 1
                return flat_characteristic_leaf(rd, wr, "Pool_hls schedule")
        return self.nested_schedule(op)

    def nested_schedule(self, op):
        """Read every folded word of a pooling window, write one when it closes.

        This nested form can only put a write at the end of its SF-cycle window.
        It is the fallback for degenerate foldings (``n_in <= 2``), whose
        schedules have not been measured.
        """
        NF, SF, reps = self.folding(op)

        # One input read per SF iteration
        read_pooling_input = CharacteristicNode("Read Pool Input", [(1, [1, 0])], True)

        readwrite_pooling_input = CharacteristicNode("Read Write Pool Input", [(1, [1, 1])], True)

        # SF - 1 reads + 1 read that overlaps with write
        compute_pool_window = CharacteristicNode(
            "Compute Pool Window",
            [(SF - 1, read_pooling_input), (1, readwrite_pooling_input)],  # overlap with output
            False,
        )

        # For each NF tile per pooling window
        compute_all_tiles = CharacteristicNode(
            "Compute All Tiles", [(NF, compute_pool_window)], False
        )

        # For each image region (spatial + batch)
        pool_top = CharacteristicNode("Top Pool Loop", [(reps, compute_all_tiles)], False)

        return pool_top


@register("FMPadding")
class FMPaddingModel(TreeModel):
    def input_need(self, op, ind=0):
        """Output word *j* needs the input up to its own pixel, a padding word
        the input up to the last pixel before it."""
        h, w = op.get_nodeattr("ImgDim")
        top, left = op.get_nodeattr("Padding")[:2]
        ho, wo = op.get_padded_odim()
        words = int(np.prod(op.get_folded_input_shape()[:-1])) // (h * w)
        row, col = np.divmod(np.arange(ho * wo), wo)
        inside = (row >= top) & (row < top + h)
        pixels = np.clip(row - top, 0, h) * w + np.where(inside, np.clip(col - left + 1, 0, w), 0)
        return np.repeat(np.maximum(pixels * words, 1), words)


@register("FMPadding_rtl")
class FMPaddingRtlModel(FMPaddingModel):
    def schedule(self, op):
        """One nested loop per padding dimension, exactly as the hardware writes it.

        FMPadding emits a full padded frame row by row. A row is either an
        *outer* row -- entirely padding, so it writes ``NF`` words per pixel and
        reads nothing -- or an *inner* row, which pads ``x_padding_left`` pixels,
        passes ``x_dim`` pixels through (one read and one write each), then pads
        ``x_padding_right``. The frame is ``y_padding_top`` outer rows, ``y_dim``
        inner rows, ``y_padding_bottom`` outer rows, repeated ``numInputVectors``
        times.

        The tree mirrors that nesting one level per loop, which is why this one is
        nested where MVAU's and Pool's are flat: here the reads and the writes are
        aligned to the *same* loop structure, so the nesting expresses the schedule
        rather than fighting it.

        The RTL backend pays no per-pixel loop overhead.
        """
        IMGDIM = op.get_nodeattr("ImgDim")
        PADDING = op.get_nodeattr("Padding")
        NUMCHANNELS = op.get_nodeattr("NumChannels")
        SIMD = op.get_nodeattr("SIMD")
        batch_size = op.get_nodeattr("numInputVectors")

        # compute new parameters
        NF = int(NUMCHANNELS / SIMD)
        y_padding_top, x_padding_left, y_padding_bottom, x_padding_right = PADDING
        y_dim = IMGDIM[0]
        x_dim = IMGDIM[1]

        loop_overhead = 0

        ch_pad = CharacteristicNode("Channel_Pad", [(NF, [0, 1]), (loop_overhead, [0, 0])], True)

        ch_pass = CharacteristicNode("Channel_Pass", [(NF, [1, 1]), (loop_overhead, [0, 0])], True)

        x_inner_line = CharacteristicNode(
            "Fill X full inner line",
            [(x_padding_left, ch_pad), (x_dim, ch_pass), (x_padding_right, ch_pad)],
            False,
        )

        x_outer_line = CharacteristicNode(
            "Pad X outer line", [(x_padding_left + x_dim + x_padding_right, ch_pad)], False
        )

        fmpadding = CharacteristicNode(
            "FMPadding FM",
            [
                (y_padding_top, x_outer_line),
                (y_dim, x_inner_line),
                (y_padding_bottom, x_outer_line),
            ],
            False,
        )

        fmpadding_top = CharacteristicNode(
            "FMPadding FM",
            [
                (batch_size, fmpadding),
            ],
            False,
        )

        return fmpadding_top  # top level phase of this node


@register("FMPadding_Pixel_hls")
class FMPaddingPixelHlsModel(TreeModel):
    def schedule(self, op):
        """Writes the padded raster every cycle, reads only on the real pixels.

        The layer inserts ``Stride - 1`` zero rows and columns between the
        pixels it is given, so it emits one word of the padded image per cycle
        and pulls an input word only on the cycles that carry a real pixel --
        those whose row and column are both multiples of the stride. The period
        is the padded frame, which is what ``get_exp_cycles`` returns.

        The schedule separates, so it is written as the raster loop nest it
        comes from: a pixel is ``SIMD``-folded words, a row is pixels with
        inserted columns between them, and a frame is rows with inserted rows
        between them. The last pixel of a row and the last row of a frame carry
        no trailing gap, which is why each level ends with a bare repetition.

        Both assume the HLS pipeline reaches II=1. Whether it does is a
        synthesis outcome rather than a property of the node, so a design whose
        pipeline schedules slower runs proportionally longer than this.
        """
        idim_h, idim_w = op.get_nodeattr("ImgDim")
        stride_h, stride_w = op.get_nodeattr("Stride")
        odim_h, odim_w = op.get_padded_odim()
        simd = op.get_nodeattr("SIMD")
        ch = op.get_nodeattr("NumChannels")
        batch = op.get_nodeattr("numInputVectors")
        fold = ch // simd
        if min(odim_h, odim_w, fold, batch, stride_h, stride_w) < 1:
            return None
        if idim_h < 1 or idim_w < 1:
            return None
        real_px = CharacteristicNode("real pixel", [(fold, [1, 1])], True)
        gap_px = CharacteristicNode("inserted columns", [((stride_w - 1) * fold, [0, 1])], True)
        px_then_gap = CharacteristicNode(
            "pixel + inserted columns", [(1, real_px), (1, gap_px)], False
        )
        # the last pixel of a row is not followed by inserted columns
        real_row = CharacteristicNode(
            "row carrying pixels", [(idim_w - 1, px_then_gap), (1, real_px)], False
        )
        pad_row = CharacteristicNode("inserted row", [(odim_w * fold, [0, 1])], True)
        row_then_gap = CharacteristicNode(
            "row + inserted rows", [(1, real_row), (stride_h - 1, pad_row)], False
        )
        frame = CharacteristicNode(
            "padded frame", [(idim_h - 1, row_then_gap), (1, real_row)], False
        )
        return CharacteristicNode("FMPadding_Pixel FM", [(batch, frame)], False)


class CropModel(TreeModel):
    #: cycles from reading a kept fold to writing it, per backend (see below)
    latency = None

    def schedule(self, op):
        """Every input fold is read; the ones inside the crop are written on.

        The layer streams its input raster once at one fold per cycle and passes
        through the folds that fall inside the kept window, so the two rows are
        the raster and the raster masked by the crop. The read row never pauses:
        the row and column counters that decide whether a fold is kept advance
        off the read itself, so a dropped row costs no cycle and the period is
        the input frame -- what ``get_exp_cycles`` returns.

        The latency is the delay from reading a kept fold to writing it. It
        rotates the write row against the read one and changes neither the
        period nor either row's token count, and it is placed in the frame of
        the rtlsim window the model is checked against: that window opens a
        cycle before the driver presents the first fold, and since the read row
        is solid, the write row carries the offset alone. The two backends are
        two circuits and each has its own:

        - ``Crop_hls``: 8, **fitted** to the three dataflow stages of the HLS
          pipeline. The raster assumes they pipeline at II=1, as
          ``get_exp_cycles`` does; a design that schedules slower stretches
          every read by that factor. Valid for ImgDim up to 48 x 48 and
          NumChannels / SIMD 1..16, integer and float words alike.
        - ``Crop_rtl``: 2, **derived** from ``finn-rtllib/crop/hdl/crop.sv``: one
          cycle for the fold to reach the output register, one for the driver
          offset above; the generated wrapper is wiring only. With the output
          ready the register always loads, so ``irdy`` never drops. Checked
          exact against rtlsim for H up to 16, W up to 48, NumChannels / SIMD
          1..48, INT8, UINT4 and FLOAT32, batched input, the degenerate
          ``H == 1`` and ``W == 1`` counters, a crop on every side and a crop
          keeping a single fold.

        Either holds for any crop that keeps at least one fold. The
        backend-agnostic node is no circuit and has no schedule.

        The write row is rolled rather than shifted because a crop that keeps
        the last folds of a frame wraps their writes past the end of the period,
        which is also why this is one flat leaf and not a nested row/column
        tree.
        """
        simd = op.get_nodeattr("SIMD")
        h, w = op.get_nodeattr("ImgDim")
        h = 1 if h == 0 else h
        cf = op.get_nodeattr("NumChannels") // simd
        num_vec = op.get_nodeattr("numInputVectors")
        n_vec = int(np.prod(num_vec)) if num_vec != [0] else 1
        if min(h, w, cf, n_vec) < 1:
            return None
        keep = np.zeros((n_vec, h, w, cf), dtype=np.int8)
        north = op.get_nodeattr("CropNorth")
        south = op.get_nodeattr("CropSouth")
        west = op.get_nodeattr("CropWest")
        east = op.get_nodeattr("CropEast")
        keep[:, north : h - south, west : w - east, :] = 1
        if keep.sum() < 1:
            return None
        wr = np.roll(keep.reshape(-1), self.latency)
        rd = np.ones_like(wr)
        return flat_characteristic_leaf(rd, wr, "Crop raster")


@register("Crop_hls")
class CropHlsModel(CropModel):
    latency = 8


@register("Crop_rtl")
class CropRtlModel(CropModel):
    latency = 2


@register("SelectToken_rtl")
class SelectTokenRtlModel(TreeModel):
    def schedule(self, op):
        """Read all ``NumTokens`` tokens, write the selected one.

        The crop core on a ``1 x NumTokens`` feature map of
        ``NumChannels / SIMD`` folds, cropped west and east down to the single
        column ``TokenIndex``, which is the geometry
        ``_get_template_param_dict`` hands it. So every fold is read at one per
        cycle and the selected token's folds are written 2 cycles later, the
        same circuit and so the same latency as ``Crop_rtl`` -- see
        ``CropModel`` for where the 2 comes from. The period is the
        token sequence, what ``get_exp_cycles`` returns.

        Built from that geometry rather than inherited from ``Crop``, whose
        model reads Crop attributes this node never sets and raised on the
        missing ``ImgDim``. One frame per input, the input being a single
        ``(1, NumTokens, NumChannels)`` sequence. Checked exact against rtlsim
        for NumTokens 1..16, NumChannels / SIMD 1..8, first, middle and last
        token, INT8 and UINT4; nothing is fitted.
        """
        params = op._get_template_param_dict()
        num_tokens, cf = params["W"], params["CF"]
        keep = np.zeros((num_tokens, cf), dtype=np.int8)
        keep[params["CROP_W"] : num_tokens - params["CROP_E"], :] = 1
        wr = np.roll(keep.reshape(-1), 2)
        rd = np.ones_like(wr)
        return flat_characteristic_leaf(rd, wr, "SelectToken raster")


@register("Pad1D_rtl")
class Pad1DRtlModel(TreeModel):
    def schedule(self, op):
        """Emits the left pad, then the input sequence, then the right pad.

        The layer writes one folded word every cycle for the whole output
        sequence and never stalls: the pad words come from constants held in the
        design, so the input stream is read only while the sequence itself is
        being passed through. That splits one period into three back-to-back
        phases -- ``PadLeft`` tokens written with nothing read, ``NumTokens``
        tokens read and written, ``PadRight`` tokens written with nothing read --
        each token being ``NumChannels / SIMD`` folded words. The period is the
        output frame, which is what ``get_exp_cycles`` returns.

        The single-cycle output register between reading a word and writing it is
        not modelled; the wind-up is shorter than one cycle of the period rtlsim
        measures and so contributes nothing to it.

        Valid for the RTL implementation at any SIMD dividing NumChannels and any
        pad widths. Assumes the module accepts a word every cycle, as
        ``get_exp_cycles`` does.
        """
        simd = op.get_nodeattr("SIMD")
        num_channels = op.get_nodeattr("NumChannels")
        num_tokens = op.get_nodeattr("NumTokens")
        pad_left = op.get_nodeattr("PadLeft")
        pad_right = op.get_nodeattr("PadRight")
        if simd < 1 or num_channels % simd != 0:
            return None
        folds = num_channels // simd
        if folds < 1 or num_tokens < 1 or min(pad_left, pad_right) < 0:
            return None
        phases = []
        if pad_left > 0:
            left = CharacteristicNode("emit the left pad", [(pad_left * folds, [0, 1])], True)
            phases.append((1, left))
        thru = CharacteristicNode("pass the sequence on", [(num_tokens * folds, [1, 1])], True)
        phases.append((1, thru))
        if pad_right > 0:
            right = CharacteristicNode("emit the right pad", [(pad_right * folds, [0, 1])], True)
            phases.append((1, right))
        return CharacteristicNode("pad a sequence", phases, False)


@register("StreamingDataWidthConverter")
class DataWidthConverterModel(TreeModel):
    def schedule(self, op):
        """Three schedules, chosen by which side of the converter is wider.

        A DWC has no compute and no storage beyond one word, so its schedule is
        fixed by the width ratio alone:

        * **down-conversion** (``inWidth > outWidth``): one read every
          ``writes_per_read`` cycles, a write every cycle.
        * **up-conversion**: a read every cycle, one write every
          ``reads_per_write`` cycles, on the cycle the last input word arrives.
        * **pass-through**: read and write on every cycle.

        Only integer ratios are modelled; a non-dividing pair returns ``None``
        and the node falls back to rtlsim characterization.

        Down-conversion's first read is at cycle 1, not 0, and its period is
        exactly ``numReps * writes_per_read``. The repeated block therefore opens
        on a write-only cycle and takes its read on the second, which is what puts
        the reads where the hardware has them.
        """
        inWidth = op.get_nodeattr("inWidth")
        outWidth = op.get_nodeattr("outWidth")

        wind_up = 0

        idle = CharacteristicNode("idle", [(1, [0, 0])], True)

        if inWidth > outWidth:
            numReps = int(np.prod(op.get_folded_input_shape()[:-1]))
            # down-conversion
            if inWidth % outWidth != 0:
                return None  # no support for gcd partial conversion yet

            writes_per_read = inWidth // outWidth

            # The packer writes one word every cycle and takes one word in every
            # ``writes_per_read``, with its first read at cycle 1, not 0. Opening
            # the repeated block on a write-only cycle puts the read there. With
            # ``writes_per_read > 1`` the last read of a period is at
            # ``period - writes_per_read + 1`` and never wraps, so the whole
            # schedule nests.
            if writes_per_read > 1 and numReps >= 1:
                down_convert_word = CharacteristicNode(
                    "write, then read the next word while writing",
                    [(1, [0, 1]), (1, [1, 1]), (writes_per_read - 2, [0, 1])],
                    True,
                )
                return CharacteristicNode(
                    "compute a set of DWCs with down conversion",
                    [(numReps, down_convert_word)],
                    False,
                )

            # read 1, write many, repeats for in-word count

            read_input = CharacteristicNode("read 1 word", [(1, [1, 1])], True)

            write_output = CharacteristicNode("write words", [(writes_per_read - 1, [0, 1])], True)

            down_convert_word = CharacteristicNode(
                "down convert all words in a single transaction",
                [(1, read_input), (1, write_output)],
                False,
            )

            dwc_top = CharacteristicNode(
                "compute a set of DWCs with down conversion",
                [(wind_up, idle), (numReps, down_convert_word)],
                False,
            )

        elif inWidth < outWidth:
            numReps = op.get_number_output_values()
            # up-conversion

            if outWidth % inWidth != 0:
                return None  # no support for gcd partial conversion yet

            reads_per_write = outWidth // inWidth
            # read 1, write many, repeats for in-word count

            read_input = CharacteristicNode(
                "read first N-1 words", [(reads_per_write - 1, [1, 0])], True
            )

            write_output = CharacteristicNode(
                "read Nth word and write output word", [(1, [1, 1])], True
            )

            up_convert_word = CharacteristicNode(
                "down convert all words in a single transaction",
                [(1, read_input), (1, write_output)],
                False,
            )

            dwc_top = CharacteristicNode(
                "compute a set of DWCs with up conversion",
                [(wind_up, idle), (numReps, up_convert_word)],
                False,
            )

        else:
            # pass-through
            numReps = int(np.prod(op.get_folded_input_shape()[:-1]))

            pass_through = CharacteristicNode("pass-through", [(1, [1, 1])], True)

            dwc_top = CharacteristicNode(
                "DWC pass-through, no conversion", [(wind_up, idle), (numReps, pass_through)], False
            )

        return dwc_top


@register("StreamingSplit_hls")
class SplitHlsModel(TreeModel):
    def schedule(self, op):
        """The schedule of the freerunning HLS splitter: one word every five cycles.

        Neither a passthrough nor II=1. ``csynth.rpt`` for the generated top level
        reports Latency 4, **Interval 5**, ``Pipelined: no``,
        ``hls_style: freerunning``; the inner ``StreamingSplit`` in
        finn-hlslib/split.hpp is ``#pragma HLS pipeline II=1``, so it is the
        freerunning wrapper around it that admits one word every five cycles.

        Round-robin over the destinations in ``ChannelsPerStream`` order, so
        destination *i* receives one contiguous burst of ``C[i]`` words per input
        vector rather than a share of every word:

            input word k        at cycle 1 + 5k
            output i, word j    at cycle 7 + 5*(offset_i + j)

        with ``offset_i = sum(C[:i])``. Row 0 -- the only row a tree model emits,
        and the only one the sizer reads -- is ``out0``, so the schedule here is
        the first destination's.

        Validated against node rtlsim for equal and unequal ``ChannelsPerStream``,
        SIMD 1/2/4 and several ``numInputVectors``.
        """
        simd = op.get_nodeattr("SIMD")
        chans = list(op.get_nodeattr("ChannelsPerStream"))
        n_vec = int(np.prod(list(op.get_nodeattr("numInputVectors"))))
        if simd < 1 or not chans or any(c % simd for c in chans) or n_vec < 1:
            return None
        folds = [c // simd for c in chans]
        total = int(sum(folds))
        if total < 1:
            return None
        ii = 5
        span = ii * total  # cycles per input vector
        period = n_vec * span
        rd = np.zeros(period, dtype=np.int8)
        wr = np.zeros(period, dtype=np.int8)
        rd[(1 + ii * np.arange(n_vec * total)) % period] = 1
        # row 0 is the schedule the FIFO sizer reads, and that is out0
        starts = span * np.arange(n_vec)
        first = (7 + ii * np.arange(folds[0]))[None, :] + starts[:, None]
        wr[first.ravel() % period] = 1
        return flat_characteristic_leaf(rd, wr, "StreamingSplit_hls schedule")


@register("StreamingConcat_hls")
class ConcatHlsModel(TreeModel):
    def schedule(self, op):
        """The schedule of the freerunning HLS concatenator.

        The mirror image of ``StreamingSplit``: same ``hls_style: freerunning``
        wrapper, same Interval 5, and the inputs are consumed in **staggered
        contiguous bursts** in ``ChannelsPerStream`` order, not concurrently.
        That matters for sizing: an imbalance between the inputs accumulates as a
        prefix sum along the stream order, not as a max over the streams.

            input i, word j     at cycle 6 + 5*(offset_i + j)
            output word k       at cycle 2 + 5k

        with ``offset_i = sum(C[:i])``.

        Row 0 -- the only row a tree model can emit, and the only one the sizer
        reads -- is ``in0``, so the schedule here is the *first* input's. The
        later inputs are read up to ``5 * offset_i`` cycles after it; that offset
        is a consequence of the single-row representation in
        ``derive_token_access_vectors_using_tree_model``, not of this model.

        Validated against node rtlsim for the same configurations as the splitter.
        """
        simd = op.get_nodeattr("SIMD")
        chans = list(op.get_nodeattr("ChannelsPerStream"))
        n_vec = int(np.prod(list(op.get_nodeattr("numInputVectors"))))
        if simd < 1 or not chans or any(c % simd for c in chans) or n_vec < 1:
            return None
        folds = [c // simd for c in chans]
        total = int(sum(folds))
        if total < 1:
            return None
        ii = 5
        span = ii * total
        period = n_vec * span
        rd = np.zeros(period, dtype=np.int8)
        wr = np.zeros(period, dtype=np.int8)
        wr[(2 + ii * np.arange(n_vec * total)) % period] = 1
        starts = span * np.arange(n_vec)
        first = (6 + ii * np.arange(folds[0]))[None, :] + starts[:, None]
        rd[first.ravel() % period] = 1
        return flat_characteristic_leaf(rd, wr, "StreamingConcat_hls schedule")


@register("DuplicateStreams_hls")
class DuplicateStreamsHlsModel(TreeModel):
    def schedule(self, op):
        """A fork is transparent: every output beat leaves on the cycle it arrives.

        ``DuplicateStreams`` has no storage and no rate change. Its HLS loop reads
        one folded word and writes the same word to each output in the same
        iteration, pipelined at II=1, so the schedule is one solid run of
        read-and-write cycles as long as the folded frame -- no wind-up, no gap,
        and nothing that depends on the number of outputs.

        The single run is also why this node is in ``_UNCHARACTERIZED_OPS`` in the
        chained-TAV pass: with writes on the same cycle as the read, the fork's
        own output edges can never accumulate occupancy, so sizing them from this
        schedule alone would say every branch needs depth 1. What a branch really
        needs is decided at the *join*, from the imbalance between the two paths.
        """

        dim = np.prod(op.get_folded_output_shape()[1:-1])

        read_write = CharacteristicNode("passing duplicate layer", [(dim, [1, 1])], True)
        duplicatestreams_top = CharacteristicNode("compute duplicate", [(1, read_write)], False)

        return duplicatestreams_top  # top level phase of this node
