# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Tree models of the matrix-vector (MVAU) and vector-vector (VVAU) units."""

import math
import numpy as np

from finn.transformation.fpgadataflow.fifo_sizing.tav import (
    CharacteristicNode,
    flat_characteristic_leaf,
)
from finn.transformation.fpgadataflow.fifo_sizing.tree_models.base import (
    TreeModel,
    register,
)


@register("MVAU")
class MVAUModel(TreeModel):
    """Reads in per-feature-map bursts, writes in one evenly spaced train.

    Both implementations compute ``numVectors`` feature maps of ``NF = MH/PE``
    output words, each needing ``SF = MW/SIMD`` input words, and both retire
    one tile per cycle -- so a feature map occupies ``NF * SF`` cycles. What
    differs between them, and what the subclasses encode, is the phase:

    * the reads of a feature map arrive as one back-to-back burst of ``SF`` at
      the start of that feature map's window;
    * the writes are *not* aligned to those windows. They form a single train
      of ``numVectors * NF`` words spaced ``SF`` apart across the whole period,
      and the last of them lands at exactly ``period``, i.e. cycle 0 of the
      next one. A nested tree cannot rotate a transaction across the period
      boundary, so all branches build flat run-length-encoded leaves.

    Three constructions, by implementation and folding:

    * ``TH > 1`` -- ``finn-rtllib/mvu_tiled``, a different RTL module.
    * ``MVAU_rtl`` -- the untiled RTL pipeline.
    * ``MVAU_hls`` -- the HLS wind-up.

    Each is used only inside the regime it was validated in. Outside it the
    two untiled ones fall back to ``legacy_tree``: the same nested shape with
    no wind-up, whose period is short by three to five cycles but whose token
    counts are right. For the tiled one the node falls back to rtlsim.

    Two configurations are declined outright rather than approximated:

    * ``mem_mode="external_mem"``. It fetches the weights over AXI-MM, so they
      cannot be assumed to keep pace with the compute, and the schedules
      all assume they do. ``internal_embedded``, ``internal_decoupled`` and
      ``external`` present the weights as an always-ready stream, which is
      the regime the constants here hold in. So does ``dynamic`` as long as
      its ``dynamic_load`` refills a bank faster than the compute replays the
      other one (``numInputVectors[-1] >= SIMD``); its own streamed input is
      added by ``derive_token_access_vectors``.
    * a folding whose period cannot hold the schedule at one transaction per
      cycle, or a single output for the whole node. See the guards in the
      subclasses.
    """

    def declines(self, op):
        """True for the memory modes the schedules do not describe (see above)."""
        mem_mode = op.get_nodeattr("mem_mode")
        n_reps = op.get_nodeattr("numInputVectors")[-1]
        if mem_mode == "external_mem":
            return True
        return mem_mode == "dynamic" and n_reps < op.get_nodeattr("SIMD")

    def folding(self, op):
        """``(SF, NF, numVectors, SIMD, TH)``."""
        MW = op.get_nodeattr("MW")
        MH = op.get_nodeattr("MH")
        SIMD = op.get_nodeattr("SIMD")
        PE = op.get_nodeattr("PE")
        numVectors = np.prod(op.get_nodeattr("numInputVectors"))
        SF = int(MW / SIMD)
        NF = int(MH / PE)
        TH = op.get_nodeattr("TH")
        return SF, NF, numVectors, SIMD, TH

    def legacy_tree(self, SF, NF, numVectors):
        """The nested shape with no wind-up: period short by three to five
        cycles, token counts right."""
        idle = CharacteristicNode("idle cycles", [(1, [0, 0])], True)
        read = CharacteristicNode("Read a burst of input", [(1, [1, 0])], True)
        write = CharacteristicNode("update output", [(1, [0, 1])], True)
        read_and_write = CharacteristicNode("update output", [(1, [1, 1])], True)
        write_PE = CharacteristicNode(
            "iterate MW/SIMD and update an output", [(SF - 1, idle), (1, write)], False
        )
        feature_map = CharacteristicNode(
            "Compute single feature map",
            [(SF - 1, read), (1, read_and_write), (NF - 1, write_PE)],
            False,
        )
        return CharacteristicNode(
            "compute set of feature maps", [(1, idle), (numVectors, feature_map)], False
        )

    def input_blocks(self, op):
        """A dynamic ``MVAU`` computes on the matrix its ``dynamic_load``
        buffered from in1: the ``numInputVectors[-1] * MH / PE`` outputs of a
        matrix need all ``MW * MH / PE`` words of it."""
        if op.get_nodeattr("mem_mode") == "dynamic":
            n_reps = op.get_nodeattr("numInputVectors")[-1]
            nf = op.get_nodeattr("MH") // op.get_nodeattr("PE")
            burst = op.get_nodeattr("MW") * op.get_nodeattr("MH") // op.get_nodeattr("PE")
            return {1: (burst, n_reps * nf)}
        return {}

    def input_window(self, op, ind=0):
        """The ``MH / PE`` outputs of a vector need all ``MW / SIMD`` words of
        it; the tiled MVU (TH > 1) computes a group of TH vectors together."""
        if ind == 0:
            th = op.get_nodeattr("TH")
            sf = op.get_nodeattr("MW") // op.get_nodeattr("SIMD")
            return sf * th, op.get_nodeattr("MH") // op.get_nodeattr("PE") * th
        return super().input_window(op, ind)


@register("MVAU_hls")
class MVAUHlsModel(MVAUModel):
    def schedule(self, op):
        if self.declines(op):
            return None
        SF, NF, numVectors, SIMD, TH = self.folding(op)

        # The MVAU_hls wind-up. Three facts fix the phases below:
        #
        #   * the first output transaction is at cycle SF + 4;
        #   * outputs are then spaced SF cycles apart;
        #   * the period is numVectors*NF*SF + 9 - LEAD_IN, where LEAD_IN is the
        #     cycle of the first input transaction.
        #
        # LEAD_IN is 3 when the node has an activation (threshold) stage *and*
        # NF >= 8, and 5 otherwise: the threshold stage costs two cycles of
        # wind-up, but only once there are enough output folds for the pipeline
        # to be filled when the first one retires. Neither NF alone nor the
        # datatype moves it.
        #
        # Valid for SF 8..512, NF 1..512 and numInputVectors 1..900.
        LEAD_IN = 3 if (op.get_nodeattr("noActivation") == 0 and NF >= 8) else 5
        FIRST_OUT = SF + 4
        n_vec = int(numVectors)
        period = n_vec * NF * SF + 9 - LEAD_IN

        if SF < 1 or NF < 1 or n_vec < 1:
            return self.legacy_tree(SF, NF, numVectors)
        if n_vec * SF > period or n_vec * NF > period:
            # More tokens than cycles: the schedule cannot be laid out at one
            # transaction per cycle at all, so the wind-up arithmetic above does
            # not apply. Keep the shape that has no wind-up.
            return self.legacy_tree(SF, NF, numVectors)
        if NF < 2 and n_vec < 2:
            # A single output for the whole node. With one output there is
            # nowhere in the steady-state window to place it and the wind-up
            # collapses -- LEAD_IN 0, first output at cycle 1, period SF + 1 --
            # which is exactly what ``legacy_tree`` produces. NF == 1 with
            # *several* feature maps is a different case and does obey the rule
            # above: it is the ordinary burst schedule with a one-output feature
            # map.
            return self.legacy_tree(SF, NF, numVectors)

        # Reads arrive as one burst of SF per feature map, at the start of that
        # feature map's NF*SF window; writes are a single train of numVectors*NF
        # spaced SF apart, which is *not* aligned to the feature-map boundaries.
        # One flat run-length encoded leaf expresses both.
        rd = np.zeros(period, dtype=np.int8)
        wr = np.zeros(period, dtype=np.int8)
        for i in range(n_vec):
            base = LEAD_IN + i * NF * SF
            rd[(base + np.arange(SF)) % period] = 1
        # Both rows wrap rather than slice. The wind-up can push the last read
        # burst past the end of the period, and the last output lands at exactly
        # ``period``, i.e. cycle 0 of the next one. Dropping either loses a token
        # per period, and the sizer's steady-state occupancy accumulates that
        # deficit over every frame -- the largest error this schedule can carry.
        out_cycles = (FIRST_OUT + SF * np.arange(n_vec * NF)) % period
        wr[out_cycles] = 1

        return flat_characteristic_leaf(rd, wr, "MVAU burst schedule")


@register("MVAU_rtl")
class MVAURtlModel(MVAUModel):
    def schedule(self, op):
        if self.declines(op):
            return None
        SF, NF, numVectors, SIMD, TH = self.folding(op)

        if TH > 1:
            return self.tiled_schedule(SF, NF, numVectors, SIMD, TH)

        # MVAU_rtl is a different pipeline from MVAU_hls: a different period
        # excess, and a first input transaction that wraps around the period
        # rather than sitting a few cycles into it. The HLS wind-up
        # (``MVAUHlsModel``) does not describe it, so it gets its own construction.
        #
        # The RTL MVAU retires one SIMD x PE tile per cycle, so a feature map
        # occupies ``SF * NF`` cycles. It reads that feature map's ``SF`` input
        # words as one back-to-back burst and emits its ``NF`` output words
        # spaced ``SF`` apart. The read burst begins at ``3 - SF`` relative to
        # the frame boundary: the node prefetches the next tile's inputs while
        # still emitting the current one. The stored vectors also carry the
        # wind-up, as ``e`` extra cycles in the period and a rotation of
        # ``2 * e`` (the window starts two periods in)::
        #
        #     L  = ceil(log2(SIMD))                adder-tree depth
        #     e  = (10 + L) // 5                   period = n_vec*SF*NF + e
        #     read burst k  at (3 - SF) + k*SF*NF  before the window rotation
        #     write j       at (8 + L)  + j*SF
        #
        # Valid for SF 1..512, NF 1..1000, numInputVectors 1..12321, SIMD
        # 3..96. ``8 + L`` is unconstrained for ``SF == 1``, where the writes
        # are contiguous and carry no phase.
        n_vec = int(numVectors)
        if SF < 1 or NF < 1 or n_vec < 1 or SIMD < 1:
            return self.legacy_tree(SF, NF, numVectors)
        fm = SF * NF
        L = int(math.ceil(math.log2(SIMD))) if SIMD > 1 else 0
        e = (10 + L) // 5
        period = n_vec * fm + e
        if period < 4 or n_vec * NF > period:
            return self.legacy_tree(SF, NF, numVectors)
        rd = np.zeros(period, dtype=np.int8)
        wr = np.zeros(period, dtype=np.int8)
        shift = 2 * e  # the recorded window starts 2*period into the trace
        starts = ((3 - SF - shift) + fm * np.arange(n_vec)) % period
        idx = (starts[:, None] + np.arange(SF)[None, :]) % period
        rd[idx.ravel()] = 1
        wr[((8 + L - shift) + SF * np.arange(n_vec * NF)) % period] = 1
        # Every burst is placed modulo the period so the token counts stay
        # exact at n_vec*SF reads and n_vec*NF writes. The sizer's occupancy
        # sum integrates those counts, so a token lost off either end matters
        # more than the phase.
        return flat_characteristic_leaf(rd, wr, "MVAU_rtl burst schedule")

    def tiled_schedule(self, SF, NF, numVectors, SIMD, TH):
        # Tiling is MVAU_rtl-only -- the HLS backend supports TH = 1 only --
        # and does not fold the ordinary RTL MVAU, it selects a *different
        # module*: ``mvu_tiled_axi`` instead of ``mvu_vvu_axi``. None of the
        # untiled constants in ``schedule`` apply to it.
        #
        # The activation ``input_gen`` runs the loop nest ``DIMS = {NF, SF,
        # TH}``, ``COEFS = {0, 1, SF}`` over an ``FM_SIZE`` of ``SF * TH``: one
        # *group* of ``TH`` input vectors is read as a single contiguous burst
        # of ``SF * TH`` words and then replayed ``NF`` times, once per output
        # fold, with the ``TH`` vectors interleaved innermost. Weights are
        # fetched once per group rather than once per vector -- the point of
        # tiling -- and sustain one beat per compute cycle, so they never pace
        # the node. ``TH`` therefore does not multiply the cycle count: a group
        # costs ``NF*SF*TH`` cycles plus a stall and covers ``TH`` vectors.
        #
        # Per group: one contiguous read burst of ``SF*TH``; then ``NF-1``
        # single outputs spaced ``SF*TH`` apart -- one per output fold, being
        # the ``th = 0`` word of that fold -- and finally a run of
        # ``TH*NF - (NF-1)`` outputs, every remaining word of the group. That
        # last run is the output reorder buffer draining: ``inst_reorder_out``
        # transposes the accumulator's ``(nf, th)`` production order into the
        # ``(th, nf)`` order the stream needs, so the word for
        # ``(th=0, nf=NF-1)`` is not available until the group is essentially
        # finished, and once it is, everything queued behind it leaves back to
        # back. Token counts are exact by construction: ``G`` groups of
        # ``SF*TH`` reads is ``n_vec*SF``, and ``G`` groups of ``TH*NF`` writes
        # is ``n_vec*NF``.
        #
        # The phase is not modelled: the recorded rows carry a
        # wind-up-dependent period excess and rotation that the steady state
        # does not determine, so the schedule is emitted at its natural phase,
        # with the group's read burst at cycle 0. That costs a single rotation
        # of both rows together -- a whole-node latency offset, not a
        # distortion; ``compare_chr_funcs_up_to_rotation`` checks the relative
        # phase.
        #
        # Valid for SF 2..64, NF 1..6, TH 2/3/6/9, SIMD 1..16 and
        # numInputVectors 9 and 18.
        n_vec = int(numVectors)
        if SF < 1 or NF < 1 or n_vec < 1 or SIMD < 1:
            return None
        if n_vec % TH:
            # the tiled MVU requires TH | numInputVectors; the RTL backend
            # raises on this, so a graph that reaches here is malformed
            return None
        fm = SF * TH
        # The stall at the group boundary is the larger of two buffer effects
        # (they overlap):
        # * activation replay: input_gen holds the group's SF*TH input words
        #   in a 2^ceil(log2(SF*TH+2))-word circular buffer and replays them
        #   NF times with the TH vectors interleaved innermost, so the TH-th
        #   beat already needs word (TH-1)*SF. The words are only freed after
        #   the last replay, so only the buffer's spare slots prefetch the next
        #   group; what its interleaved start needs beyond that (plus 3 cycles)
        #   arrives one word per cycle. The power-of-two buffer is why the
        #   stall is not smooth in SF.
        # * output reorder: input_gen holds all NF*TH words of a group in a
        #   2^ceil(log2((NF-1)*TH+3))-word buffer until the last fold arrives;
        #   then the remaining run drains back to back while the next group
        #   already produces outputs. If those do not fit the spare slots
        #   (plus TH-4 words of slack), the pipeline freezes until the drain is
        #   done.
        # Exact for power-of-two TH (and the replay part for any TH), within
        # ~10 % of a group otherwise; verified against rtlsim for SF 1..64,
        # NF 1..48, TH 2..16 and bert's foldings.
        buf_in = 1 << math.ceil(math.log2(fm + 2))
        in_stall = max(0, (TH - 1) * (SF - 1) + 3 - (buf_in - fm))
        spare = (1 << math.ceil(math.log2((NF - 1) * TH + 3))) - NF * TH
        run = TH * NF - (NF - 1)
        out_stall = 0
        if NF > 1:
            # cycle (from the drain start) at which the first non-fitting word
            # of the next group arrives: its outputs come in bursts of TH every
            # fm cycles
            w = max(0, spare + TH - 4)
            t_block = fm * (w // TH + 1) + w % TH
            out_stall = max(0, run - t_block, -spare)
        if out_stall > max(in_stall, -spare):
            # the output reorder buffer stalls the group (NF*TH just under a
            # power of two with few compute cycles per fold): its run then
            # drains while the next group's first folds queue up behind it, a
            # shape the schedule below does not have. Leave the node to
            # rtlsim. (A buffer a few words too small for the group, spare < 0,
            # only trickles.)
            return None
        stall = max(in_stall, out_stall)
        span = NF * fm + stall
        # the first output of a group, as an offset from the group's first
        # read; tree_depth = clog2(ceil(SIMD/3)) is the add_multi depth inside
        # acc_stage, which delays every output by that many cycles, and
        # 10 - TH is the one fitted constant
        chainlen = (SIMD + 2) // 3
        tree_depth = int(math.ceil(math.log2(chainlen))) if chainlen > 1 else 0
        d = stall + fm + (10 - TH) + tree_depth
        G = n_vec // TH
        period = G * span
        if period < 1:
            return None
        rd = np.zeros(period, dtype=np.int8)
        wr = np.zeros(period, dtype=np.int8)
        for g in range(G):
            base = g * span
            rd[(base + np.arange(fm)) % period] = 1
            if NF > 1:
                wr[(base + d + fm * np.arange(NF - 1)) % period] = 1
            wr[(base + d + (NF - 1) * fm + np.arange(run)) % period] = 1
        return flat_characteristic_leaf(rd, wr, "MVAU_rtl tiled burst schedule")


class VVAUModel(TreeModel):
    """Shared by the VVAU backends."""

    def folding(self, op):
        """``(SF, NF, numReps)``: ``SF = prod(Kernel)/SIMD`` folded words per
        output, ``NF = Channels/PE`` outputs per pixel, ``numReps = prod(Dim)``
        pixels."""
        SIMD = op.get_nodeattr("SIMD")
        PE = op.get_nodeattr("PE")
        Channels = op.get_nodeattr("Channels")
        Kernel_2 = np.prod(op.get_nodeattr("Kernel"))
        NF = int(Channels / PE)
        numReps = np.prod(op.get_nodeattr("Dim"))
        SF = Kernel_2 // SIMD
        return SF, NF, numReps

    def nested_schedule(self, op):
        """One read per cycle, one write every SF.

        The depthwise VVAU accumulates ``SF`` folded words per output and
        produces ``NF`` outputs per pixel, over ``numReps`` pixels: a solid read
        run of ``numReps * NF * SF`` cycles with a write on every ``SF``-th --
        the same accumulate-and-flush shape as ``Pool``, and unlike ``MVAU``
        there is no per-feature-map read burst to place.

        This nested tree has no wind-up. It is the RTL variant's schedule, which
        has not been measured, and the HLS variant's fallback for small foldings.
        """
        SF, NF, numReps = self.folding(op)

        write_out = CharacteristicNode("write out simd (1 for hls)", [(1, [1, 1])], True)

        compute_one_sf = CharacteristicNode("read one SF input", [(1, [1, 0])], True)

        compute_sf = CharacteristicNode(
            "process SF-1 inputs", [(SF - 1, compute_one_sf), (1, write_out)], False
        )

        compute_transaction = CharacteristicNode(
            "Compute VVAU one transaction",
            [
                (NF, compute_sf),
            ],
            False,
        )

        vvau_top = CharacteristicNode(
            "Compute VVAU input set", [(numReps, compute_transaction)], False
        )

        return vvau_top  # top level phase of this node


@register("VVAU_rtl")
class VVAURtlModel(VVAUModel):
    def schedule(self, op):
        return self.nested_schedule(op)


@register("VVAU_hls")
class VVAUHlsModel(VVAUModel):
    def schedule(self, op):
        """The HLS VVAU reads one folded word per cycle, back to back, and
        emits one every SF, with a fixed five-cycle wind-up in front of both:
        reads start at cycle 5 and the first write is at SF + 4.

        The write lands at the END of each SF block -- read k at 5 + k, write j
        at 4 + SF*(j+1), i.e. offset SF-1 within the j-th block -- so a
        five-cycle prefix node in front of ``NF * numReps`` copies of an
        SF-cycle accumulate block is the whole schedule, cycle for cycle. Small
        foldings fall back to ``nested_schedule``.
        """
        SF, NF, numReps = self.folding(op)
        n_in = int(numReps) * NF * SF
        n_out = int(numReps) * NF
        if SF >= 1 and n_out >= 1 and n_in > 8:
            # The period is n_in + 5. rtlsim reports n_in + 4 because its
            # period is cycles_rtlsim // periods_to_simulate and rounds down;
            # n_in + 5 is the length that puts the wind-up and every read and
            # write on the cycle the hardware has them, with no wrap-around.
            #
            # Validated on VVAU_hls nodes with PE >= 4. Nodes with PE <= 2
            # record a saturated window instead -- period n_in + 1, no gap --
            # and keep an error of about 8 cycles; no attribute other than PE
            # separates them, and PE alone is too little to fit a rule on.
            wind_up = CharacteristicNode("wind-up", [(5, [0, 0])], True)
            accumulate = CharacteristicNode(
                "accumulate SF inputs, flush on the last",
                [(SF - 1, [1, 0]), (1, [1, 1])],
                True,
            )
            return CharacteristicNode("VVAU_hls frame", [(1, wind_up), (n_out, accumulate)], False)
        return self.nested_schedule(op)
