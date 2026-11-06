# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Tree models of the transposing operators, InnerShuffle and OuterShuffle."""

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
from finn.util.basic import get_vivado_version


@register("InnerShuffle")
class InnerShuffleModel(TreeModel):
    def extra_latency(self, op):
        """An ``InnerShuffle`` transposes a page in a bank: none of its words
        leaves before all ``I * J / SIMD`` of them are in, at most one per
        cycle."""
        in_shape = op.get_nodeattr("transpose_in_shape")
        return int(in_shape[-2]) * int(in_shape[-1]) // op.get_nodeattr("SIMD")

    def input_window(self, op, ind=0):
        """A page leaves only once all of it is in."""
        page = self.extra_latency(op)
        return page, page


@register("InnerShuffle_rtl")
class InnerShuffleRtlModel(InnerShuffleModel):
    def schedule(self, op):
        """A page in at one word per cycle, a word out every two.

        The banks take one access per cycle, so a word costs two: one to store
        it and one to read it back transposed. That is what paces the node -- a
        frame of ``N`` words takes ``2 * N`` cycles in steady state, which is
        ``get_exp_cycles`` without the one-time fill of the first page.

        Reads come a page at a time: the ``I * J / SIMD`` words of a page are
        taken back to back while the other bank is drained, so a page turn is a
        burst of ``page`` reads every ``2 * page`` cycles. The writes are that
        drain, modelled at the rate the bank sustains: one word every other
        cycle. Where inside its pair of cycles a write actually lands is not
        modelled -- it is worth at most one token of cumulative difference,
        which is inside the budget the tree-model test allows.

        The phase between the two rows is not modelled -- the recorded window
        opens wherever the first page fill left it -- so both start at cycle 0.
        """
        simd = op.get_nodeattr("SIMD")
        in_shape = op.get_nodeattr("in_shape")
        if len(in_shape) < 2 or simd < 1:
            return None
        total_elems = int(np.prod(in_shape)) // simd
        page = (int(in_shape[-2]) * int(in_shape[-1])) // simd
        if total_elems < 1 or page < 1 or total_elems % page:
            # a page turn has to divide the frame for the steady-state pattern
            # to repeat
            return None
        period = 2 * total_elems
        rd = np.zeros(period, dtype=np.int8)
        wr = np.zeros(period, dtype=np.int8)
        starts = np.arange(total_elems // page) * 2 * page
        rd[(starts[:, None] + np.arange(page)[None, :]).ravel()] = 1
        wr[::2] = 1
        return flat_characteristic_leaf(rd, wr, "InnerShuffle page schedule")


@register("OuterShuffle")
class OuterShuffleModel(TreeModel):
    def input_need(self, op, ind=0):
        """Output word *j* is input word ``src[j]`` of the transposed frame, so
        it needs the first ``max(src[:j + 1]) + 1`` words (only when the
        innermost dimension, folded by SIMD, stays innermost)."""
        perm = list(op.get_nodeattr("perm"))
        shape = list(op.get_nodeattr("transpose_in_shape"))
        if perm[-1] != len(perm) - 1:
            return None
        shape[-1] //= op.get_nodeattr("SIMD")
        src = np.arange(int(np.prod(shape))).reshape(shape).transpose(perm).reshape(-1)
        return np.maximum.accumulate(src) + 1


@register("OuterShuffle_hls")
class OuterShuffleHlsModel(OuterShuffleModel):
    def schedule(self, op):
        """The input generator's steady-state schedule, in three phases.

        A frame takes ``num_words`` cycles of input and ``num_words`` cycles of
        output, but the two cannot fully overlap: the reader has to wait for
        words the writer has not reached, and the writer has to wait for slots
        the reader has not released. The surplus over ``num_words`` is the same
        on both sides -- that is what keeps a period at one frame of tokens --
        and it is spent in two different places:

        * a demand-limited phase, where the input streams in solid and the
          output comes in bursts, one per step of the level whose coefficient
          outruns its output stride;
        * a free-limited phase, where the output drains solid and the input is
          admitted only as the free pointer releases slots;
        * between them a stretch where both run at one word per cycle.

        Two details of where the surplus falls, both of them the pipeline
        carrying state across the frame boundary rather than starting cold.
        Not every iteration of the demand level waits: the surplus only pays for
        so many, and the rest run their burst straight into the next one at the
        head of the period, on the lead the previous frame left. And the first
        wait of each limited phase is the short one, since the party about to be
        limited is already part-way through it when the frame turns over.

        Declines where the buffer does not hold a frame. There the writer stalls
        against its own frame rather than the previous one, several times over,
        and the period stops being one wait of each kind.
        """

        def loop_nest():
            """The ``Nest<>`` the HLS input generator is instantiated with.

            ``input_gen`` walks the *output* in order and reads the input word each
            output needs, so the nest is the output shape with the input strides
            permuted to match. Returns ``(extents, coeffs, num_words)`` after SIMD
            folds the innermost dimension: output word ``k`` at multi-index ``i``
            wants input word ``sum(i[t] * coeffs[t])``.
            """
            simd = op.get_nodeattr("SIMD")
            in_shape = list(op.get_nodeattr("transpose_in_shape"))
            perm = list(op.get_nodeattr("perm"))
            extents = [in_shape[p] for p in perm]
            adjusted = in_shape + [1]
            in_strides = [int(np.prod(adjusted[i + 1 :])) for i in range(len(in_shape))]
            coeffs = [in_strides[p] for p in perm]
            extents[-1] = int(extents[-1] / simd)
            coeffs = [1 if x == 1 else int(x / simd) for x in coeffs]
            return extents, coeffs, int(np.prod(extents))

        def output_strides(extents):
            """Distance in output words between successive iterations of each level."""
            strides = [1] * len(extents)
            acc = 1
            for t in range(len(extents) - 1, -1, -1):
                strides[t] = acc
                acc *= extents[t]
            return strides

        def demand_lead(extents, coeffs):
            """How far the reader's demand runs ahead of its own output index.

            Output ``k`` cannot leave until the writer has supplied every input word
            up to ``max_{j<=k} addr(j)``, so what paces the node is the largest
            ``addr(k) - k``. Both are linear in the loop indices --
            ``sum(i[t] * (coeffs[t] - strides[t]))`` -- so the maximum is reached by
            running every level with a positive difference to its end and leaving
            the rest at zero. The prefix maximum adds nothing: at any later ``k`` the
            same numerator is divided by a larger index.
            """
            strides = output_strides(extents)
            return 1 + sum(
                (extents[t] - 1) * max(0, coeffs[t] - strides[t]) for t in range(len(extents))
            )

        def free_pointer_steps(extents, coeffs, num_words):
            """When the free pointer releases buffer slots, as ``(period, words)``.

            The writer may not overwrite a word the reader has still to use, so the
            nest carries a free pointer alongside the read pointer. A level updates
            it only while its own sweep is monotonic and fits inside the enclosing
            one -- ``Nest``'s ``R_INNER`` -- and the chain stops at the first level
            where it does not. Each level that keeps it steps the pointer once per
            full period of that level; ``period`` is that period in output words.

            Returned outermost first, because the steps do not add: when several
            levels terminate on the same tick the outer one *replaces* the inner's
            increment rather than following it, so the pointer after ``k`` outputs is
            ``sum_i step_i * (k // period_i - k // period_{i-1})``.
            """
            L = len(extents)
            # W[j] is the read-pointer increment one full period of level j stands for
            W = [num_words] + [coeffs[j - 1] for j in range(1, L + 1)]
            carries = [True] * (L + 1)
            for j in range(L):
                carries[j + 1] = carries[j] and coeffs[j] > 0 and coeffs[j] * extents[j] <= W[j]
            rewind = [0] * (L + 1)
            for j in range(L - 1, -1, -1):
                rewind[j] = ((extents[j] - 1) * coeffs[j] + rewind[j + 1]) if carries[j + 1] else 0
            outputs_per_period = [1] * (L + 1)
            for j in range(L - 1, -1, -1):
                outputs_per_period[j] = outputs_per_period[j + 1] * extents[j]
            steps = [(outputs_per_period[j], W[j] - rewind[j]) for j in range(L) if carries[j]]
            if carries[L]:
                steps.append((1, W[L]))  # the innermost loop releases a word per output
            return [(p, w) for p, w in steps if p > 0 and w > 0]

        def free_pointer_at(outputs, steps):
            """Slots released after ``outputs`` outputs, for a scalar or an array."""
            outputs = np.asarray(outputs, dtype=np.int64)
            released = np.zeros_like(outputs)
            enclosing = None
            for period, step in steps:
                outer = np.zeros_like(outputs) if enclosing is None else outputs // enclosing
                released = released + step * (outputs // period - outer)
                enclosing = period
            return released

        def buffer_depth(extents, coeffs, num_words):
            """``BUF_SIZE`` of the reorder buffer, as ``input_gen()`` dimensions it.

            A completed loop always leaves the read pointer net forward, so the only
            backward movement is a single loop's terminal retraction; the deepest of
            those bounds how far behind the write pointer a read can reach.

            The closed form over the nest of what ``get_exp_cycles`` reads off
            ``_NestSim.max_rp_retract``, to the same ``WP_DELAY``.
            """
            WP_DELAY = 4
            L = len(extents)
            W = [num_words] + [coeffs[j - 1] for j in range(1, L + 1)]
            rewind = [0] * (L + 1)
            for j in range(L - 1, -1, -1):
                rewind[j] = (extents[j] - 1) * coeffs[j] + rewind[j + 1]
            retract = max([0] + [-(W[j] - rewind[j]) for j in range(L)])
            addr_bits = max(1, math.ceil(math.log2(max(1, retract + WP_DELAY + 2))))
            return 1 << addr_bits

        def free_lead(extents, coeffs, num_words, buf_size):
            """How far into a frame the reader must get before the writer may go on.

            The writer may place input word ``m`` of the next frame only once the
            reader has released the slot ``buf_size`` words behind it. The free
            pointer is a staircase, so a whole run of words waits on the same step of
            it, and the first word of that run waits longest. The worst such wait is
            what one frame of the writer's progress costs.
            """
            steps = free_pointer_steps(extents, coeffs, num_words)
            if any(period == 1 for period, _ in steps):
                return 0  # a word is released per output: the writer never waits
            edges = set()
            for period, _ in steps:
                if period <= num_words:
                    edges.update(range(period, num_words + 1, period))
            edges.add(num_words)
            ks = np.array(sorted(edges), dtype=np.int64)
            fp = free_pointer_at(ks, steps)
            # words served by each step of the staircase, as a half-open run
            hi = np.minimum(fp + buf_size - num_words - 1, num_words - 1)
            lo = np.maximum(0, np.concatenate(([-1], hi[:-1])) + 1)
            served = hi >= lo
            if not np.any(served):
                return None
            return int(np.max(ks[served] - lo[served]))

        def demand_phase(extents, coeffs):
            """The stride the reader's demand is paced at, as ``(burst, gap)``.

            Only levels whose coefficient outruns their output stride make the
            reader wait. The outermost of those sets the rhythm: everything below it
            is a burst of ``burst`` outputs the reader already has the words for,
            and the level's own step then leaves a ``gap`` with nothing to emit.
            Returns its iteration count too, since only some of those iterations
            wait -- the rest run their burst straight into the next one.
            """
            strides = output_strides(extents)
            outrun = [t for t in range(len(extents)) if coeffs[t] > strides[t] and extents[t] > 1]
            if not outrun:
                return None
            t0 = min(outrun)
            return strides[t0], coeffs[t0] - strides[t0], extents[t0]

        def beats(runs, ii, label):
            """One phase of the schedule, as ``(cycles, [read, write])`` runs.

            At II=1 the runs are the phase. Above it each transaction still happens
            on the first cycle of its iteration, so every run becomes that many
            iterations of one beat.
            """
            runs = [(int(n), v) for n, v in runs if n > 0]
            if ii == 1:
                return CharacteristicNode(label, runs, True)
            return CharacteristicNode(
                label,
                [(n, CharacteristicNode(label, [(1, v), (ii - 1, [0, 0])], True)) for n, v in runs],
                False,
            )

        # the beat an output takes to appear behind the write pointer, which
        # runs WP_DELAY cycles ahead of it
        READ_TO_WRITE = 3
        extents, coeffs, num_words = loop_nest()
        buf_size = buffer_depth(extents, coeffs, num_words)
        if buf_size < num_words:
            return None
        lead = free_lead(extents, coeffs, num_words, buf_size)
        if lead is None:
            return None
        # Vivado pipelines input_gen at II=1, except from 2024.2 on where a buffer
        # deep enough to be inferred as URAM carries its read latency of 3 into
        # the recurrence (as in get_exp_cycles, which warns when no Vivado is set)
        vivado_version = get_vivado_version() or (2024, 2)
        ii = 3 if vivado_version >= (2024, 2) and buf_size > 262144 else 1
        if lead == 0:
            # a slot is released per output, so the writer never waits and a
            # frame costs exactly its own words
            period = num_words
        else:
            # one beat under the frame count: a period is measured between two
            # last-outputs, which spans one cycle fewer than a frame does
            span = lead + demand_lead(extents, coeffs) + READ_TO_WRITE - 1
            period = max(num_words, span)
        surplus = period - num_words
        if surplus == 0:
            step = beats([(num_words, [1, 1])], ii, "OuterShuffle word")
            return CharacteristicNode("OuterShuffle frame", [(1, step)], False)

        paced = demand_phase(extents, coeffs)
        released = [(p, w) for p, w in free_pointer_steps(extents, coeffs, num_words) if p > 1]
        if paced is None or not released:
            return None
        burst, gap, iterations = paced
        admit_every, admit = min(released, key=lambda s: s[0])
        if admit >= admit_every:
            return None

        # Each turn of a limited phase spends a fixed share of the surplus, so
        # the surplus fixes both how many turns there are and how much shorter
        # than a full turn the odd one out is.
        turns_demand = -(-surplus // gap)
        turns_free = -(-surplus // (admit_every - admit))
        last_gap = surplus - (turns_demand - 1) * gap
        last_idle = surplus - (turns_free - 1) * (admit_every - admit)
        overlap = num_words - turns_demand * burst - surplus - turns_free * admit
        if overlap < 0:
            return None  # the two limited phases would have to share cycles

        # Only `turns_demand` of the level's iterations actually wait; the rest
        # run their burst straight into the next one, and they go first, while
        # the reader still has the head start the previous frame left it.
        running = max(0, min(overlap, (iterations - turns_demand - 1) * burst))
        overlap -= running

        # The short turn of each phase comes first: at a frame boundary the
        # party that is about to be limited is already part-way through its
        # first wait, so that one wait is the one cut short.
        phases = []
        if running > 0:
            phases.append((1, beats([(running, [1, 1])], ii, "OuterShuffle running start")))
        phases.append(
            (1, beats([(burst, [1, 1]), (last_gap, [1, 0])], ii, "OuterShuffle demand head"))
        )
        if turns_demand > 1:
            phases.append(
                (
                    turns_demand - 1,
                    beats([(burst, [1, 1]), (gap, [1, 0])], ii, "OuterShuffle demand"),
                )
            )
        if overlap > 0:
            phases.append((1, beats([(overlap, [1, 1])], ii, "OuterShuffle overlap")))
        phases.append(
            (1, beats([(admit, [1, 1]), (last_idle, [0, 1])], ii, "OuterShuffle free head"))
        )
        if turns_free > 1:
            phases.append(
                (
                    turns_free - 1,
                    beats(
                        [(admit, [1, 1]), (admit_every - admit, [0, 1])], ii, "OuterShuffle free"
                    ),
                )
            )
        return CharacteristicNode("OuterShuffle frame", phases, False)
