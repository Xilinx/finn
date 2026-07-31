############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

import math
import numpy as np
import os
import re
import warnings
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.basic import Characteristic_Node

# Write-pointer pipeline latency in input_gen(), and the extra beat an output
# takes to appear behind it. Both are properties of the HLS source.
_WP_DELAY = 4
_READ_TO_WRITE = _WP_DELAY - 1

# Above this buffer depth the reorder buffer is mapped to URAM, whose read
# latency of 3 the pipeline cannot hide, so it schedules at II=3 instead of 1.
# Vivado reached II=1 at every depth before 2024.2.
_URAM_DEPTH_THRESHOLD = 262144


class OuterShuffle(HWCustomOp):
    """Abstraction layer for HW OuterShuffle (rearrange and transpose) layers.
    Only permutations that do not effect the inner most dimensions are feasible"""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            "data_type": ("s", True, ""),
            "transpose_in_shape": ("ints", True, []),
            "in_shape": ("ints", True, []),
            "transpose_out_shape": ("ints", True, []),
            "out_shape": ("ints", True, []),
            "loop_coeffs": ("ints", True, []),
            "perm": ("ints", True, []),
            "SIMD": ("i", False, 1),
            "NumChannels": ("i", False, 128),
            "original_node_name": ("s", False, ""),  # Track original shuffle name for SIMD config
            "original_simd": ("i", False, 1),  # Track original shuffle SIMD for config export
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_normal_input_shape(self, ind=0):
        return self.get_nodeattr("in_shape")

    def get_normal_output_shape(self, ind=0):
        return self.get_nodeattr("out_shape")

    def execute_node(self, context, graph):
        node = self.onnx_node
        input_data = context[node.input[0]]
        input_reshaped = input_data.reshape(self.get_nodeattr("transpose_in_shape"))
        transposed = np.transpose(input_reshaped, axes=self.get_nodeattr("perm"))
        output_reshaped = transposed.reshape(self.get_nodeattr("out_shape"))
        context[node.output[0]] = output_reshaped

    def get_input_datatype(self, ind=0):
        data_type = DataType[self.get_nodeattr("data_type")]
        return data_type

    def infer_node_datatype(self, model):
        node = self.onnx_node
        dt = model.get_tensor_datatype(node.input[0])
        if dt != self.get_input_datatype():
            warn_str = (
                f"data_type changing for {node.name}: {str(self.get_input_datatype())} -> {str(dt)}"
            )
            warnings.warn(warn_str)
        self.set_nodeattr("data_type", dt.name)
        model.set_tensor_datatype(node.output[0], dt)

    def verify_node(self):
        raise NotImplementedError("This function is not yet immplemented.")

    def get_instream_width(self, ind=0):
        ibits = self.get_input_datatype().bitwidth()
        simd = self.get_nodeattr("SIMD")
        return ibits * simd

    def get_outstream_width(self, ind=0):
        obits = self.get_output_datatype().bitwidth()
        simd = self.get_nodeattr("SIMD")
        return obits * simd

    def get_output_datatype(self, ind=0):
        data_type = DataType[self.get_nodeattr("data_type")]
        return data_type

    def get_folded_output_shape(self, ind=0):
        normal_oshape = list(self.get_normal_output_shape())
        simd = self.get_nodeattr("SIMD")
        assert normal_oshape[-1] % simd == 0, "SIMD must divide into the innermost output dimension"
        fold = int(normal_oshape[-1] / simd)
        folded_oshape = normal_oshape[:-1] + [fold, simd]
        return tuple(folded_oshape)

    def get_folded_input_shape(self, ind=0):
        normal_ishape = list(self.get_normal_input_shape())
        simd = self.get_nodeattr("SIMD")
        assert normal_ishape[-1] % simd == 0, "SIMD must divide into the innermost input dimension"
        fold = int(normal_ishape[-1] / simd)
        folded_ishape = normal_ishape[:-1] + [fold, simd]
        return tuple(folded_ishape)

    def loop_nest(self):
        """The ``Nest<>`` the HLS input generator is instantiated with.

        ``input_gen`` walks the *output* in order and reads the input word each
        output needs, so the nest is the output shape with the input strides
        permuted to match. Returns ``(extents, coeffs, num_words)`` after SIMD
        folds the innermost dimension: output word ``k`` at multi-index ``i``
        wants input word ``sum(i[t] * coeffs[t])``.
        """
        simd = self.get_nodeattr("SIMD")
        in_shape = list(self.get_nodeattr("transpose_in_shape"))
        perm = list(self.get_nodeattr("perm"))
        extents = [in_shape[p] for p in perm]
        adjusted = in_shape + [1]
        in_strides = [int(np.prod(adjusted[i + 1 :])) for i in range(len(in_shape))]
        coeffs = [in_strides[p] for p in perm]
        extents[-1] = int(extents[-1] / simd)
        coeffs = [1 if x == 1 else int(x / simd) for x in coeffs]
        return extents, coeffs, int(np.prod(extents))

    def output_strides(self, extents):
        """Distance in output words between successive iterations of each level."""
        strides = [1] * len(extents)
        acc = 1
        for t in range(len(extents) - 1, -1, -1):
            strides[t] = acc
            acc *= extents[t]
        return strides

    def demand_lead(self, extents, coeffs):
        """How far the reader's demand runs ahead of its own output index.

        Output ``k`` cannot leave until the writer has supplied every input word
        up to ``max_{j<=k} addr(j)``, so what paces the node is the largest
        ``addr(k) - k``. Both are linear in the loop indices --
        ``sum(i[t] * (coeffs[t] - strides[t]))`` -- so the maximum is reached by
        running every level with a positive difference to its end and leaving
        the rest at zero. The prefix maximum adds nothing: at any later ``k`` the
        same numerator is divided by a larger index.
        """
        strides = self.output_strides(extents)
        return 1 + sum(
            (extents[t] - 1) * max(0, coeffs[t] - strides[t]) for t in range(len(extents))
        )

    def free_pointer_steps(self, extents, coeffs, num_words):
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

    def free_pointer_at(self, outputs, steps):
        """Slots released after ``outputs`` outputs, for a scalar or an array."""
        outputs = np.asarray(outputs, dtype=np.int64)
        released = np.zeros_like(outputs)
        enclosing = None
        for period, step in steps:
            outer = np.zeros_like(outputs) if enclosing is None else outputs // enclosing
            released = released + step * (outputs // period - outer)
            enclosing = period
        return released

    def buffer_depth(self, extents, coeffs, num_words):
        """``BUF_SIZE`` of the reorder buffer, as ``input_gen()`` dimensions it.

        A completed loop always leaves the read pointer net forward, so the only
        backward movement is a single loop's terminal retraction; the deepest of
        those bounds how far behind the write pointer a read can reach.
        """
        L = len(extents)
        W = [num_words] + [coeffs[j - 1] for j in range(1, L + 1)]
        rewind = [0] * (L + 1)
        for j in range(L - 1, -1, -1):
            rewind[j] = (extents[j] - 1) * coeffs[j] + rewind[j + 1]
        retract = max([0] + [-(W[j] - rewind[j]) for j in range(L)])
        addr_bits = max(1, math.ceil(math.log2(max(1, retract + _WP_DELAY + 2))))
        return 1 << addr_bits

    def free_lead(self, extents, coeffs, num_words, buf_size):
        """How far into a frame the reader must get before the writer may go on.

        The writer may place input word ``m`` of the next frame only once the
        reader has released the slot ``buf_size`` words behind it. The free
        pointer is a staircase, so a whole run of words waits on the same step of
        it, and the first word of that run waits longest. The worst such wait is
        what one frame of the writer's progress costs.
        """
        steps = self.free_pointer_steps(extents, coeffs, num_words)
        if any(period == 1 for period, _ in steps):
            return 0  # a word is released per output: the writer never waits
        edges = set()
        for period, _ in steps:
            if period <= num_words:
                edges.update(range(period, num_words + 1, period))
        edges.add(num_words)
        ks = np.array(sorted(edges), dtype=np.int64)
        fp = self.free_pointer_at(ks, steps)
        # words served by each step of the staircase, as a half-open run
        hi = np.minimum(fp + buf_size - num_words - 1, num_words - 1)
        lo = np.maximum(0, np.concatenate(([-1], hi[:-1])) + 1)
        served = hi >= lo
        if not np.any(served):
            return None
        return int(np.max(ks[served] - lo[served]))

    def demand_phase(self, extents, coeffs):
        """The stride the reader's demand is paced at, as ``(burst, gap)``.

        Only levels whose coefficient outruns their output stride make the
        reader wait. The outermost of those sets the rhythm: everything below it
        is a burst of ``burst`` outputs the reader already has the words for,
        and the level's own step then leaves a ``gap`` with nothing to emit.
        Returns its iteration count too, since only some of those iterations
        wait -- the rest run their burst straight into the next one.
        """
        strides = self.output_strides(extents)
        outrun = [t for t in range(len(extents)) if coeffs[t] > strides[t] and extents[t] > 1]
        if not outrun:
            return None
        t0 = min(outrun)
        return strides[t0], coeffs[t0] - strides[t0], extents[t0]

    def pipeline_ii(self):
        """Cycles per pipeline iteration: 1, or 3 where the buffer lands in URAM.

        Vivado pipelines ``input_gen`` at II=1 whatever the buffer depth up to
        2024.1. From 2024.2 a buffer deep enough to be inferred as URAM carries
        that memory's read latency into the recurrence and the loop schedules at
        II=3 instead.
        """
        extents, coeffs, num_words = self.loop_nest()
        if self.buffer_depth(extents, coeffs, num_words) <= _URAM_DEPTH_THRESHOLD:
            return 1
        vivado_path = os.environ.get("XILINX_VIVADO")
        match = re.search(r"\b(20\d{2})\.(1|2)\b", vivado_path) if vivado_path else None
        if match is None:
            return 1
        return 1 if (int(match.group(1)), int(match.group(2))) < (2024, 2) else 3

    def word_addresses(self, extents, coeffs):
        """The input word each output wants, over one frame."""
        indices = np.meshgrid(*[np.arange(n) for n in extents], indexing="ij")
        addresses = np.zeros(indices[0].shape, dtype=np.int64)
        for index, coeff in zip(indices, coeffs):
            addresses += index * coeff
        return addresses.ravel()

    def stalled_frame_cycles(self, extents, coeffs, num_words, buf_size):
        """One frame where the writer waits on the free pointer of that frame.

        Two waits hold each other up. Output ``k`` cannot leave before the
        writer has supplied ``needed(k)`` words; input word ``m`` cannot be
        placed before the reader has released the slot ``buf_size`` behind it.
        Neither side is one wait deep -- each release lets the writer run until
        the next, so the two alternate as many times as the free pointer steps,
        and the frame is the fixed point rather than a sum.

        Both recurrences are ``x(i) = max(x(i-1) + 1, y(f(i)) + c)``, which is a
        prefix maximum of ``y(f(i)) + c - i``. So a pass over either side is one
        ``maximum.accumulate`` over the loop nest's own words -- no cycle is ever
        stepped through, and the alternations converge in as many passes as there
        are steps in the free pointer.
        """
        addresses = self.word_addresses(extents, coeffs)
        needed = np.maximum.accumulate(addresses) + 1
        steps = self.free_pointer_steps(extents, coeffs, num_words)
        released = self.free_pointer_at(np.arange(1, num_words + 1), steps)
        words = np.arange(num_words, dtype=np.int64)
        # the output after which word m may be written; words within buf_size of
        # the frame start are already free and wait for nothing
        waits = words + 1 - buf_size
        release_of = np.searchsorted(released, waits, side="left")
        held = waits > 0
        if np.any(held & (release_of >= num_words)):
            return None  # a word this frame needs is never released inside it
        # one pass carries one writer-reader alternation, and there are no more
        # of those than the free pointer has steps
        passes = num_words // min(period for period, _ in steps) + 8
        emitted = np.zeros(num_words, dtype=np.int64)
        for _ in range(passes):
            ready = np.where(held, emitted[np.clip(release_of, 0, num_words - 1)] + 1, 1)
            written = words + np.maximum.accumulate(ready - words)
            arrived = written[needed - 1] + _WP_DELAY
            settled = words + np.maximum.accumulate(arrived - words)
            if np.array_equal(settled, emitted):
                break
            emitted = settled
        return int(emitted[-1])

    def get_exp_cycles(self):
        """Cycles for one frame: the demand lead, then the frame drained at II=1.

        The output cannot run faster than one word per cycle, and output ``k``
        additionally waits for the input word it needs. The frame therefore ends
        at ``max_k (needed(k) + (num_words - k))``, which is the demand lead
        past the frame, plus the write-pointer pipeline behind it.

        That is the whole story while the reorder buffer holds a frame, which is
        the shape the decomposition usually produces: the writer streams in
        without ever waiting, so a closed form over the nest is exact. Below that
        it waits on the free pointer of its own frame and the two waits have to
        be solved together -- see ``stalled_frame_cycles``. This count also sets
        the window ``derive_characteristic`` records a node over, so running
        short here would quietly truncate that too.
        """
        extents, coeffs, num_words = self.loop_nest()
        cycles = num_words + self.demand_lead(extents, coeffs) + _READ_TO_WRITE
        buf_size = self.buffer_depth(extents, coeffs, num_words)
        if buf_size < num_words:
            stalled = self.stalled_frame_cycles(extents, coeffs, num_words, buf_size)
            if stalled is not None:
                cycles = max(cycles, stalled)
        return int(cycles * self.pipeline_ii())

    def beats(self, runs, ii, label):
        """One phase of the schedule, as ``(cycles, [read, write])`` runs.

        At II=1 the runs are the phase. Above it each transaction still happens
        on the first cycle of its iteration, so every run becomes that many
        iterations of one beat.
        """
        runs = [(int(n), v) for n, v in runs if n > 0]
        if ii == 1:
            return Characteristic_Node(label, runs, True)
        return Characteristic_Node(
            label,
            [(n, Characteristic_Node(label, [(1, v), (ii - 1, [0, 0])], True)) for n, v in runs],
            False,
        )

    def get_tree_model(self):
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
        extents, coeffs, num_words = self.loop_nest()
        buf_size = self.buffer_depth(extents, coeffs, num_words)
        if buf_size < num_words:
            return None
        lead = self.free_lead(extents, coeffs, num_words, buf_size)
        if lead is None:
            return None
        ii = self.pipeline_ii()
        if lead == 0:
            # a slot is released per output, so the writer never waits and a
            # frame costs exactly its own words
            period = num_words
        else:
            # one beat under the frame count: a period is measured between two
            # last-outputs, which spans one cycle fewer than a frame does
            span = lead + self.demand_lead(extents, coeffs) + _READ_TO_WRITE - 1
            period = max(num_words, span)
        surplus = period - num_words
        if surplus == 0:
            step = self.beats([(num_words, [1, 1])], ii, "OuterShuffle word")
            return Characteristic_Node("OuterShuffle frame", [(1, step)], False)

        paced = self.demand_phase(extents, coeffs)
        released = [(p, w) for p, w in self.free_pointer_steps(extents, coeffs, num_words) if p > 1]
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
            phases.append((1, self.beats([(running, [1, 1])], ii, "OuterShuffle running start")))
        phases.append(
            (1, self.beats([(burst, [1, 1]), (last_gap, [1, 0])], ii, "OuterShuffle demand head"))
        )
        if turns_demand > 1:
            phases.append(
                (
                    turns_demand - 1,
                    self.beats([(burst, [1, 1]), (gap, [1, 0])], ii, "OuterShuffle demand"),
                )
            )
        if overlap > 0:
            phases.append((1, self.beats([(overlap, [1, 1])], ii, "OuterShuffle overlap")))
        phases.append(
            (1, self.beats([(admit, [1, 1]), (last_idle, [0, 1])], ii, "OuterShuffle free head"))
        )
        if turns_free > 1:
            phases.append(
                (
                    turns_free - 1,
                    self.beats(
                        [(admit, [1, 1]), (admit_every - admit, [0, 1])], ii, "OuterShuffle free"
                    ),
                )
            )
        return Characteristic_Node("OuterShuffle frame", phases, False)
