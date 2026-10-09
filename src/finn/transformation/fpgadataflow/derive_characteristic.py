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


"""FIFO sizing from token access vectors: the chained-TAV strategy only.

The flow, as ``step_set_fifo_depths`` runs it:

1. ``JustInTimeSynthesize`` builds IP for the nodes that have to be simulated.
2. ``DeriveTokenAccessVectors`` gives every characterizable node its token
   access vectors (TAVs) ``io_chrc_in``/``io_chrc_out``: cumulative
   per-cycle transaction counts over two periods, one row per stream. This is
   the only place a TAV is written. Nothing below modifies one.
3. ``DeriveFIFOSizes`` reads the TAVs, derives a wall-clock schedule for every
   edge with ``derive_chained_tav_depths`` and writes the resulting depths to
   ``inFIFODepths``/``outFIFODepths``.
"""

import numpy as np
import os
import qonnx.custom_op.registry as registry
import warnings
from qonnx.transformation.base import NodeLocalTransformation, Transformation

from finn.transformation.fpgadataflow.prepare_ip import _codegen_single_node
from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.util.basic import decompress_string_to_numpy
from finn.util.fpgadataflow import is_hls_node, is_rtl_node

#: Ops that are never characterized: fork, join and the FIFOs themselves.
_UNCHARACTERIZED_OPS = [
    "AddStreams_hls",
    "DuplicateStreams_hls",
    "StreamingFIFO_hls",
    "StreamingFIFO_rtl",
]

#: Ops the chained-TAV pass treats as transparent, by op type: a token handed to
#: them is handed on in the same cycle. Any node without a TAV is treated the
#: same way. Note that this matches *every* ElementwiseAdd, including a unary
#: one with a constant operand, which does carry a TAV of its own.
_TRANSPARENT_OPS = (
    "DuplicateStreams_hls",
    "AddStreams_hls",
    "ElementwiseAdd_hls",
    "ElementwiseAdd_rtl",
)


# ---------------------------------------------------------------------------
# Stage 1: TAV generation
# ---------------------------------------------------------------------------


def is_stream_join(node):
    """True if ``node`` merges two streamed inputs into one output.

    Both ``AddStreams`` (deprecated) and ``ElementwiseAdd`` are matched; an
    ``ElementwiseAdd`` is only a join when *both* of its inputs are streams.
    Joins are not characterized: rtlsim characterization drives ``in0`` only
    and would stall waiting on the second stream.
    """
    if node is None or not node.op_type.startswith(("AddStreams", "ElementwiseAdd")):
        return False
    if len(node.input) < 2:
        return False
    inst = registry.getCustomOp(node)
    attr_types = inst.get_nodeattr_types()
    for style in ("lhs_style", "rhs_style"):
        if style in attr_types and inst.get_nodeattr(style) != "input":
            return False
    return True


class JustInTimeSynthesize(Transformation):
    def __init__(self, part, clk_period, only_without_tree_model=False):
        super().__init__()
        self.part = part
        self.clk_period = clk_period
        self.only_without_tree_model = only_without_tree_model

    def apply(self, model):
        for node in model.graph.node:
            inst = registry.getCustomOp(node)
            if (is_hls_node(node) or is_rtl_node(node)) and (
                (
                    (inst.get_tree_model() is None and self.only_without_tree_model)
                    or not self.only_without_tree_model
                )
                and (inst.get_nodeattr("io_chrc_in") == "")
            ):
                _codegen_single_node(
                    node,
                    model,
                    self.part,
                    self.clk_period,
                )

                op_type = node.op_type
                if is_hls_node(node):
                    try:
                        # ensure that code is generated
                        assert (
                            inst.get_nodeattr("code_gen_dir_ipgen") != ""
                        ), """Node
                        attribute "code_gen_dir_ipgen" is empty. Please run
                        transformation PrepareIP first."""
                        if os.path.isdir(inst.get_nodeattr("ipgen_path")) or inst.get_nodeattr(
                            "code_gen_dir_ipgen"
                        ) not in inst.get_nodeattr("ipgen_path"):
                            # call the compilation function for this node
                            inst.ipgen_singlenode_code()
                        else:
                            warnings.warn("Using pre-existing IP for %s" % node.name)
                        # ensure that executable path is now set
                        assert (
                            inst.get_nodeattr("ipgen_path") != ""
                        ), """Transformation
                        HLSSynthIP was not successful. Node attribute "ipgen_path"
                        is empty."""
                    except KeyError:
                        raise Exception("Custom op_type %s is currently not supported." % op_type)

        model = model.transform(ReplaceVerilogRelPaths())
        for node in model.graph.node:
            inst = registry.getCustomOp(node)
            if (
                (is_hls_node(node) or is_rtl_node(node))
                and (
                    (inst.get_tree_model() is None and self.only_without_tree_model)
                    or not self.only_without_tree_model
                )
                and node.op_type not in _UNCHARACTERIZED_OPS
                and not is_stream_join(node)
                and (inst.get_nodeattr("rtlsim_so") == "")
            ):
                try:
                    inst.prepare_rtlsim()
                    # ensure that executable path is now set
                    assert (
                        inst.get_nodeattr("rtlsim_so") != ""
                    ), "Failed to prepare RTLSim, no rtlsim_so attribute found."
                except KeyError:
                    raise Exception("Custom op_type %s is currently not supported." % op_type)

        model = model.transform(SetExecMode("rtlsim"))

        return (model, False)


class DeriveTokenAccessVectors(NodeLocalTransformation):
    """For each node in the graph, derive the token access vectors, either from
    the node's tree model or by rtlsim of the node in isolation, and store them
    in ``io_chrc_in``/``io_chrc_out``.

    * period (int) desired period over which the characteristic function
      will be derived.

    * num_workers (int or None) number of parallel workers, see documentation in
      NodeLocalTransformation for more details.
    """

    def __init__(
        self,
        model,
        period,
        strategy,
        fpga_part,
        clk_period,
        num_workers=None,
        nodes_to_ignore=[],
    ):
        super().__init__(num_workers=num_workers)
        self.model = model
        self.period = period
        self.strategy = strategy
        self.fpga_part = fpga_part
        self.clk_period = clk_period
        self.nodes_to_ignore = set(nodes_to_ignore)

    def applyNodeLocal(self, node):
        op_type = node.op_type
        if is_hls_node(node) or is_rtl_node(node):
            try:
                inst = registry.getCustomOp(node)
                if node.name in self.nodes_to_ignore:
                    return (node, False)
                # Forks, joins and FIFOs get no TAV; the sizing pass treats
                # them as transparent.
                if op_type not in _UNCHARACTERIZED_OPS and not is_stream_join(node):
                    inst.derive_token_access_vectors(
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


# ---------------------------------------------------------------------------
# Stage 2: reading TAVs
# ---------------------------------------------------------------------------


def read_tav_rows(inst):
    """A node's input and output TAVs, one row per stream, or None if absent.

    Each row is a cumulative token count per clock cycle over two periods.
    rtlsim traces one row per stream (inputs with a nonzero stream width, in
    ``node.input`` order; outputs in ``node.output`` order). A tree model emits
    a single row.
    """
    out = []
    for name in ("io_chrc_in", "io_chrc_out"):
        raw = inst.get_nodeattr(name)
        out.append(
            None if raw == "" else np.atleast_2d(decompress_string_to_numpy(raw)).astype(np.int64)
        )
    return out[0], out[1]


def input_rows_collapsed(model, node):
    """QUIRK inherited from the full flow: True if this node's input TAV is
    reduced to row 0.

    In the full flow ``DelayCharacteristicFunctions`` re-saved ``io_chrc_in``
    as its first row for every node whose ``input[0]`` has a producer (looking
    through one DWC). The sidecar path depends only on the node and attribute
    name, so this also overwrote the file ``io_chrc_in_original`` pointed to.
    The per-stream input binding in ``stream_curves`` therefore only ever saw
    several rows on nodes fed directly by a graph input.
    """
    producer = model.find_producer(node.input[0]) if len(node.input) else None
    if producer is not None and "StreamingDataWidthConverter" in producer.name:
        producer = model.find_producer(producer.input[0])
    return producer is not None


def inherited_fork_join_tav_lengths(model):
    """QUIRK inherited from the full flow: TAV lengths forks and joins carry.

    In the full flow ``HandleBranches`` copied a neighbour's TAV onto a fork
    and the join it reconverges at, but only for pairs it could trace (both
    join inputs walk back along ``input[0]`` to the same fork). The fork got
    its producer's output TAV (or, if it forks a graph input, its first
    consumer's input TAV); the join got its consumer's input TAV. Joins are
    processed in graph order, and a copy made earlier is what a later lookup
    sees.

    The chained-TAV pass treats forks and joins as transparent, but still reads
    the copied TAV's *length* as the node's own period, which feeds
    ``chain_period`` and ``drives_pacer``. Returns ``{node name: length}``.
    """
    fork_ops = ("DuplicateStreams_hls", "ReplicateStream_hls")
    assigned = {}

    def tav_len(node, attr):
        if node.name in assigned:
            return assigned[node.name]
        raw = registry.getCustomOp(node).get_nodeattr(attr)
        return None if raw == "" else np.atleast_2d(decompress_string_to_numpy(raw)).shape[-1]

    def fork_of(join, idx):
        last = model.find_producer(join.input[idx])
        seen = set()
        while last is not None and last.op_type not in fork_ops:
            if last.name in seen or len(last.input) < 1:
                return None
            seen.add(last.name)
            last = model.find_producer(last.input[0])
        return last

    for join in [n for n in model.graph.node if is_stream_join(n)]:
        f0, f1 = fork_of(join, 0), fork_of(join, 1)
        if f0 is None or f1 is None or f0 is not f1:
            continue
        prod = model.find_producer(f0.input[0])
        if prod is not None:
            assigned[f0.name] = tav_len(prod, "io_chrc_out")
        else:
            assigned[f0.name] = tav_len(model.find_consumer(f0.output[0]), "io_chrc_in")
        cons = model.find_consumer(join.output[0])
        if cons is not None:
            assigned[join.name] = tav_len(cons, "io_chrc_in")
    return {k: v for k, v in assigned.items() if v is not None}


def stream_curves(node, inst, rows_in, rows_out):
    """Bind TAV rows to tensor names: ``({tensor: curve}, {tensor: curve})``.

    The per-stream binding is only taken when the row count matches the stream
    count, which is the shape rtlsim produces. Otherwise (tree models) row 0 is
    used for every stream.
    """
    in_curves, out_curves = {}, {}
    if rows_in is not None:
        streamed = []
        for i, t in enumerate(node.input):
            try:
                if inst.get_instream_width(i) == 0:
                    continue
            except Exception:
                pass
            streamed.append(t)
        if len(streamed) == rows_in.shape[0]:
            in_curves = {t: rows_in[k] for k, t in enumerate(streamed)}
        else:
            in_curves = {t: rows_in[0] for t in node.input}
    if rows_out is not None:
        if len(node.output) == rows_out.shape[0]:
            out_curves = {t: rows_out[k] for k, t in enumerate(node.output)}
        else:
            out_curves = {t: rows_out[0] for t in node.output}
    return in_curves, out_curves


# ---------------------------------------------------------------------------
# Stage 3: schedule and occupancy arithmetic
# ---------------------------------------------------------------------------


def arrival_of(schedule, counts):
    """Time at which ``counts[i]`` tokens have arrived, given per-token times.

    Counts beyond the schedule are clipped to its last token. An empty schedule
    constrains nothing, so every arrival time is 0.
    """
    if len(schedule) == 0:
        return np.zeros(len(counts), dtype=np.int64)
    idx = np.clip(counts, 0, len(schedule)) - 1
    return np.where(counts > 0, schedule[np.maximum(idx, 0)], 0)


def periodic_peak(w, r, period):
    """Peak steady-state occupancy of one frame's schedules repeated every
    ``period`` cycles::

        occ(t) = sum over f of g(t - f*period),   g(u) = |w <= u| - |r <= u|

    ``g`` has finite support, so the sum is finite, and occupancy only changes
    at an event, so event times modulo ``period`` are the only candidates.
    """
    w = np.sort(np.asarray(w, dtype=np.int64))
    r = np.sort(np.asarray(r, dtype=np.int64))
    lo = int(min(w[0], r[0]))
    hi = int(max(w[-1], r[-1]))
    # how many periods of history can still be in flight
    reps = int((hi - lo) // period) + 2
    cand = np.unique(np.concatenate([w, r]) % period) + lo - (lo % period)
    cand = np.concatenate([cand, cand + period])
    occ = np.zeros(len(cand), dtype=np.int64)
    for k in range(reps):
        u = cand + k * period
        occ += np.searchsorted(w, u, side="right") - np.searchsorted(r, u, side="right")
    return int(max(0, occ.max()))


def peak_occupancy(write_times, read_times, period, per_frame, fill_only):
    """Peak occupancy of an edge in steady state.

    The schedules hold two frames. When ``per_frame`` splits them cleanly,
    each frame is evaluated on its own: the first (fill) frame carries the
    pipeline-filling transient and the latency difference a join has to buffer;
    the second (steady) frame carries a producer's run-ahead. Chain edges take
    the larger; ``fill_only`` edges (joins, reconvergent branches) take the
    fill frame alone.
    """
    n = min(len(write_times), len(read_times))
    if n == 0:
        return 0
    w = np.sort(np.asarray(write_times[:n], dtype=np.int64))
    r = np.sort(np.asarray(read_times[:n], dtype=np.int64))
    if per_frame and 2 * per_frame <= n:
        fill = periodic_peak(w[:per_frame], r[:per_frame], period)
        if fill_only:
            return fill
        steady = periodic_peak(w[n - per_frame : n], r[n - per_frame : n], period)
        return max(fill, steady)
    return periodic_peak(w, r, period)


def causal_writes(write_times, in_scheds):
    """Hold each output token of the first frame until the inputs that carry it
    have arrived.

    A TAV has an arbitrary phase, so its ``first_write - first_read`` is not the
    node's latency: a 64:1 width converter measures as writing one cycle after
    its first read. With ``N_in`` inputs and ``N_out`` outputs a frame, output
    *j* cannot precede input ``ceil((j+1) * N_in / N_out)``. Only applied over
    the first frame (where the pipeline starts empty) and only to compacting
    nodes (``N_in > N_out``).
    """
    if not in_scheds:
        return write_times
    n_out = len(write_times)
    if n_out == 0:
        return write_times
    out = np.array(write_times, dtype=np.int64, copy=True)
    fill = max(1, n_out // 2)  # a TAV stores two frames
    idx = np.arange(1, fill + 1, dtype=np.int64)
    for sched, _ in in_scheds.values():
        n_in = len(sched)
        if n_in == 0 or n_in <= n_out:
            continue  # expanding or one-to-one: the clock already covers it
        need = np.minimum(-(-idx * (n_in // 2) // fill), n_in)
        out[:fill] = np.maximum(out[:fill], sched[need - 1])
    return np.maximum.accumulate(out)


def burst_above_rate(read_times, rate):
    """Largest excursion of demand above a constant supply ``rate``:
    ``max over t1 < t2 of C(t2) - C(t1) - rate * (t2 - t1)``."""
    if len(read_times) == 0 or rate <= 0:
        return 0
    g = np.arange(1, len(read_times) + 1) - rate * read_times
    return int(np.ceil(np.max(g - np.minimum.accumulate(g))))


# ---------------------------------------------------------------------------
# Stage 4: the chained-TAV pass
# ---------------------------------------------------------------------------

#: The pacer tolerance: a node within 0.5% of the slowest node's period counts
#: as the pacer, so a bottleneck several nodes wide is treated as one.
PACER_TOLERANCE = 0.995

#: Depth cap for edges whose consumer can be throttled (it neither runs at the
#: pacer's period nor feeds anything that does). The SRL depth limit.
THROTTLED_CAP = 256

#: The cap only fires when the producer's blocking budget exceeds the edge's
#: peak by this factor; an allowance that only just covers the peak is a
#: first-order estimate that cannot be trusted.
CAP_MARGIN = 1.25


def derive_chained_tav_depths(model, global_period, causal):
    """Per-edge FIFO depths from token arrival times on the dataflow DAG.

    Returns ``{tensor_name: depth}``. Edges not in the dict get the minimum.

    Each node's TAV is a schedule on a *local* clock. Visiting nodes in
    topological order, the local clock of node v is mapped to wall-clock time by
    the max-plus recurrence

        R(c) = max(R(c - 1) + 1, arrival time of the tokens read by cycle c)

    so a node stalls exactly as long as its latest input is late. The depth an
    edge needs is the peak of ``written_by(t) - read_by(t)`` over those
    schedules, relaxed by the producer's slack and floored by the consumer's
    burst demand.

    ``causal`` applies ``causal_writes`` to every node's output schedule.
    """
    nodes = [n for n in model.graph.node if is_hls_node(n) or is_rtl_node(n)]

    # -- read every TAV once -------------------------------------------------
    tavs = {}  # node -> (row 0 of input TAV, row 0 of output TAV)
    curves = {}  # node -> ({in tensor: curve}, {out tensor: curve})
    for node in nodes:
        try:
            inst = registry.getCustomOp(node)
            rows_in, rows_out = read_tav_rows(inst)
            if rows_in is not None and input_rows_collapsed(model, node):
                rows_in = rows_in[:1]
            tavs[node.name] = (
                None if rows_in is None else rows_in[0],
                None if rows_out is None else rows_out[0],
            )
            curves[node.name] = stream_curves(node, inst, rows_in, rows_out)
        except Exception:
            tavs[node.name] = (None, None)
            curves[node.name] = ({}, {})

    # own_period[v]: half the TAV length. Forks and joins have no TAV and are
    # transparent below, but still carry a period (see
    # inherited_fork_join_tav_lengths).
    inherited = inherited_fork_join_tav_lengths(model)
    own_period = {}
    for node in nodes:
        tin = tavs[node.name][0]
        if tin is not None:
            own_period[node.name] = len(tin) // 2
        elif node.name in inherited:
            own_period[node.name] = inherited[node.name] // 2

    # -- graph-level quantities ---------------------------------------------
    # Global period: the slower of the caller's estimate and the slowest
    # measured TAV period (some ops' cycle estimates under-state their period).
    periods = list(own_period.values())
    measured = max(periods) if periods else 0
    global_period = max(int(global_period or 0), measured, 1)
    graph_inputs = {x.name for x in model.graph.input}
    pacer_period = max([global_period] + periods) * PACER_TOLERANCE

    # drives_pacer[v]: v runs at the pacer's period, or feeds something that
    # does. A consumer for which this is False can be throttled.
    drives_pacer = {}
    for node in reversed(nodes):
        own = own_period.get(node.name, 0)
        below = False
        for t in node.output:
            cons = model.find_consumer(t)
            if cons is not None and drives_pacer.get(cons.name):
                below = True
        drives_pacer[node.name] = own >= pacer_period or below

    # in_branch: nodes between a fork and the node its branches reconverge at.
    def reachable(tensor):
        seen, stack = set(), [model.find_consumer(tensor)]
        while stack:
            n = stack.pop()
            if n is None or n.name in seen:
                continue
            seen.add(n.name)
            for t in n.output:
                stack.append(model.find_consumer(t))
        return seen

    in_branch = set()
    for node in nodes:
        outs = [t for t in node.output if model.find_consumer(t) is not None]
        if len(outs) < 2:
            continue
        reach = [reachable(t) for t in outs]
        shared = set.intersection(*reach)
        if not shared:
            continue  # the branches never meet again; ordinary chains
        for r in reach:
            in_branch |= r - shared

    arrival = {}  # tensor -> per-token wall-clock write times
    depths = {}
    #: tensor -> (tokens the producer writes on it per frame,
    #:            t_up: longest period in the chain feeding this edge since the
    #:                  last pacer,
    #:            the value to hand further downstream: 0 past a pacer)
    supply = {}

    def chain_period(node):
        own = own_period.get(node.name, 0)
        upstream = [supply[t][2] for t in node.input if t in supply]
        chain = max([own] + upstream) if (upstream or own) else global_period
        propagated = 0 if own >= pacer_period else chain
        return chain, propagated

    def idle_window_floor(tensor, curve, consumer):
        """Tokens that arrive while the consumer is not reading this stream:
        a frame's tokens times the fraction of the consumer's period spent
        outside its read window. Zero for continuous readers."""
        if curve is None or tensor not in supply:
            return 0
        half = len(curve) // 2
        if half < 2:
            return 0
        per_frame = int(curve[half - 1])
        if per_frame <= 0:
            return 0
        reads = np.searchsorted(curve, np.arange(1, per_frame + 1), side="left")
        duty = min(1.0, float(reads[-1] - reads[0] + 1) / float(half))
        want = int(round(per_frame * (1.0 - duty)))
        if not drives_pacer.get(consumer, True):
            want = min(want, THROTTLED_CAP)
        return want

    def absorbed_frame_floor(node, tensor):
        """One folded input frame, for a consumer whose TAV reads nothing in
        the stored window (a node that absorbs its whole input frame before
        releasing anything, e.g. a FINNLoop)."""
        cons = registry.getCustomOp(node)
        try:
            idx = list(node.input).index(tensor)
            shape = cons.get_folded_input_shape(idx)
        except Exception:
            return 0
        if shape is None or len(shape) < 2:
            return 0
        n = 1
        for d in shape[:-1]:
            n *= int(d)
        return int(n)

    def depth_from_peak(tensor, peak, read_times, consumer, join, idle_floor=0):
        """Turn an edge's peak occupancy into its depth."""
        if tensor not in supply or join:
            # Joins and reconvergent branches are never relaxed: throttling the
            # early input of a join starves the fork, and with it the branch
            # the join is waiting for (a deadlock, not a slowdown).
            return max(int(peak), int(idle_floor))
        per_frame, t_up, _ = supply[tensor]
        # Slack relaxation: a producer whose chain finishes a frame in
        # t_up < global_period can be blocked for the difference, and draining
        # k tokens costs k * global_period / per_frame cycles.
        allowance = per_frame * (1.0 - min(t_up, global_period) / float(global_period))
        after_slack = int(round(peak - allowance))
        # Burst floor: the consumer's largest demand above the edge's rate line.
        floor = min(peak, burst_above_rate(read_times, per_frame / float(global_period)))
        # Throttled cap: bounds the floor when the consumer can be throttled,
        # the producer can be blocked (not at the pacer), and the blocking
        # budget is not marginal.
        if (
            not drives_pacer.get(consumer, True)
            and t_up < pacer_period
            and not (peak > 0 and allowance < CAP_MARGIN * peak)
        ):
            floor = min(floor, THROTTLED_CAP)
        return max(after_slack, floor)

    for node in nodes:
        tin, tout = tavs[node.name]
        dyn_inputs = [t for t in node.input if model.find_producer(t) is not None]
        is_join = len(dyn_inputs) > 1
        fill_only = is_join or node.name in in_branch
        transparent = tin is None or tout is None or node.op_type in _TRANSPARENT_OPS

        if transparent:
            # Forward the input timeline with zero latency: a join waits for its
            # latest input, a fork hands one schedule to every output.
            out_sched = None
            srcs = [arrival[t] for t in dyn_inputs if t in arrival]
            if srcs:
                n = min(len(s) for s in srcs)
                out_sched = np.max(np.stack([s[:n] for s in srcs]), axis=0)
            t_up, t_prop = chain_period(node)
            for t in dyn_inputs:
                if t in arrival and out_sched is not None:
                    peak = peak_occupancy(
                        arrival[t],
                        out_sched,
                        global_period,
                        supply.get(t, (0,))[0],
                        fill_only,
                    )
                    depths[t] = depth_from_peak(t, peak, out_sched, node.name, fill_only)
            for t in node.output:
                if out_sched is not None:
                    arrival[t] = out_sched
                    supply[t] = (max(1, len(out_sched) // 2), t_up, t_prop)
            continue

        in_curves, out_curves = curves[node.name]
        span = len(tin)
        cycles = np.arange(span, dtype=np.int64)

        # req[c]: wall-clock time by which every token read by local cycle c
        # has arrived, over all streamed inputs
        req = np.zeros(span, dtype=np.int64)
        in_scheds = {}
        for t in node.input:
            curve = in_curves.get(t)
            if curve is None:
                continue
            if t in arrival:
                sched = arrival[t]
            elif t in graph_inputs and model.find_producer(t) is None:
                # graph input: paced at one frame per global_period
                rate = global_period / max(int(curve[len(curve) // 2 - 1]), 1)
                sched = (np.arange(1, int(curve[-1]) + 1) * rate).astype(np.int64)
            else:
                continue  # weights / thresholds: no stream behind them
            if len(sched) == 0:
                continue  # no token crosses this edge in the window
            in_scheds[t] = (sched, curve)
            req = np.maximum(req, arrival_of(sched, curve))

        # the max-plus recurrence, as a running maximum of req(c) - c
        clock = cycles + np.maximum.accumulate(req - cycles)

        def token_times(curve):
            n = int(curve[-1])
            return clock[
                np.minimum(np.searchsorted(curve, np.arange(1, n + 1), side="left"), span - 1)
            ]

        for t, (sched, curve) in in_scheds.items():
            if model.find_producer(t) is None:
                continue  # graph input: sized by io_fifo_depth
            if int(curve[-1]) == 0:
                depths[t] = max(depths.get(t, 0), absorbed_frame_floor(node, t))
                continue
            read_times = token_times(curve)
            n_frame = int(curve[len(curve) // 2 - 1]) if len(curve) >= 2 else 0
            peak = peak_occupancy(sched, read_times, global_period, n_frame, fill_only)
            depths[t] = depth_from_peak(
                t,
                peak,
                read_times,
                node.name,
                fill_only,
                idle_floor=idle_window_floor(t, curve, node.name),
            )

        t_up, t_prop = chain_period(node)
        for t in node.output:
            curve = out_curves.get(t)
            if curve is None:
                continue
            wt = token_times(curve)
            arrival[t] = causal_writes(wt, in_scheds) if causal else wt
            per_frame_out = int(curve[len(curve) // 2 - 1]) if len(curve) >= 2 else int(curve[-1])
            supply[t] = (max(1, per_frame_out), t_up, t_prop)

    return depths


class DeriveFIFOSizes(Transformation):
    """Prerequisite: DeriveTokenAccessVectors already called on the graph.

    Sizes every edge with the chained-TAV pass and writes the depths to the
    ``inFIFODepths``/``outFIFODepths`` attributes of both endpoints.
    """

    def __init__(self, period=None, io_fifo_depth=5):
        super().__init__()
        self.period = period
        self.io_fifo_depth = io_fifo_depth
        self.minimum_size = 2

    def apply(self, model):
        # Two derivations of the same requirement, differing only in whether
        # a node's first-frame writes are held to the inputs that carry them
        # (``causal_writes``). Without it a producer's run-ahead survives; with
        # it a compacting node shows its real latency. Each is a lower bound, so
        # the larger is taken per edge.
        phased = derive_chained_tav_depths(model, self.period, causal=False)
        depths = derive_chained_tav_depths(model, self.period, causal=True)
        for tensor, depth in phased.items():
            if depth > depths.get(tensor, 0):
                depths[tensor] = depth

        # InsertFIFO takes max(producer.outFIFODepths, consumer.inFIFODepths),
        # so depths already on the nodes (e.g. from a folding config) would
        # override the sizer. Reset both, then write both.
        hw_nodes = [n for n in model.graph.node if is_hls_node(n) or is_rtl_node(n)]
        for node in hw_nodes:
            inst = registry.getCustomOp(node)
            inst.set_nodeattr("inFIFODepths", [self.minimum_size] * len(node.input))
            inst.set_nodeattr("outFIFODepths", [self.minimum_size] * len(node.output))

        graph_inputs = [x.name for x in model.graph.input]
        for node in hw_nodes:
            assert not node.op_type.startswith("StreamingFIFO"), "Found existing FIFOs"
            prod = registry.getCustomOp(node)
            out_fifo_depths = []
            for output_name in node.output:
                cons_node = model.find_consumer(output_name)
                if cons_node is None:
                    # graph output
                    out_fifo_depths.append(self.io_fifo_depth)
                    continue
                fifo_depth = max(depths.get(output_name, self.minimum_size), self.minimum_size)
                out_fifo_depths.append(fifo_depth)
                cons = registry.getCustomOp(cons_node)
                in_depths = cons.get_nodeattr("inFIFODepths")
                for i, inp in enumerate(cons_node.input):
                    if inp == output_name:
                        in_depths[i] = fifo_depth
                cons.set_nodeattr("inFIFODepths", in_depths)
            prod.set_nodeattr("outFIFODepths", out_fifo_depths)

            # graph inputs get at least io_fifo_depth
            in_fifo_depths = prod.get_nodeattr("inFIFODepths")
            for i, input_name in enumerate(node.input):
                if input_name in graph_inputs:
                    in_fifo_depths[i] = max(self.io_fifo_depth, in_fifo_depths[i])
            prod.set_nodeattr("inFIFODepths", in_fifo_depths)

        return (model, False)
