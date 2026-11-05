.. _fifo_sizing_heuristics:

********************************
Analytical FIFO sizing from TAVs
********************************

This page describes how ``step_set_fifo_depths`` sizes the FIFOs between
dataflow layers when ``auto_fifo_strategy`` is ``heuristic_analytical`` or
``heuristic_rtlsim``: the *chained-TAV* sizer ``DeriveFIFOSizes`` in
:py:mod:`finn.transformation.fpgadataflow.fifo_sizing.derive_fifo_sizes`.

In short
========

* ``largefifo_rtlsim`` (the default) simulates the whole accelerator with
  oversized FIFOs and trims every FIFO to the occupancy it reached. It needs
  every node synthesized and one long rtlsim of the stitched design.
* ``heuristic_rtlsim`` simulates every node on its own to get its schedule, then
  sizes the FIFOs analytically from those schedules. It needs every node
  synthesized, but no rtlsim of the full design.
* ``heuristic_analytical`` takes each node's schedule from the node's *tree
  model* and only simulates nodes that have none. In an estimate-only build it
  uses tree models only, so it needs no synthesis at all.

The heuristic strategies size for the steady state of a source that delivers
one frame per frame interval of the accelerator, not for a source that delivers
as fast as the stream allows. They are built to reach the estimated throughput
without deadlocking at joins; the ``fifo_sizing`` benchmark tests check that
(see `Checking the sizer`_). The analytical pass is a set of rules, each of
which trades FIFO depth for a stated assumption; where an assumption does not
hold, a FIFO is larger than it needs to be or, more rarely, too small for full
throughput. The rules are described below with the reasoning behind them; their
exact conditions are documented in the functions they name.

Token access vectors
====================

A **token access vector (TAV)** describes one node in isolation: for every
clock cycle of a frame, how many tokens (stream words, not bits) it has read on
each input stream and written on each output stream, cumulatively, assuming it
is never made to wait. Two back-to-back frames are stored, so the second frame
shows the steady-state phase after the first frame's wind-up. A TAV comes from
the node's tree model (registered for the node's op in
:py:mod:`finn.transformation.fpgadataflow.fifo_sizing.tree_models`) or from an
rtlsim of the node alone; ``DeriveTokenAccessVectors`` stores it in the
``io_chrc_in`` / ``io_chrc_out`` node attributes.

Half the TAV length is the node's **period**: the cycles one frame takes when
nothing stalls it. The slowest node is the **pacer**; its period is the
**global period**, the frame interval the whole graph settles to.

Forks (``DuplicateStreams``), joins (an elementwise binary operation with two
streamed inputs) and FIFOs carry no TAV, nor does a node without a tree model in
an estimate-only build; the sizer treats every node without a TAV as
transparent.

The chained pass
================

``derive_chained_tav_depths`` visits the nodes in topological order and turns
every node's local schedule into wall-clock time:

1. Every node with a TAV gets its period; the global period is the larger of the
   caller's estimate and the slowest TAV period. ``pacer_drivers`` marks every
   node that runs at the pacer's period or feeds one; the others can be
   *throttled*: they may finish a frame later without slowing the graph.
2. ``reconvergent_branch_nodes`` marks the nodes between a fork and the node
   its branches meet again at.
3. For each node, the wall-clock arrival time of every input token is known
   from its producer (or from the `paced graph input`_). ``local_clock`` maps
   the node's local cycles to wall-clock time with the max-plus recurrence

   .. math:: R(c) = \max(R(c-1) + 1,\; \text{arrival of the tokens read by cycle } c)

   so the node stalls exactly as long as its latest input is late. A transparent
   node forwards its input timeline unchanged (a join waits for its latest
   input, a fork hands one schedule to every output).
4. For each input edge, ``peak_occupancy`` is the largest number of tokens
   written but not yet read, with the two-frame schedule repeated every global
   period, and ``edge_depth`` turns that peak into the FIFO depth using the
   rules below.

``DeriveFIFOSizes`` raises the inputs of every join to the `Join window floor`_,
writes the depths into ``inFIFODepths`` / ``outFIFODepths`` (at least 2, and
``io_fifo_depth`` (5) on the graph input and output), and ``InsertFIFO`` creates
the FIFOs.

Without any rule, the depth of an edge is this peak occupancy. It gets two
things right that comparing a producer's and a consumer's TAVs directly cannot:
a producer whose own upstream lets it finish a frame early *does* run ahead and
the buffer must hold the surplus; and a node that looks fast in isolation but is
starved by its own producer never runs ahead and needs no buffer.

The rules
=========

The rules fall into two classes:

* **depth-minimizing** rules take depth away from the peak occupancy; removing
  one makes FIFOs deeper and can never cost throughput: the paced graph input,
  the two-frame split, the slack relaxation, the throttled cap and the
  decoupled chain.
* **throughput-protecting** rules put depth back where a minimizing rule would
  over-minimize, or where a TAV hides a latency; removing one makes FIFOs
  shallower and risks a stall: causal writes and phase lead, node latency, the
  burst, refill and idle-window floors, the pacer-bound chains, and the join
  rules.

The burst floor and the paced graph input have the largest effect on most
models; the join rules matter only for graphs with forks and joins.

.. _paced graph input:

Paced graph input
-----------------

*What:* the graph input delivers one frame per global period, evenly spaced
(``graph_input_schedule``).

*Why:* an infinitely fast source makes every FIFO above the pacer grow with the
number of frames simulated; a source matched to the accelerator's own rate is
the steady state the design has to sustain. ``largefifo_rtlsim`` sizes the first
FIFOs much deeper, because its testbench *is* a saturating source; the burst it
absorbs there buys nothing at steady state.

Two-frame split
---------------

*What:* the fill frame and the steady frame are evaluated separately and the
larger peak counts; joins and reconvergent branches take the fill frame alone
(``peak_occupancy``).

*Why:* the fill frame carries the pipeline-filling transient and the latency a
join has to buffer; the steady frame carries a producer's run-ahead. Repeating
both frames as one unit double-counts the transient and roughly doubles every
FIFO above the minimum.

Causal writes and phase lead
----------------------------

*What:* a compacting node's first-frame outputs are held until the inputs that
carry them have arrived (``causal_writes``), and output tokens a TAV writes
before its first read are counted to the previous frame (``phase_lead``).
``DeriveFIFOSizes`` runs the pass with and without these holds and keeps the
larger depth per edge.

*Why:* a TAV has an arbitrary phase, so ``first write - first read`` is not the
node's latency: a 64:1 width converter measures as writing one cycle after its
first read, and an ``MVAU_rtl`` emits a feature map's outputs during the next
feature map's window, so its TAV opens with writes whose inputs lie before the
window. Taken as they are, outputs would precede the inputs that carry them and
a branch of such nodes would look a feature map faster than it is. Neither pass
is an upper bound on its own (without the hold a producer's run-ahead survives,
with it a compacting node shows its real latency), so the larger counts.

.. _node latency:

Node latency
------------

*What:* every characterized node's writes are delayed by
``TAV_PIPELINE_LATENCY`` (32 cycles) plus its tree model's ``extra_latency()``
(``node_latency``), and in the causal pass held to the input blocks of its
``input_blocks()`` (``block_writes``, e.g. the whole matrix of a dynamic
``MVAU``). The pass also runs without latency, and the larger depth counts.

*Why:* a TAV records the steady state at an arbitrary phase, so no latency is in
it. In a chain that does not matter (producer and consumer move together), but
where two paths meet (a join, a residual shortcut, the two operands of a dynamic
``MVAU``) the slower path's latency is what the early input has to buffer. At
fast foldings it is hundreds of tokens. 32 cycles is deliberately above the
typical pipeline latency of a single node: an over-sized shortcut costs memory,
an under-sized one throughput.

Slack relaxation
----------------

*What:* a producer whose chain since the last pacer finishes a frame early gives
up the corresponding part of the peak (``edge_depth``). Nodes within 0.5 % of the
global period count as pacers (``PACER_TOLERANCE``), and past a pacer the chain
starts again.

*Why:* the peak occupancy is the depth at which the producer *never* waits,
which is more than throughput needs. A producer with slack can be blocked for
that slack every frame and still deliver on time. The tolerance treats a
bottleneck that is several nodes wide, with periods a cycle or two apart, as
one pacer, so the slack accounting restarts at the right place.

Pacer-bound chains
------------------

*What:* for a consumer that runs at the pacer's period or feeds a node that
does, the slack relaxation uses the slowest period of the whole chain feeding
the edge, without the restart past a pacer, so below a pacer the allowance is
(almost) zero (``edge_depth``, ``t_full``).

*Why:* a chain below a pacer is input-bound at the global period, however fast
its own nodes are: once blocked it cannot catch up, and with the shallow FIFOs
between it and the pacer the block reaches the pacer.

Burst floor
-----------

*What:* the depth never drops below the consumer's largest demand above the
edge's average rate (``burst_above_rate``).

*Why:* the slack relaxation is average-rate arithmetic. A consumer that reads
its frame in bursts (an MVAU reads ``SF`` words at the start of each output
window, a sliding window generator reads rows) needs the burst buffered even
when the average rate is matched; otherwise it stalls at every burst. This rule
sizes most of the storage, and removing it costs the most throughput. Its known
over-size: an MVAU at the pacer that reads its whole input in one burst right
behind the graph input gets that burst buffered, although a saturating source
would serve it directly.

Refill floor
------------

*What:* for a consumer that runs at the pacer's period or feeds one, the depth
never drops below what it reads during a pause in the producer's writes, beyond
what its own slack and the full FIFOs above the producer cover
(``refill_floor``, ``upstream_cover``).

*Why:* the slack relaxation gives up the producer's run-ahead, so the producer
delivers just in time. Where its writes pause (a sliding window generator
upstream re-reading rows at a frame boundary), the FIFO alone has to feed the
consumer through the pause; the burst floor, an average-rate argument, does not
see it.

.. _Throttled cap:

Throttled cap
-------------

*What:* the burst floor is capped at ``THROTTLED_CAP`` (256, the depth up to
which a FIFO is a shift register) when the consumer can be throttled, the
producer can be blocked, its blocking allowance exceeds the peak by
``CAP_MARGIN`` (1.25), and the slow stages on both sides of the edge take at
most ``CAP_MARGIN`` periods together (``edge_depth``).

*Why:* a throttleable consumer needs only short-timescale smoothing; it never
has to hold a whole frame of the producer running ahead, because it may finish
its frame later. The margin keeps the cap off edges whose allowance only just
covers the peak, where the first-order estimate cannot be trusted. With slow
stages on both sides, the cap and the slack relaxation further down the chain
would spend the same blocking budget twice, and the stages would stall each
other.

Decoupled chain
---------------

*What:* no burst floor at all for an edge into a throttleable consumer when a
slow node (at least 20 % of the period) sits between the pacer and the producer,
and it and the slowest node at or below the consumer take at most
``CAP_MARGIN`` periods together (``edge_depth``, ``t_above`` / ``t_below``).

*Why:* on a chain of fast nodes between slow ones the burst floor puts the peak
back at every edge: each fast producer runs a row ahead of the slow node below
it, and the same backlog shows up again at every edge down the chain. With
shallow FIFOs the two slow stages are coupled, but mostly they still overlap,
and a slow stage above the producer means the producer is not fed just in time
by the pacer, so blocking it does not stall the pacer at once. The floor stays
where it is needed: right below the pacer, and between two slow stages that
would stall each other.

Joins and reconvergent branches
-------------------------------

*What:* an edge into a join, or into any node between a fork and its join, is
sized to its fill-frame peak and never relaxed. Inside a branch the depth also
covers what arrives while the consumer is not reading (``idle_window_floor``).
Forks and joins have no period, so they are never pacers.

*Why:* every relaxation assumes the consumer can simply be made to wait. The
early input of a join cannot be throttled without throttling the fork it came
from, which starves the other branch, which is what the join is waiting for:
a shortcut FIFO shorter than the long branch's occupancy deadlocks the design.
For the same reason a node inside a branch cannot be made to wait while it is
not reading (``FMPadding`` emitting padding rows, an MVAU between its read
bursts).

Join window floor
-----------------

*What:* every input of a two-stream join and of a ``StreamingConcat`` gets a
floor that does not come from any TAV (``join_window_floors``): walking back to
the fork, each node's ``input_window()`` / ``input_need()`` hint from its tree
model gives how many fork words its outputs need, and the floor is the larger of
the deadlock backlog and the paced backlog, plus ``JOIN_FLOOR_MARGIN`` (25 %).

*Why:* a fork writes all its outputs together, so it stalls as soon as one
branch is full. If the long branch reads many fork words before its first
output reaches the join (a 48:1 width converter needs 48 words a word, an MVAU a
whole vector, a sliding window a row and a pixel more than its output position),
the shortcut has to buffer all of them, or the fork stalls and the graph
deadlocks. For full throughput it also has to buffer what the fork writes while
the long branch's slowest node emits its block. The timeline pass misses this
where a TAV writes before it reads and the fork is transparent. The margin
covers a node that reads ahead of its window or an upstream that delivers in
bursts.

Absorbed-frame floor
--------------------

*What:* a consumer whose TAV reads nothing in the stored window gets one folded
input frame as the depth of that edge (``absorbed_frame_floor``).

*Why:* a node that absorbs its whole input frame before releasing anything (a
``FINNLoop``) can be characterized over a window that misses the absorb phase;
there is then no schedule to compare against, and the only safe answer is the
frame itself.

Checking the sizer
==================

Two sets of tests keep the sizer honest, and are the place to measure a change
to a rule:

* the tree-model tests (``pytest -m node_tree_modeling``) compare every tree
  model against the rtlsim of its node, so a schedule that drifts after a
  finn-hlslib or finn-rtllib change fails there;
* the ``fifo_sizing`` tests of the benchmark suite (``pytest tests/benchmark -m
  fifo_sizing``) build the finn-examples models with both heuristic strategies
  up to stitched IP and require the rtlsim output to match and the throughput
  to reach 98 % of the estimate. Each build writes ``report/feature_metrics.json``
  with the estimated and simulated throughput and the total FIFO storage, which
  is what a rule change should be compared on, against ``largefifo_rtlsim`` as
  the reference (see "FIFO sizing tests" in ``tests/benchmark/README.md``).
