.. _fifo_sizing_heuristics:

********************************
Analytical FIFO sizing from TAVs
********************************

This page describes how ``step_set_fifo_depths`` sizes the FIFOs between
dataflow layers when ``auto_fifo_strategy`` is ``heuristic_analytical`` or
``heuristic_rtlsim``: the *chained-TAV* sizer in
:py:mod:`finn.transformation.fpgadataflow.derive_characteristic`. The core of
the sizer is a schedule simulation; everything on top of it is a rule that
trades depth for a stated assumption. Each rule is listed below with a measured
example from a finn-examples model, so that it can be judged (and removed) on
evidence.

The reference every number is compared with is ``largefifo_rtlsim``: the
accelerator is simulated with oversized FIFOs and every FIFO is trimmed to the
occupancy it reached. All measurements on this page are ZCU104 builds of the
committed folding configurations, run through
``tests/benchmark/test_fifo_sizing_flow.py`` (stitched-IP rtlsim, one frame).

Token access vectors
====================

A **token access vector (TAV)** describes one node in isolation: for every
clock cycle of a frame, how many tokens (stream words, not bits) it has read on
each input stream and written on each output stream, cumulatively, assuming it
is never made to wait. Two back-to-back frames are stored, so the second frame
shows the steady-state phase after the first frame's wind-up. A TAV comes from
the node's *tree model* (``get_tree_model``, a run-length encoded schedule, used
by ``heuristic_analytical``) or from an rtlsim of the node alone
(``heuristic_rtlsim``); ``DeriveTokenAccessVectors`` stores it in the
``io_chrc_in`` / ``io_chrc_out`` node attributes and nothing modifies it
afterwards.

Half the TAV length is the node's **period**: the cycles one frame takes when
nothing stalls it. The slowest node is the **pacer**; its period is the
**global period**, the frame interval the whole graph settles to.

Forks (``DuplicateStreams``), joins (``ElementwiseAdd`` with two streamed
inputs, ``AddStreams``) and FIFOs carry no TAV; the sizer treats every node
without a TAV as transparent (see below). An earlier rule also made every
``ElementwiseAdd`` transparent by op type, including a unary one with a
constant operand that does carry a TAV; it was removed because it changes no
depth on any regression model (resnet50 has two such nodes).

The chained pass, step by step
==============================

``derive_chained_tav_depths`` visits the nodes in topological order and turns
every node's local schedule into wall-clock time:

1. ``read_node_tavs`` reads every TAV once and binds its rows to the node's
   stream tensors.
2. ``node_periods`` gives every node with a TAV its period; forks and joins
   have none (see `Fork and join periods`_).
3. The global period is the larger of the caller's estimate and the slowest
   TAV period; ``pacer_drivers`` marks every node that runs at the pacer's
   period or feeds one (the others can be *throttled*: they may finish a frame
   later without slowing the graph).
4. ``reconvergent_branch_nodes`` marks the nodes between a fork and the node
   its branches meet again at.
5. For each node, the wall-clock arrival time of every input token is known
   from its producer (or from the `paced graph input`_). ``local_clock`` maps
   the node's local cycles to wall-clock time with the max-plus recurrence

   .. math:: R(c) = \max(R(c-1) + 1,\; \text{arrival of the tokens read by cycle } c)

   so the node stalls exactly as long as its latest input is late, and
   ``token_times`` places every read and write of the node on the wall clock.
   A transparent node forwards its input timeline unchanged (a join waits for
   its latest input, a fork hands one schedule to every output).
6. For each input edge, ``peak_occupancy`` is the largest number of tokens
   written but not yet read, with the two-frame schedule repeated every global
   period (``periodic_peak``), and ``edge_depth`` turns that peak into the
   FIFO depth using the rules below.

``DeriveFIFOSizes`` writes the result into ``inFIFODepths`` /
``outFIFODepths`` with a minimum of 2 and ``io_fifo_depth`` (5) on the graph
input and output; ``InsertFIFO`` then creates the FIFOs.

The core: peak occupancy on the chained schedule
=================================================

Without any rule, the depth of an edge is the peak occupancy on the chained
schedule. Two things it gets right that comparing the producer's and the
consumer's TAVs directly cannot: a producer whose own upstream lets it finish a
frame early *does* run ahead and the buffer must hold the surplus; and a node
that looks fast in isolation but is starved by its own producer never runs
ahead and needs no buffer (its tokens never get an early arrival time).

On cnv-w1a1 every FIFO deeper than the minimum is exactly this peak; none of
the rules below changes it. The three largest are ``StreamingDataWidthConverter_rtl_0
-> ConvolutionInputGenerator_rtl_0`` (60), ``StreamingDataWidthConverter_rtl_8 ->
ConvolutionInputGenerator_rtl_5`` (52) and ``Pool_hls_1 ->
ConvolutionInputGenerator_rtl_6`` (72); the sized accelerator reaches 12255
frames/s in rtlsim, the same as the reference, with 11.9 kbit of FIFO storage
against the reference's 22.3 kbit.

The rules
=========

.. _paced graph input:

Paced graph input
-----------------

*What:* ``graph_input_schedule`` delivers the graph input at one frame per
global period, evenly spaced, instead of as fast as the stream allows.

*Why:* an infinitely fast source makes every FIFO above the pacer grow with the
number of frames simulated; a source matched to the accelerator's own rate is
the steady state the design has to sustain.

*Example:* cnv-w1a1, ``StreamingDataWidthConverter_rtl_0 ->
ConvolutionInputGenerator_rtl_0`` (24-bit words). With the input released at
time 0 the depth is 638; paced it is 60. rtlsim throughput is 12255 frames/s
in both cases; total FIFO storage 25.5 kbit against 11.9 kbit. The
``largefifo_rtlsim`` reference sizes this FIFO to 617, because its testbench
*is* a saturating source; the burst it absorbs there buys nothing at steady
state.

Causal writes
-------------

*What:* ``causal_writes`` holds output token *j* of a node's first frame until
input ``ceil((j+1) * N_in / N_out)`` has arrived, for compacting nodes
(``N_in > N_out``).

*Why:* a TAV has an arbitrary phase, so ``first write - first read`` is not the
node's latency: a 64:1 width converter measures as writing one cycle after its
first read, which would let the consumer's schedule start before the tokens it
needs can exist.

``DeriveFIFOSizes`` runs the pass twice, with and without the rule, and keeps
the larger depth per edge: each is a lower bound (without the rule a
producer's run-ahead survives, with it a compacting node shows its real
latency).

*Example:* on the linear models the causal pass is the larger one, by one
token on one edge: ``Pool_hls_1 -> ConvolutionInputGenerator_rtl_6`` in
cnv-w1a1 (72 against 71), ``Pool_hls_0 -> ConvolutionInputGenerator_rtl_3`` in
gtsrb (59 / 58), ``StreamingDataWidthConverter_rtl_2 -> FMPadding_rtl_0`` in
mobilenet_v1 (1824 / 1823). On resnet50 the pass *without* the rule is the
larger one on seven edges, among them the residual shortcuts
``Thresholding_rtl_44 -> ElementwiseAdd_hls_7`` (14106 against 11870) and
``StreamingDataWidthConverter_rtl_30 -> ElementwiseAdd_hls_3`` (13979 /
12094) and ``StreamingDataWidthConverter_rtl_28 -> MVAU_hls_12`` (49 / 4);
total depth 362655 against 355475. Whether those seven depths are needed has
not been measured (no resnet50 rtlsim is available here), which is why both
passes are kept.

Two-frame split
---------------

*What:* ``peak_occupancy`` evaluates the first (fill) frame and the second
(steady) frame of the two-frame schedules separately and takes the larger;
joins and reconvergent branches take the fill frame alone.

*Why:* the fill frame carries the pipeline-filling transient and the latency a
join has to buffer; the steady frame carries a producer's run-ahead. Repeating
both frames as one unit double-counts the transient.

*Example:* cnv-w1a1 without the split: ``StreamingDataWidthConverter_rtl_8 ->
ConvolutionInputGenerator_rtl_5`` 104 instead of 52, ``Pool_hls_1 ->
ConvolutionInputGenerator_rtl_6`` 142 instead of 72, every other non-minimal
FIFO doubled (total depth 416 against 271, 18.4 kbit against 11.9 kbit).
rtlsim throughput 12255 frames/s either way.

Pacer tolerance
---------------

*What:* ``PACER_TOLERANCE = 0.995``: a node within 0.5 % of the global period
counts as the pacer.

*Why:* the slowest stage of a graph is usually several nodes wide and their
periods differ by a cycle or two; an exact comparison picks one of them and
the slack accounting (next two rules) restarts at the wrong place.

*Example:* vgg10-radioml has five ``ConvolutionInputGenerator -> MVAU`` edges
whose consumers are 1-2 cycles short of the slowest node; with an exact test
they are no longer pacers, their input FIFOs go from 2 to 3 and the total depth
from 131 to 136. On mobilenet_v1 the exact test moves the pacer and with it
the slack reset and the throttled cap: the total depth goes from 9432 to
64586 (see `Throttled cap`_).

Slack relaxation
----------------

*What:* for an edge whose producer chain (the nodes since the last pacer)
finishes a frame in ``t_up < global period`` cycles, ``edge_depth`` subtracts
``per_frame * (1 - t_up / global period)`` tokens from the peak. Past a pacer
the chain period is reset (``t_prop = 0``): an edge leaving the pacer has no
slack, an edge two hops down is limited only by the segment since the pacer.

*Why:* the peak occupancy is the depth at which the producer *never* waits,
which is more than throughput needs. A producer with that much slack can be
blocked for ``global period - t_up`` cycles per frame and still deliver on
time; draining ``k`` tokens costs ``k * global period / per_frame`` cycles.

*Example:* gtsrb, ``Pool_hls_0 -> ConvolutionInputGenerator_rtl_3``: peak 176,
relaxed to 59. cnv-w2a2, ``StreamingDataWidthConverter_rtl_6 ->
ConvolutionInputGenerator_rtl_4``: 1152 relaxed to 594 (total depth 1974 to
1405). Without the reset past the pacer the mobilenet_v1 total goes from 9432
to 64587. rtlsim of gtsrb with the relaxation switched off: 8134 frames/s
either way, 10.8 kbit against 9.9 kbit.

Burst floor
-----------

*What:* the depth never drops below ``min(peak, burst_above_rate(reads,
per_frame / global period))``: the consumer's largest demand above the edge's
average rate line.

*Why:* the slack relaxation is average-rate arithmetic. A consumer that reads
its frame in bursts (an MVAU reads ``SF`` words at the start of each output
window, a sliding window generator reads rows) needs the burst buffered even
when the average rate is matched; otherwise it stalls at every burst.

*Example:* cnv-w1a1 without the floor: ``StreamingDataWidthConverter_rtl_0 ->
ConvolutionInputGenerator_rtl_0`` 2 instead of 60, ``..._rtl_8 ->
ConvolutionInputGenerator_rtl_5`` 28 instead of 52, ``Pool_hls_1 ->
ConvolutionInputGenerator_rtl_6`` 66 instead of 72. rtlsim throughput drops
from 12255 to 12077 frames/s (-1.5 %) for 9.7 instead of 11.9 kbit. This is
the rule that sizes most of the storage: without it the regression set's
totals fall to 176/271 (cnv-w1a1), 882/1405 (cnv-w2a2), 521/1298 (gtsrb),
26/452 (kws) and 3003/9432 (mobilenet_v1).

*Known over-size:* kws, ``Thresholding_rtl_0 -> MVAU_hls_0``: depth 428 (peak
428, burst 428; the MVAU is the pacer and reads its 490 input words in one
burst per frame at 1/8 of the paced input rate). rtlsim gives 25471 frames/s
with 428 and with 2, and the reference sizes it to 2 (0.9 kbit against 4.3
kbit): the testbench's source saturates the 490-cycle Thresholding node, so
the burst is served from the source itself. Under a rate-matched source the
MVAU would stall for most of its first output window without the buffer. The
sizer keeps the conservative answer; a build that knows its source bursts can
drop it.

.. _Throttled cap:

Throttled cap
-------------

*What:* ``THROTTLED_CAP = 256`` (the depth up to which a FIFO is a shift
register): the burst floor is capped at 256 when the consumer can be throttled
(neither it nor anything it feeds runs at the pacer's period) and the producer
can be blocked (its chain is not at the pacer).

*Why:* a throttleable consumer needs only short-timescale smoothing; it never
has to hold a whole frame of the producer running ahead, because it is allowed
to finish its frame later.

*Example:* mobilenet_v1 (ZCU104 folding, 141 edges): the cap holds 22 edges at
256, among them ``StreamingDataWidthConverter_rtl_50 ->
ConvolutionInputGenerator_rtl_13`` (7656 without), ``..._rtl_44 ->
ConvolutionInputGenerator_rtl_11`` (2465) and eleven ``Thresholding ->
FMPadding`` edges (481-1009); the total depth is 9432 with the cap and 64369
without. No other regression model has a throttleable consumer with a burst
above 256. mobilenet_v1's rtlsim throughput with and without the cap has not
been measured (its ZCU104 stitched-IP rtlsim is not available in the
benchmark), so the cap is the one rule on this page without a throughput
example.

The cap only fires when the producer's blocking allowance exceeds the peak by
``CAP_MARGIN`` (1.25): the allowance is first-order average-rate arithmetic,
and an edge whose allowance only just covers its peak is one the estimate
cannot be trusted about. The margin changes one depth on the regression set,
``Pool_hls_1 -> MVAU_rtl_0`` in resnet50 (272 with the margin, 256 without),
on no other model.

Joins and reconvergent branches
-------------------------------

*What:* an edge into a join, or into any node between a fork and its join, is
sized to its fill-frame peak and never relaxed.

*Why:* every relaxation assumes the consumer can simply be made to wait. The
early input of a join cannot be throttled without throttling the fork it came
from, which starves the other branch, which is what the join is waiting for:
a shortcut FIFO shorter than the long branch's occupancy deadlocks the design.

*Example:* resnet50 (V80 folding, 398 edges, 16 fork/join pairs) is the only
regression model with joins. Relaxing join and branch edges like chain edges
takes the total depth from 362655 to 54433: the shortcut
``StreamingDataWidthConverter_rtl_49 -> ElementwiseAdd_hls_6`` from 17405 to
59, ``..._rtl_38 -> MVAU_hls_16`` (inside a branch) from 414 to 17, the
``StreamingDataWidthConverter -> FMPadding`` edges inside branches from
1666-1760 to 2. These shortcuts hold a whole frame while the long branch
computes it; a shallower one deadlocks. Taking both frames instead of the fill
frame alone (`Two-frame split`_) would instead *grow* the branch edges
(``FMPadding_rtl_15 -> ConvolutionInputGenerator_rtl_18`` 1791 to 3578, total
378030). Neither variant has been simulated (no resnet50 rtlsim here).

Idle-window floor
-----------------

*What:* for an edge into a node inside a reconvergent branch,
``idle_window_floor`` is a second floor: the frame's tokens times the fraction
of the consumer's period it spends outside its read window (first to last
read of that stream); capped at ``THROTTLED_CAP`` when the consumer can be
throttled. Chain edges do not use it.

*Why:* inside a branch the consumer cannot be made to wait (its deadline is the
join), so whatever the producer delivers while the consumer is not reading has
to be buffered. ``FMPadding`` reads nothing while it emits padding rows; an
MVAU reads its inputs in a short burst per output window.

*Example:* resnet50 without the floor: ``StreamingDataWidthConverter_rtl_123
-> FMPadding_rtl_15`` 2 instead of 3332, ``..._rtl_7 -> FMPadding_rtl_2`` 2
instead of 1760, ``..._rtl_99 -> MVAU_hls_37`` 27 instead of 32; total depth
330410 instead of 362655. Not simulated (see above); kept because the rule
only ever adds depth where a stall would reach the join.

Fork and join periods
---------------------

Forks and joins have no TAV and therefore no period: ``pacer_drivers`` never
counts them as pacers and ``chain_period`` carries the upstream chain period
through them. An earlier rule copied a neighbour's TAV length onto each
fork/join pair (``inherited_fork_join_tav_lengths``, a leftover of the removed
``HandleBranches`` pass); it was removed because it changes no depth on
resnet50, the only regression model with forks and joins.

Absorbed-frame floor
--------------------

*What:* a consumer whose TAV reads nothing in the stored window gets one folded
input frame as the depth of that edge (``absorbed_frame_floor``).

*Why:* a node that absorbs its whole input frame before releasing anything (a
``FINNLoop`` body) can be characterized over a window that misses the absorb
phase; there is then no schedule to compare against, and the only safe answer
is the frame itself.

*Example:* not exercised by any regression model (no loop bodies); kept as the
guard for the MLO flow.

Minimum depth and graph I/O
---------------------------

Every FIFO is at least 2 deep and the graph input and output FIFOs at least
``io_fifo_depth`` (5). This, not a heuristic, is where the heuristic sizer
uses *more* storage than the reference on the MLPs: tfc-w1a1 18.3 kbit
against 9.1 kbit and cybersecurity-mlp 3.7 against 1.9 kbit are entirely the
3136-bit and 600-bit input FIFOs at depth 5 instead of 2; every internal FIFO
is 2 in both.

Comparison with the simulated reference
=======================================

Committed foldings, ZCU104, stitched-IP rtlsim, one frame:

=================  ==============  ===============  ===============  ===============
model              reference fps   heuristic fps    reference kbit   heuristic kbit
=================  ==============  ===============  ===============  ===============
tfc-w1a1           12.5 M          12.5 M           9.1              18.3
cnv-w1a1           12255           12255            22.3             11.9
cybersecurity-mlp  16.7 M          16.7 M           1.9              3.7
kws                25471           25471            0.9              4.3
gtsrb              5315            8134             24.0             9.9
=================  ==============  ===============  ===============  ===============

(The MLP throughputs are the one-frame testbench's interval, not a sustained
rate.) Where the heuristic is larger it is the graph-input FIFO (tfc,
cybersecurity-mlp) or the burst floor under the paced-source assumption (kws);
where it is smaller (cnv-w1a1, gtsrb) the reference measured the saturating
testbench's burst into the first FIFO (617 on ``StreamingDataWidthConverter_rtl_0
-> ConvolutionInputGenerator_rtl_0`` in both models). On gtsrb the heuristic
sizing is also *faster* than the reference: the one-frame reference run trims
``StreamingDataWidthConverter_rtl_13 -> MVAU_hls_8`` to 2 and
``StreamingDataWidthConverter_rtl_7 -> ConvolutionInputGenerator_rtl_5`` to 2
where the heuristic keeps 215 and 151 (burst floors of bursty consumers), and
the trimmed design stalls at 5315 frames/s against 8134.

Regression set
==============

The sizer is regressed on characterized fixtures of tfc-w1a1, tfc-w2a2,
cnv-w1a1, cnv-w2a2, cybersecurity-mlp, gtsrb, kws, vgg10-radioml and
mobilenet_v1 (ZCU104) and resnet50 (the only model with forks and joins); a
change to the pass has to give identical depths on all of them or come with
rtlsim evidence. ``tests/fpgadataflow/test_fifosizing_chained_tav.py`` pins
the depths of small synthetic graphs and the arithmetic helpers without
synthesis; ``tests/benchmark/test_fifo_sizing_flow.py`` compares each strategy's
rtlsim throughput with the reference (``FEATURE_FLOW_QUICK=1`` for depths only).
