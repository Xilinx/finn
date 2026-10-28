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
committed folding configurations with Vivado 2024.2, run through
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

``DeriveFIFOSizes`` raises the inputs of every join to the `Join window floor`_
and writes the result into ``inFIFODepths`` /
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

``DeriveFIFOSizes`` runs the pass with and without the rule (and each with
and without `node latency`_) and keeps the largest depth per edge: each is a
lower bound (without the rule a producer's run-ahead survives, with it a
compacting node shows its real latency).

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

Phase lead
----------

*What:* in the causal pass, the output tokens a node's TAV writes before its
first read (``phase_lead``) are taken to belong to the previous frame: the
node's output tokens are renumbered past them, in every frame. Not applied to
nodes that write more words than they read per frame, which may legitimately
write first (``FMPadding`` emits its padding rows before it reads).

*Why:* the stored window can start inside a frame. An ``MVAU_rtl`` emits a
feature map's ``MH / PE`` outputs during the next feature map's window, after it
prefetched the inputs, so its TAV opens with those writes and its read burst
wraps to the end of the window. Taken as they are, the outputs precede the
inputs that carry them and every ``MVAU_rtl`` in a branch looks one feature map
faster than it is; ``causal_writes`` alone only holds the first frame of
compacting nodes, and to the next input token, not the next feature map.

*Example:* transformer_benchmark on ZCU104, committed folding (pacer
``MVAU_rtl`` MW = MH = 32, SIMD = PE = 1, 65536 cycles). The residual shortcut
``Thresholding_rtl_1 -> ElementwiseAdd_hls_0`` waits for the attention branch
(phase lead 31 on three of its ``MVAU_rtl``; the fourth, ``Q K^T``, writes
more words than it reads and is left alone) and was sized 1985;
rtlsim reaches 65536 cycles per frame with it at 2077 or more, 67415 at 2048
and 2064, 67895 at 1985-2020. With the rule it is 2077. On resnet50 (V80) the
rule changes one edge of the causal pass by 10 tokens and no final depth.

.. _node latency:

Node latency
------------

*What:* every characterized node's write times are delayed by
``TAV_PIPELINE_LATENCY`` (32 cycles) plus ``tav_extra_latency()`` of the node
(``node_latency``; ``InnerShuffle``: one page, ``I * J / SIMD`` cycles, since no
word of a page leaves before the whole page is in its bank). In the causal pass
a node's writes are also held to the input blocks of
``tav_input_blocks()`` (``block_writes``; a dynamic ``MVAU`` computes on the
whole matrix ``dynamic_load`` buffered from in1, while its TAV reads in1 alongside
the compute); held words leave at most one per cycle. ``DeriveFIFOSizes`` keeps
the passes without latency too, so no edge gets smaller than without the rule.

*Why:* a TAV records the steady state at an arbitrary phase, so no latency is in
it. In a chain that does not matter (producer and consumer move together), but
where two paths meet (a join, a residual shortcut, the two operands of a dynamic
``MVAU``) the slower path's latency is what the early input has to buffer. At
slow foldings it is negligible against the period; at fast ones it is hundreds
of tokens. 32 cycles is deliberately above the typical pipeline latency
(single-node rtlsim of transformer_benchmark, first output against the tree
model: width converters 1, ``HWSoftmax`` 7, ``ElementwiseMul`` 11, ``StreamingSplit``
16, ``Thresholding_rtl`` 18, ``MVAU_rtl`` 4 to 69): an over-sized shortcut costs
memory, an under-sized one throughput.

*Example:* transformer_benchmark on ZCU104, folding ``maximize`` (1280 cycles a
frame). The residual shortcut ``Thresholding_rtl_1 -> ElementwiseAdd_hls_0`` was
sized 1232 (1755 cycles a frame in rtlsim); it needs 1501 to 1550 for 1280.
With the rule it is 2004 (pipeline latency alone 1769, plus the dynamic
``MVAU`` hold 1799, plus the ``InnerShuffle`` page 2004). The feature test
reaches 1280 cycles a frame in rtlsim, with 246 kbit of FIFO storage against
210. The committed folding (65536 cycles) moves
from 2077 (needs 2065 to 2077) to 2140, 91.7 kbit against 89.7. On the linear
models (bnn-pynq, cybersecurity-mlp, gtsrb, kws, mobilenet_v1, vgg10-radioml,
espcn) no depth changes; total FIFO storage grows by 0.1 to 0.7 % on resnet50,
resnet18 and yolov8, 1.2 % on bert, 7 to 16 % on transformer_vision, _language
and _radioml (offline re-sizing of the same TAVs). The sizing pass takes twice
as long (four passes instead of two; 210 s against 108 s on resnet50).

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
The reset only applies to consumers that can be throttled, see `Pacer-bound
chains`_.

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
from 12255 to 12077 frames/s (-1.5 %) for 9.7 instead of 11.9 kbit. On
cnv-w2a2 it drops from 3051 to 1644 frames/s (-46 %, 22.2 instead of 25.5
kbit) and on gtsrb from 8134 to 6751 (-17 %, 7.5 instead of 9.9 kbit). This
is the rule that sizes most of the storage: without it the regression set's
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

Pacer-bound chains
------------------

*What:* for a consumer that runs at the pacer's period or feeds a node that
does, the slack relaxation uses ``t_full``, the slowest period of the whole
chain feeding the edge, without the reset past a pacer. Below a pacer the
allowance is therefore (almost) zero for these consumers.

*Why:* a chain below a pacer is input-bound at the global period, however fast
its own nodes are: once blocked it cannot catch up, and with the shallow FIFOs
between it and the pacer the block reaches the pacer. Its run-ahead is what
carries a consumer that cannot be throttled through the producer's gaps (a
sliding window generator refilling rows at the frame boundary).

*Example:* cnv-w2a2, optimizer folding in maximize mode on AUP-ZU3 (pacers
``MVAU_hls_2`` and ``MVAU_hls_4``, 27655 cycles; ``MVAU_hls_3`` 25607).
``StreamingDataWidthConverter_rtl_4 -> MVAU_hls_3``: the chain since
``MVAU_hls_2`` measures 14690 cycles, so the allowance (187 tokens) exceeded
the peak (48) and the burst floor (32) set the depth. Every frame
``ConvolutionInputGenerator_rtl_4`` waits for 26 new pixels from
``MVAU_hls_2`` before its first window, longer than ``MVAU_hls_3``'s slack.
rtlsim interval 28226 cycles (-2.0 %); with 48 (the peak, now the depth) it is
27655. ``StreamingDataWidthConverter_rtl_6 -> MVAU_hls_4`` goes from 41 to 49.
Of the committed foldings only vgg10-radioml changes (four FIFOs 2 to 3, 14.2
to 15.7 kbit).

Refill floor
------------

*What:* for a consumer that runs at the pacer's period or feeds a node that
does, the depth never drops below ``min(peak, refill_floor(...))``: the most
tokens the consumer reads during one gap in the producer's writes, not
counting the first cycles of the gap that are covered otherwise: the
consumer's own slack (``global period - consumer period``) and the supply
held in the FIFOs above the producer (``upstream_cover``: ``depth * global
period / per_frame`` per FIFO, back along the single-input chain to a pacer,
a join or the graph input; they are full while the producer is blocked).

*Why:* the slack relaxation gives up the producer's run-ahead, so the producer
delivers just in time. Where its writes pause (a sliding window generator
upstream re-reading ``(K - 1)`` rows at a frame boundary), the FIFO alone has
to feed the consumer through the pause; the burst floor, an average-rate
argument, does not see it.

*Example:* cnv-w1a2, optimizer folding in maximize mode on KV260 (pacer
``MVAU_hls_1``, 12551 cycles; ``MVAU_hls_0`` 10807).
``StreamingDataWidthConverter_rtl_3 -> MVAU_hls_1``: peak 404, allowance 436,
depth 4. The producer's writes pause for 824 cycles per frame while
``ConvolutionInputGenerator_rtl_1`` waits for 2 rows and 3 pixels from
``MVAU_hls_0``; the pacer reads 208 tokens meanwhile. rtlsim interval 13262
cycles (-5.4 %) at depth 4, 12881 at 100 and 12551 (the estimate) at 190, 260
and 404; the floor gives 184, also 12551. The same edge pattern (sliding
window generator -> width converter -> MVAU) in other maximize-mode foldings:
gtsrb (16 to 65) 12992 to 12806 cycles (estimate 12807), cnv-w1a1 (2 to 56)
14457 to 14406 (14407); cnv-w2a2 in target mode (7 to 225) 33320 to 33206. No
committed folding changes.

*Cover:* mobilenet_v1, ``MVAU_hls_1 -> StreamingDataWidthConverter_rtl_4``
(a pacer): the producer's writes pause for 10750 cycles, but the 202-deep FIFO
in front of ``ConvolutionInputGenerator_rtl_1`` holds 12474 cycles of supply,
so the depth stays 2 (rtlsim reaches the estimate with 2; without the cover
the floor gives 677, +116 kbit). Without the cover one committed cnv-w1a2 /
cnv-w2a2 edge would also grow (594 to 608), and the foldings above would get
208, 71, 58 and 242 (same throughput).

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

Where the chain is decoupled (`Decoupled chain`_) the floor is dropped
altogether. The cap only fires when the producer's blocking allowance exceeds
the peak by ``CAP_MARGIN`` (1.25): the allowance is first-order average-rate arithmetic,
and an edge whose allowance only just covers its peak is one the estimate
cannot be trusted about. The margin changes one depth on the regression set,
``Pool_hls_1 -> MVAU_rtl_0`` in resnet50 (272 with the margin, 256 without),
on no other model.

Throttled cap between slow stages
---------------------------------

*What:* the throttled cap only fires where the slow stages around the edge can
take turns: the slowest node between the last pacer and the producer and the
slowest node at or below the consumer (``t_above`` and ``t_below`` of
`Decoupled chain`_) take at most ``CAP_MARGIN`` periods together. Otherwise the
edge keeps its burst floor.

*Why:* the cap blocks the producer during the consumer's bursts and spends the
chain's blocking budget (its slack below the period) on it. The slack relaxation
of the edges further down the same chain spends the same budget again, and with
slow stages on both sides of the edge nothing is left to absorb the stall: the
stages stall each other and the pacer loses time.

*Example:* cnv-w2a2, small folding (AUP-ZU3 and U55C, pacer ``MVAU_hls_0`` at
129606 cycles). ``Pool_hls_0 -> ConvolutionInputGenerator_rtl_3`` (peak 307,
burst floor 307, allowance 1533) was capped at 256, with 0.88 of the period
above it and 0.91 below. The edge two nodes further down, ``MVAU_hls_2 ->
ConvolutionInputGenerator_rtl_4``, was relaxed by the chain's whole allowance
(3968 to 1715). Stitched-IP rtlsim: 132298 cycles a frame, 0.980 of the estimate
with both ``heuristic_analytical`` and ``heuristic_rtlsim``. Unbounded FIFOs, or
only the capped edge at 280 or 307: 129606, the estimate (the relaxed edge alone
needs 2400). With the rule the edge is 307 and both strategies run at 1.000 on
both boards (23.5 kbit). cnv-w1a2's small folding has the same edge (0.982 to
1.000). Offline re-sizing of the saved TAVs changes no other depth except two
``StreamingConcat -> MVAU_rtl`` edges of yolov8's small folding (256 to 856 and
2431).

Decoupled chain
---------------

*What:* no burst floor (and no throttled cap) for an edge whose consumer can be
throttled, when two more conditions hold: the slowest node between the last
pacer and the producer (the producer excluded, ``t_above``) takes at least
``1 - 1 / CAP_MARGIN`` (20 %) of the period, and it and the slowest node at or
below the consumer (``t_below``, ``downstream_periods``) take at most
``CAP_MARGIN`` periods together. The depth is then what the slack relaxation
leaves, at least 2.

*Why:* the peak occupancy is the depth at which the producer never waits. The
burst floor puts that peak back wherever the consumer's reads are bursty against
the frame rate, which on a chain of fast nodes between slow ones is every edge:
each fast producer runs a row ahead of the slow node below it, and the same
backlog shows up again at every such edge down the chain. With a shallow FIFO the
slow stage above the producer and the slow stage below the consumer are coupled:
at worst they take turns, and a frame takes ``t_above + t_below``; mostly they
still overlap (the fast nodes in between pass rows on as they come), so up to
``CAP_MARGIN`` periods are taken as harmless. A slow stage above the producer
also means the producer is not fed just in time by the pacer, so blocking it
does not stall the pacer at once. The exceptions are where the floor is needed: a
producer right below the pacer (only fast nodes in between) holds the pacer's
continuous output while the nodes below pause (padding rows, a sliding window
generator's first rows); two stages at 0.67 or 0.9 of the period on either side
of the edge stall each other.

*Example:* mobilenet_v1, committed U55C folding (pacer the ``Thresholding_rtl``
with one channel per cycle below ``MVAU_hls_1``, 788544 cycles; the nodes below
take at most 0.51 of the period). The sizer put 97.2 kB into 141 FIFOs, 76 % of
it on wide edges with a burst floor equal to the peak (``VVAU_hls -> DWC`` 640 /
352 / 176 bit at 77-223, ``FMPadding_rtl -> ConvolutionInputGenerator_rtl`` at
202-256, ``MVAU_hls -> DWC`` at 176-226). Stitched-IP rtlsim (2 frames,
interval) with depths changed against the sized design: every FIFO into a
consumer that can be throttled at 2: 794807 (-0.8 %, 9.5 kB); the same except the
three below the pacer (``DWC -> FMPadding_rtl_1`` 14, ``FMPadding_rtl_1 ->
ConvolutionInputGenerator_rtl_2`` 256, ``VVAU_hls_1 -> DWC`` 80): 788544, 17.0
kB, which is what the rule gives. Each of those three alone at 2 loses 0.1 to 0.4
%; all ``VVAU_hls -> DWC`` edges below them at 2: 788544, all
``FMPadding -> ConvolutionInputGenerator`` below them at 2: 788544. Small
folding (``MVAU_hls`` at 0.90 of the period on both sides of every edge, so the
second condition fails): every ``DWC -> MVAU_hls`` at 2 loses 29 %, every FIFO
into a throttleable consumer at 2 loses 40 %; the rule changes nothing there.
``test_fifosizing_linear[cnv]`` (target 1000 fps; ``MVAU_hls_6`` and ``_7`` at
0.67 of the period): ``StreamingDataWidthConverter_rtl_12 -> MVAU_hls_7`` 63 to
20 without the second condition dropped rtlsim to 0.83 of the estimate; with it
the edge keeps 63. Other models (offline re-sizing of the same TAVs, total
kbit): vgg10-radioml 15.6 to 15.3 (rtlsim 1027 cycles either way), transformer_radioml
387.0 to 386.2 (2304 either way), resnet18 2237.7 to 2225.2, transformer_vision
1291.1 to 1290.9, espcn and bert by less than 0.1 %, the others unchanged.

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

Join window floor
-----------------

*What:* ``join_window_floors`` gives every input of a two-stream join
(``ElementwiseAdd`` / ``AddStreams``) and of a ``StreamingConcat`` a floor that
does not come from any TAV (see `Concat joins and sliding windows`_).
From the join it walks back to the fork its inputs share, composing each
node's ``tav_input_window()``: output block *b* of ``out_words`` words
needs the first ``(b + 1) * in_words`` words of the input (default: the ratio
of the folded frame sizes in lowest terms, so a 48:1 width converter needs 48
words a word; ``MVAU`` in0 ``(MW / SIMD, MH / PE)``, a vector (TH vectors when tiled); ``HWSoftmax`` a
row; ``InnerShuffle`` a page; a dynamic ``MVAU``'s in1 the whole matrix; a node
that reorders a frame gives the need of every output word instead,
``tav_input_need()``, e.g. ``OuterShuffle`` from its permutation). That
gives ``need_i(k)``, the fork words input *i* needs for its first *k* tokens,
and two backlogs over two frames:

* deadlock: to deliver join token *k* on every input the fork has to write
  ``W(k) = max_i need_i(k)`` words, so input *i* holds the tokens it makes of
  the first ``W(k) - 1`` words minus the ``k - 1`` the join has taken;
* paced: the fork writes at the graph's period and every node emits word *j*
  once its window is in and no faster than its own ``get_exp_cycles()``,
  ``T(j) = max(T_in(need(j)), T(j - 1) + r)``; input *i* holds what has arrived
  when the join takes token *k*, minus ``k - 1``.

The floor is the larger backlog plus ``JOIN_FLOOR_MARGIN`` (25 %).

*Why:* a fork writes all its outputs together, so it stalls as soon as one
branch is full. If the long branch reads many fork words before its first
output reaches the join, the shortcut has to buffer all of them, or the fork
stalls, the long branch never completes its window and the graph deadlocks.
For full throughput it also has to buffer what the fork writes while a window
node at the pacer's rate emits its block. The timeline pass misses both where
a TAV writes before it reads (an ``MVAU`` tree model places the first output 44
to 953 cycles before the first input) and the fork is transparent. The
backlogs ignore the storage inside the nodes and in the other FIFOs of the
shortcut, which only helps; the margin covers a node that reads ahead of its
window or an upstream that delivers in bursts.

*Example:* transformer_radioml (ZCU104). Committed folding: the MLP shortcut
``Thresholding_rtl_13 -> ElementwiseAdd_hls_4`` was sized 20 before the
`Node latency`_ rule and deadlocked (no output in 1.71 M cycles; 128 completes
a frame); the ``StreamingDataWidthConverter`` 8 -> 384 bit in the long branch
needs all 48 fork words of a token: deadlock backlog 47, paced 90, floor 113
(the timeline now gives 172). Small folding: shortcut 122, deadlock backlog 95;
rtlsim 1912 frames/s against an estimate of 3052 (0.63x). Raising only that FIFO
to 190 runs at the estimate (32768 cycles a frame): the long branch's
``MVAU_rtl_11`` (SIMD 24, PE 3, the pacer) emits a token's 128 words over a
token period after the token is in, so the shortcut holds two tokens; paced
backlog 191, floor 239. transformer_vision: the MLP shortcuts were 8
(committed folding, before this rule; 134 with the node latency) and 92 / 108
(optimizer folding at the target), 92 / 108 deadlocked (512 completes);
floor 419 (paced backlog 335). The attention shortcuts take the floor too
(radioml 7680 against 6449 from the timeline: the dynamic ``MVAU`` needs the
whole matrix and two ``InnerShuffle`` pages lie in front of it). bert, small
optimizer folding: the attention shortcut (its fork is fed by the graph input,
so the timeline pass leaves it at 2) got 6785 with ``OuterShuffle`` taken as
one-to-one and hung (no output in 9.5 M cycles); at 98304 it runs at the
estimate (786432 cycles a frame); with the ``OuterShuffle`` need the floor is
118545. Its committed small folding (tiled ``MVAU``, TH = 8: a group of 8
vectors is one window) ran at 0.82x with the FFN shortcut at 4566; the floor
is 5813, and at 5813 it runs at the estimate. Total FIFO storage (offline re-sizing of the same TAVs):
transformer_benchmark +4.7 % (attention shortcut 2140 -> 2698; it needs 2065
to 2077), transformer_radioml, _vision and _language +2.5 to 2.6 %, bert
+18 % (attention shortcut 21338 -> 30303, 128 bit wide; 21338 passes),
resnet50 +0.2 %; bnn-pynq, cybersecurity-mlp, gtsrb, kws, mobilenet_v1,
vgg10-radioml, espcn, resnet18 and yolov8 unchanged (their joins are above
the floor).

Concat joins and sliding windows
--------------------------------

*What:* the join window floor also covers ``StreamingConcat``. A join token is
the greatest common divisor of the input frame sizes, so an add keeps one word a
token and a concat takes each input's share of a pixel. ``FMPadding``,
``ConvolutionInputGenerator`` and ``UpsampleNearestNeighbour`` give the input
each output word needs (``tav_input_need``): a padding word the input up to
the last real pixel before it, a window its last pixel, an upsampled pixel its
nearest neighbour. Before, all three counted as one-to-one through the frame
ratio.

*Why:* every C2f, FPN and PAN stage of yolov8 is a fork whose branches meet again
at a concat, and the long branches run through padding, 3x3 windows, stride-2
windows and upsampling. A 3x3 window needs a row and a pixel more than its output
position, so each of these stages delays the long branch by rows of its own
resolution. Taken as one-to-one the floor never sees that lag, and the timeline
pass alone has no margin on it.

*Example:* yolov8_4w4a, small folding (U55C, 11.8 M cycles a frame). The PAN
concat ``StreamingConcat_hls_12`` joins the SPPF output (6x10, shortcut
``DuplicateStreams_hls_11 -> StreamingConcat_hls_12``) with a path through two
upsamplings, the FPN and PAN C2f stages and two stride-2 windows. The timeline
gave the shortcut 9086 words (of a 15360-word frame). Stitched-IP rtlsim: no
output word in 383 M cycles (the run with unbounded FIFOs finishes the frame
in 34.9 M cycles and peaks at 15358 on that shortcut). A rerun with monitors
shows the shortcut full from 22.9 M cycles on, with nothing reaching the concat's
other input. The structural backlog is 8704 words, the paced one 11073, and the
floor is 13842. With it the design gives its first output word at 23.7 M cycles.
Every other join FIFO of this build was already at or above its unbounded-FIFO
peak. Total FIFO storage (offline re-sizing of the same TAVs): yolov8 +1.5 to
+3.3 % (the C2f shortcuts take the 25 % margin over a window lag the timeline
already got right), resnet18 +0.2 % (full) and +1.6 % (small), resnet50 +5.6 %
(full) and +5.9 % (small); the other models are unchanged (no concat, or no
sliding window between fork and join).

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
cnv-w2a2           3045            3051             83.5             25.5
cybersecurity-mlp  16.7 M          16.7 M           1.9              3.7
kws                25471           25471            0.9              4.3
gtsrb              8134            8134             26.7             9.9
vgg10-radioml      243427          243427           14.0             14.2
=================  ==============  ===============  ===============  ===============

(The MLP throughputs are the one-frame testbench's interval, not a sustained
rate.) Where the heuristic is larger it is the graph-input FIFO (tfc,
cybersecurity-mlp) or the burst floor under the paced-source assumption (kws);
where it is smaller (cnv-w1a1, cnv-w2a2, gtsrb) the reference measured the
saturating testbench's burst into the first FIFOs (617 on
``StreamingDataWidthConverter_rtl_0 -> ConvolutionInputGenerator_rtl_0`` in
cnv-w1a1 and gtsrb, 2529 on ``Thresholding_rtl_0 ->
ConvolutionInputGenerator_rtl_0`` in cnv-w2a2) and the run-ahead the slack
relaxation gives up (497 against 59 on gtsrb's ``Pool_hls_0 ->
ConvolutionInputGenerator_rtl_3``, 1152 against 594 on cnv-w2a2's
``StreamingDataWidthConverter_rtl_6 -> ConvolutionInputGenerator_rtl_4``).
In the other direction the heuristic keeps burst floors the saturating
testbench never needs (``Pool_hls_1 -> ConvolutionInputGenerator_rtl_6`` 192
against 2 on cnv-w2a2, ``StreamingDataWidthConverter_rtl_7 ->
ConvolutionInputGenerator_rtl_5`` 151 against 2 on gtsrb). Both reach the
reference throughput within 0.2 %.

Impact of each rule
===================

How much each rule matters was measured by removing it alone from the full
sizer (an *ablation*) and sizing every regression model again: the change of
the total FIFO depth and of the total storage (depth x stream width, kbit)
says what the rule buys or costs; a stitched-IP rtlsim of the ablated depths
(ZCU104, Vivado 2024.2, one frame, ``auto_fifo_depths=False`` with the ablated
depths as a folding config) says whether throughput depends on it. tfc-w1a1,
tfc-w2a2 and cybersecurity-mlp are at the minimum depth everywhere and no rule
changes them. mobilenet_v1 and resnet50 are depth-only where noted
(mobilenet_v1's ZCU104 stitched-IP rtlsim returns undefined output bits on
2024.2 as it did on 2023.1, so its builds fail before a throughput is
measured; resnet50's V80 build has no rtlsim here).

Two classes of rules:

* **depth-minimizing** rules take depth away from the peak occupancy; removing
  one makes FIFOs deeper and can never cost throughput. They are: the paced
  graph input, the slack relaxation (with its two helpers, the reset past the
  pacer and the pacer tolerance), the throttled cap, the two-frame split and
  the fill-frame-only evaluation of join/branch edges.
* **throughput-protecting** rules put depth back where a minimizing rule would
  over-minimize; removing one makes FIFOs shallower and risks a stall. The
  burst floor corrects the slack relaxation and the paced input (a bursty
  consumer needs its burst buffered whatever the average rate says); the cap
  margin corrects the throttled cap; the join/branch rule and the idle-window
  floor correct the slack relaxation and the cap on edges whose deadline is a
  join; the max of the causal and non-causal pass corrects each pass by the
  other; the measured global period corrects the caller's period estimate.

Ranking by storage effect (median and largest change of the total storage over
the models the rule changes, the largest change of the total depth, and the
worst rtlsim throughput change measured when the rule is removed):

====================================  ==========  ======  =============  ==========  ===========  ================  =====================================================================
rule                                  class       models  median kbit %  max kbit %  max depth %  worst fps %       models affected                                                      
====================================  ==========  ======  =============  ==========  ===========  ================  =====================================================================
paced graph input                     minimizing  5       +189.9         +969.5      +213.3       none (3 sim.)     cnv-w1a1, cnv-w2a2, gtsrb, mobilenet_v1, resnet50                    
slack relaxation                      minimizing  5       +11.5          +460.1      +587.4       none (3 sim.)     cnv-w2a2, gtsrb, vgg10-radioml, mobilenet_v1, resnet50               
slack reset past the pacer            minimizing  2       +204.9         +398.4      +584.8       none (1 sim.)     vgg10-radioml, mobilenet_v1                                          
pacer tolerance                       minimizing  2       +204.9         +398.4      +584.8       none (1 sim.)     vgg10-radioml, mobilenet_v1                                          
throttled cap                         minimizing  1       +371.6         +371.6      +582.5       -                 mobilenet_v1                                                         
two-frame split                       minimizing  6       +35.6          +87.8       +113.6       none (4 sim.)     cnv-w1a1, cnv-w2a2, gtsrb, vgg10-radioml, mobilenet_v1, resnet50     
join/branch edges never relaxed       protecting  1       -79.6          -79.6       -85.0        -                 resnet50                                                             
burst floor                           protecting  7       -18.6          -78.0       -94.2        -46.1 (cnv-w2a2)  kws, cnv-w1a1, cnv-w2a2, gtsrb, vgg10-radioml, mobilenet_v1, resnet50
idle-window floor (branch edges)      protecting  1       -9.7           -9.7        -8.9         -                 resnet50                                                             
fill frame only on join/branch edges  minimizing  1       +5.3           +5.3        +4.2         -                 resnet50                                                             
non-causal pass (max of two passes)   protecting  1       -1.6           -1.6        -2.0         -                 resnet50                                                             
causal pass (max of two passes)       protecting  4       -0.0           -0.1        -0.4         none (3 sim.)     cnv-w1a1, gtsrb, mobilenet_v1, resnet50                              
cap margin                            protecting  1       -0.0           -0.0        -0.0         -                 resnet50                                                             
measured global period                protecting  0       +0.0           +0.0        +0.0         -                 none                                                                 
====================================  ==========  ======  =============  ==========  ===========  ================  =====================================================================

Per model, the change of total depth / total storage / rtlsim throughput when
the rule is removed (``=`` no change; throughput only where simulated):

====================================  ====================  ======================  ======================  ======================  ===================  ===============  =============
rule removed                          kws                   cnv-w1a1                cnv-w2a2                gtsrb                   vgg10-radioml        mobilenet_v1     resnet50     
====================================  ====================  ======================  ======================  ======================  ===================  ===============  =============
paced graph input                     =                     +213.3 / +113.7 / +0.0  +189.0 / +258.3 / +0.0  +117.5 / +189.9 / +0.0  =                    +40.8 / +969.5   +34.1 / +61.8
slack relaxation                      =                     =                       +40.5 / +25.1 / +0.0    +9.0 / +9.3 / +0.0      +3.8 / +11.5 / +0.0  +587.4 / +460.1  =            
slack reset past the pacer            =                     =                       =                       =                       +3.8 / +11.5 / +0.0  +584.8 / +398.4  =            
pacer tolerance                       =                     =                       =                       =                       +3.8 / +11.5 / +0.0  +584.8 / +398.4  =            
throttled cap                         =                     =                       =                       =                       =                    +582.5 / +371.6  =            
two-frame split                       =                     +53.5 / +54.2 / +0.0    +113.6 / +76.5 / +0.0   +41.6 / +17.1 / +0.0    +3.1 / +1.7 / +0.0   +4.6 / +11.3     +89.4 / +87.8
fill frame only on join/branch edges  =                     =                       =                       =                       =                    =                +4.2 / +5.3  
burst floor                           -94.2 / -78.0 / +0.0  -35.1 / -18.6 / -1.4    -37.2 / -13.1 / -46.1   -59.9 / -23.6 / -17.0   -6.1 / -1.8 / +0.0   -68.2 / -65.0    -1.1 / -2.1  
causal pass (max of two passes)       =                     -0.4 / -0.0 / +0.0      =                       -0.1 / -0.1 / +0.0      =                    =                =            
non-causal pass (max of two passes)   =                     =                       =                       =                       =                    =                -2.0 / -1.6  
join/branch edges never relaxed       =                     =                       =                       =                       =                    =                -85.0 / -79.6
idle-window floor (branch edges)      =                     =                       =                       =                       =                    =                -8.9 / -9.7  
====================================  ====================  ======================  ======================  ======================  ===================  ===============  =============

What the numbers say:

* **Heavy-duty depth minimizers**: the paced graph input (x2-x4 depth on every
  convolutional model without it, +970 % storage on mobilenet_v1), the slack
  relaxation together with its pacer reset and tolerance (x6 depth on
  mobilenet_v1, +25-40 % on cnv-w2a2) and the throttled cap (x6 depth on
  mobilenet_v1; it is the only model with a throttleable consumer whose burst
  exceeds 256). The two-frame split is the fourth: +40-115 % depth on the
  convolutional models, +89 % on resnet50. None of these costs throughput
  where it was simulated (cnv-w1a1, cnv-w2a2, gtsrb, kws, vgg10-radioml).
* **The one heavy-duty throughput protector** is the burst floor: it is what
  keeps a bursty consumer (an MVAU reading ``SF`` words at the start of each
  output window, a sliding window generator reading rows) fed once the slack
  relaxation and the paced input have taken the run-ahead away. Without it
  cnv-w2a2 loses 46 % and gtsrb 17 % of its throughput, cnv-w1a1 1.5 %; the
  depth it adds is 35-95 % of the total on those models, but only 13-24 % of
  the storage, because it lands on narrow edges. On kws the same rule
  over-sizes (4.3 kbit instead of 0.9 for no throughput), see `Burst floor`_.
* **Minor rules**: the causal/non-causal pass pair changes one token on one
  edge of the linear models and 2 % of resnet50's depth (non-causal larger);
  the cap margin changes 16 tokens on one resnet50 edge; the fill-frame-only
  rule 4 % of resnet50's depth; the idle-window floor 9 % of resnet50's
  storage (deep DWC -> FMPadding FIFOs inside residual branches). The join
  rule is minor in count (one model) but not in size: without it resnet50's
  shortcut FIFOs collapse (-80 % storage) into what the deadlock argument says
  cannot work; it was not simulated.
* **No effect on any model**: the measured global period (the builder already
  passes the slowest TAV period; the rule only matters for a caller that
  passes none) and, on the linear models, the cap margin. Both are
  candidates for removal if a resnet50 simulation shows the 16-token edge
  does not need the margin; the measured period is one line and guards
  ``DeriveFIFOSizes(period=None)``.
