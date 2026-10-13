.. _streaming_fifo:

Streaming FIFO
=======================================
In a FINN dataflow accelerator, all layers run concurrently and exchange data
through streams. Their production and consumption patterns rarely match
exactly: layers differ in latency, and many produce or consume data in bursts
rather than at a steady rate. FIFOs between layers absorb these differences so
that a briefly stalled consumer does not stall its producer, which is necessary
to reach the full throughput of a given folding configuration. Where paths fork
and rejoin, such as the skip connection of a residual block, sufficiently deep
FIFOs are additionally required to avoid deadlock.

Because the required depths depend on the folding configuration, FINN can size
these FIFOs automatically (see ``auto_fifo_depths`` in
:py:class:`~finn.builder.build_dataflow_config.DataflowBuildConfig`).
The FIFOs are implemented in
`fifo.sv <https://github.com/Xilinx/finn/blob/main/finn-rtllib/fifo/hdl/fifo.sv>`_.

Debug Logging
-------------
Streaming FIFOs can be used to debug designs. In behavioral RTL simulation,
every FIFO is replaced by
`fifo_gauge.sv <https://github.com/Xilinx/finn/blob/main/finn-rtllib/fifo/hdl/fifo_gauge.sv>`_,
an *unbounded* queue that never applies backpressure. With debug logging
enabled, each of these queues also *records* the transactions it sees to a log
file. Since the FIFOs are unbounded in these simulations, the logs show how full
each FIFO would need to be, not stalls caused by the FIFO depths of the final
design. Typical use cases are:

- **Mid-simulation observation**: let a script or agent check whether the FIFO
  logs are progressing, or whether a FIFO is silently accumulating the entire
  input.
- **Optimization**: cycles, occupancy and fill rates help to find better
  parallelism configurations for the design.
- **Visualization tools**: verbose debug logs can be combined with the ONNX
  graph to highlight hotspots and starvation, or to give general insight into
  accelerator performance.

Debug logging is enabled and parameterized by the following fields in
:py:class:`~finn.builder.build_dataflow_config.DataflowBuildConfig`:

.. literalinclude:: /../../src/finn/builder/build_dataflow_config.py
   :language: python
   :start-at: #: Enable FIFOs to output log files documenting what they witness.
   :end-at: fifo_log_flush_cycles: Optional[int] = 10000
   :dedent:

- ``debug_fifo``: enables FIFO logging for this build.

  - This automatically enables ``verify_rtlsim_behavioral``, since the logs are
    written by the behavioral FIFO model.

- ``fifo_log_verbose``: selects the log format (see `Log Verbosity`_).
- ``fifo_log_flush_cycles``: how many simulation cycles each FIFO waits before
  flushing its log to disk. A FIFO only flushes if it has logged something since
  its last flush, so all logs lag the simulation by at most this many cycles,
  which makes it possible to watch the logs while a long simulation is still
  running. Smaller values keep the logs more up to date but slow down the
  simulation (see `Performance Characterization`_). ``0`` disables periodic
  flushing, so the logs are only written when the simulation ends.

``fifo_log_verbose`` and ``fifo_log_flush_cycles`` only have an effect when
``debug_fifo=True``.

Log outputs are located under ``<output_dir>/debug/fifo_logs/<phase>/<scope>/``:

- ``<phase>``: phase of the build, i.e. ``fifo_sizing`` or ``stitched_ip_rtlsim``.
- ``<scope>``: ``main`` for the top-level graph, or the name of a ``FINNLoop``
  (e.g. ``FINNLoop_0``) for FIFOs inside a loop body.
- Each log is named after its ONNX node, prefixed by the enclosing FINNLoop if
  there is one, and followed by the suffix of the node's IP generation
  directory, e.g. ``FINNLoop_0_StreamingFIFO_rtl_12_t8515yv3.log``.

Log Verbosity
~~~~~~~~~~~~~
``fifo_log_verbose`` controls how much each FIFO logs, which is a trade-off
between design visibility and simulation speed.

With ``fifo_log_verbose=False`` (the default), an example log file could be:

.. code-block:: text

   # data
   1f0f
   1101
   1f0f
   0f10
   # [StreamingFIFO_rtl_1.inst.fifo @4216778000] Cycles: 11432; MaxFill: 2; Transactions: in=4 out=3

After a header line, the log lists every data word seen at the input port, in
hexadecimal, followed by a summary line with the FIFO's hierarchical instance
name (shortened here), the number of simulated cycles, the maximum fill level
and the number of input and output transactions. This is enough for FIFO
sizing, which only needs the maximum fill level (``MaxFill``).

With ``fifo_log_verbose=True``, the same FIFO logs:

.. code-block:: text

   # data dir cycle
   1f0f 0 1080
   1101 0 2481
   1f0f 1 4221
   1f0f 0 5112
   1101 1 6223
   1f0f 1 7283
   0f10 0 8922
   # [StreamingFIFO_rtl_1.inst.fifo @4216778000] Cycles: 11432; MaxFill: 2; Transactions: in=4 out=3

Each line now records the data word, its direction (``0`` for input, ``1`` for
output) and the cycle of the transaction. While this slows down the simulation
(see below), it lets you see when each word entered and left the FIFO, which
helps to debug stalls and to understand where data is waiting.

The logs can be loaded in Python with :py:func:`finn.util.rtlsim.parse_fifo_log`
for a single file, or :py:func:`finn.util.rtlsim.read_fifo_log_snapshot` for a
whole ``<phase>/<scope>`` directory.

Performance Characterization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Logging is not free, so it is important to understand how it affects your
build times. The table below shows the FIFO sizing simulation of a small ONNX
graph with a residual path.

+---------------+-----------------+------------------------+-----------------+
| Log format    | Flush (cycles)  | FIFO Sizing (Seconds)  | Log Size (MiB)  |
+===============+=================+========================+=================+
| Logging off   |                 | 1.6                    |                 |
+---------------+-----------------+------------------------+-----------------+
| Plain         | 1               | 7.3                    | 23              |
+---------------+-----------------+------------------------+-----------------+
| Verbose       | 1               | 9.0                    | 244             |
+---------------+-----------------+------------------------+-----------------+
| Plain         | 10,000 (default)| 2.7                    | 23              |
+---------------+-----------------+------------------------+-----------------+
| Verbose       | 10,000 (default)| 4.4                    | 244             |
+---------------+-----------------+------------------------+-----------------+
| Plain         | 100,000         | 2.7                    | 23              |
+---------------+-----------------+------------------------+-----------------+
| Verbose       | 100,000         | 4.4                    | 244             |
+---------------+-----------------+------------------------+-----------------+

Takeaways:

- **Logging itself**: with the default flush interval, plain logs already add
  about **70%** to the FIFO sizing simulation time (1.6 s to 2.7 s).
- **File size**: verbose logs are about 10x larger than plain ones.
- **Flushing every cycle**: at a flush interval of 1, verbose logs add **23%**
  simulation time compared to plain logs.
- **Larger flush intervals**: at 10,000 cycles or more, verbose logs add **63%**
  simulation time compared to plain logs. Beyond 10,000 cycles, increasing the
  interval brings no further speedup.
