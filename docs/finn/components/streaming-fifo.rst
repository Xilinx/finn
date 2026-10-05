.. _streaming_fifo:

Streaming FIFO
=======================================
A FINN-compiled accelerator requires FIFOs to buffer residual paths or
parallelism mismatches.
These FIFOs are implemented in
`fifo.sv <https://github.com/Xilinx/finn/blob/main/finn-rtllib/fifo/hdl/fifo.sv>`_.

Debug Logging
-------------
Streaming FIFOs can be used to debug designs. We introduce debug logging
to our FIFO implementation, which makes every FIFO into an *unbound*, *recorded*
queue. Typical use cases for this can be:

- **Mid-simulation** observation: Let a script or agent observe whether the fifo
  logs are progressing, or if a FIFO is silently accumulating the entire input.
- **Optimisation**: cycles, occupancy and fill rates allow parallelism
  intelligent parallelism configurations on the design.
- **Visualisation tools**: Verbose debug logs combine with an ONNX graph to
  visually highlight hotspots, starvation, or give general insight into
  accelerator performance.

Such debug logging can be enabled and parameterised by setting the following
fields in
:py:class:`~finn.builder.build_dataflow_config.DataflowBuildConfig`:

.. literalinclude:: /../../src/finn/builder/build_dataflow_config.py
   :language: python
   :start-at: #: Enable FIFOs to output log files documenting what they witness.
   :end-at: fifo_log_flush_cycles: Optional[int] = 10000
   :dedent:

- ``debug_fifo``: Enable FIFO logging for this build.
  - This automatically enables ``verify_rtlsim_behavioural``.
- ``fifo_log_verbose``: the verbosity of the log (see below).
- ``fifo_log_flush_cycles``: how many simulation cycles to wait before flush to
  disk.

These parameters are passed down to
`fifo_gauge.sv <https://github.com/Xilinx/finn/blob/main/finn-rtllib/fifo/hdl/fifo_gauge.sv>`_,
which implements the logging logic.

Logs outputs are located under
``<output_dir>/debug/fifo_logs/<phase>/<scope>/``:
- ``<phase>``: phase of the build, such as ``fifo_sizing`` or stiched_ip_rtlsim``
- ``<scope>``: ``main`` or a ``FINNLoop``.
- Each log is named after the FINNLoop and ONNX node name, e.g. ``FINNLoop_0_StreamingFIFO_rtl_12_t8515yv3.log``.

Log Verbosity
~~~~~~~~~~~~~
The ``LOG_VERBOSE`` parameter controls how verbose the logs are.
This gives control of a trade-off between design visibility and simulation speed.

When verbose logs are off (``LOG_VERBOSE=0``), an example log file could be:

.. code-block:: text

   # data_in
   1f0f
   1101
   1f0f
   0f10
   1101
   # [StreamingFIFO_1.inst @4216778000] Cycles: 11432; MaxFill: 2; Transactions: in=4 out=3

This verbosity only prints the data that it has seen at the input port, tailed
by a summary of tho total properties. This is enough for the FIFO sizing
simulation, as the only number it cares for is MaxFill (2).

When verbosity is turned on (``LOG_VERBOSE=1``), the same FIFO would log this:

.. code-block:: text

   # data direction cycles
   1f0f 0 1080
   1101 0 2481
   1f0f 1 4221
   1f0f 0 5112
   1101 1 6223
   1f0f 1 7283
   0f10 0 8922
   # [StreamingFIFO_1.inst @4216778000] Cycles: 11432; MaxFill: 2; Transactions: in=4 out=3

The main differences are that we no longer monitor just the input, but also the
*output*, and *cycle timestamp* of each transaction. Whilst slowing down the
simulation (see below), this lets you debug IP stalls, FIFO resource estimations,
or identify resource estimations based on the amount of time spent waiting.

Performance Characterisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Increased logging isn't free, which is why it's important to understand how
it affects your build times. The table below documents a small ONNX graph
with a residual path.

+---------------+-----------------+------------------------+-----------------+
| Configuration | Flush (cycles)  | FIFO Sizing (Seconds)  | Log Size (MiB)  |
+===============+=================+========================+=================+
| Log Off       |                 | 1.6                    |                 |
+---------------+-----------------+------------------------+-----------------+
| Default       | 1               | 7.3                    | 23              |
+---------------+-----------------+------------------------+-----------------+
| Verbose       | 1               | 9.0                    | 244             |
+---------------+-----------------+------------------------+-----------------+
| Default       | 10,000          | 2.7                    | 23              |
+---------------+-----------------+------------------------+-----------------+
| Verbose       | 10,000          | 4.4                    | 244             |
+---------------+-----------------+------------------------+-----------------+
| Default       | 100,000         | 2.7                    | 23              |
+---------------+-----------------+------------------------+-----------------+
| Verbose       | 100,000         | 4.4                    | 244             |
+---------------+-----------------+------------------------+-----------------+

Out-takes:

- **File size**: Verbose FIFO logs are 10x larger than non-Verbose ones.
- **Single Flush**: At flush=1, verbose logs add **23%** execution time to
  the FIFO sizing simulation.
- **Large Flush**: At flushes > 10,000, verbose logs add **63%** execution
  time to the FIFO sizing simulation.
