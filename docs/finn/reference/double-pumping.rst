.. _double_pumping:

*************************************
Double Pumping
*************************************

MVAUs can run parts of their datapath at twice the clock frequency of the rest
of the design: ``pumpedCompute=1`` doubles the DSP compute (half the DSPs for the
same throughput), ``pumpedMemory=1`` the weight memory (half the memory width).
Such a node has a second clock input ``ap_clk2x``. This clock must run at twice
the frequency of ``ap_clk`` and be **phase-aligned** with it: rising edges of
``ap_clk`` coincide with rising edges of ``ap_clk2x``, so that the crossings
between both domains are timed as synchronous paths.

Internal clock generator
========================

By default, the kernels integrated into a shell derive ``ap_clk2x`` themselves.
The stitched IP that is delivered (``DataflowOutputType.STITCHED_IP``,
``PORTABLE_RTL``) does so with
``DataflowBuildConfig.clk2x_source = Clk2xSource.GENERATOR``; by default it has
an ``ap_clk2x`` input (see `External ap_clk2x`_). If one of its nodes needs
``ap_clk2x``,
:py:mod:`finn.transformation.fpgadataflow.create_stitched_ip.CreateStitchedIP`
(``clk2x_gen=True``) instantiates a clock generator fed by ``ap_clk``, and the IP
exposes only ``ap_clk`` and ``ap_rst_n`` (see ``finn-rtllib/clk2x_gen``):

* An MMCM generates the 1x and the 2x clock. ``clk2x_domain.v`` instantiates the
  primitive of the device family directly (``MMCME2_ADV``, ``MMCME3_ADV``,
  ``MMCME4_ADV`` or ``MMCME5``), without the Clocking Wizard IP, so the generator
  does not depend on the IP versions of a Vivado installation. The whole design
  logic runs on the 1x output, which reproduces ``ap_clk`` (same frequency and
  phase), the pumped parts on the 2x output.
* The MMCM is configured for an ``ap_clk`` of ``1000/synth_clk_period_ns`` MHz.
  Its VCO is put at the geometric center of its range, so that the IP tolerates
  the widest range of actual input frequencies, e.g. a PS clock that cannot be
  synthesized exactly. The tolerated range is printed during stitching and
  recorded in ``clk2x_gen.json`` in the stitched IP project. If the actual
  ``ap_clk`` of a system is outside it, Vivado reports the DRC error PDRC-179.
* Both generated clocks form one ``CLOCK_DELAY_GROUP``, so that the router
  balances their clock trees. The constraint ships with the IP as module
  constraint of ``clk2x_domain`` (``clk2x_domain.xdc``, see
  ``HWCustomOp.get_xdc_files``), which Vivado reads for the generator wherever the
  IP is implemented, also for IPs packaged as synthesized checkpoints. VitisLink
  additionally sets the group in an ``opt_design`` pre-hook, as long as the
  module constraint has not been confirmed in a Vitis link.
* ``ap_rst_n`` also resets the MMCM. The design logic leaves reset only after the
  MMCM has locked. Until then, the handshakes of all external AXI-Stream, AXI-Lite
  and AXI-MM interfaces are blocked: AXI-Lite transfers stall (no ``READY``)
  rather than fail, and no AXI-MM transfer is issued.
* In RTL simulation, the MMCM is replaced by ideal clocks (``ifdef SYNTHESIS`` in
  ``clk2x_domain.v``). A portable RTL export contains both branches.

Inner IPs, which are stitched into an enclosing IP (FINNLoop bodies) or stitched
only for FIFO sizing or simulation, keep an ``ap_clk2x`` input.

Shell flows
===========

* **Zynq (PYNQ)**: the kernels derive ``ap_clk2x`` from the PL clock.
* **Alveo (Vitis)**: the kernels derive ``ap_clk2x`` from kernel clock 0, which
  VitisLink sets to ``1000/synth_clk_period_ns`` MHz. This must be an integer in
  MHz, and v++ may not lower the kernel clock if timing fails
  (``compiler.enableAutoFrequencyScaling=false``).
* **Versal (VCK190 golden reference) and SLASH (V80)**: double-pumped designs are
  rejected. The internal clock generator is the intended path, but has not been
  verified with these shells yet.

External ap_clk2x
=================

The delivered stitched IP has an ``ap_clk2x`` input by default, which the
integrating system must drive. With ``clk2x_source = Clk2xSource.EXTERNAL``, so
do the shell kernels. Of the shell flows, only Vitis supports this, for
platforms that provide a phase-aligned clock pair: set ``vitis_clk2x_clock_ids``
to the two platform clock ids for ``ap_clk`` and ``ap_clk2x``. The kernels then
run on these clocks instead of kernel clock 0. For example, the VCK190 Vitis
base platform lists such a pair as ids ``[2, 1]`` (156.25 and 312.5 MHz); this
has not been verified with a link.
