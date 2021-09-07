.. _faq:

***********************
Frequently Asked Questions
***********************

.. note:: **This page is under construction.**

Can I install FINN out of the Docker container?
===============================================

We do not support out of the Docker implementations at the moment. This is due
to the high complexity of the FINN project dependencies.

Since FINN uses ONNX, can I compile any model from the ONNX Model Zoo to an FPGA accelerator?
=============================================================================================

The short answer is no. FINN uses ONNX in a specific (non-standard) way, including custom layer
types and quantization annotations. Networks must be first quantized using Brevitas and exported
to FINN-ONNX to be converted to FPGA accelerators.


Can I deploy custom NNs with arbitrary precisions and layers using FINN?
=========================================================================

Yes, though the effort required and quality of results will vary.
Although we do support arbitrary
precision, the way we create the hardware isn't typically practical for more than
4 bits, or very large networks for a single FPGA.
In terms of layers, only a subset of quantized layers covered by the various FINN examples
are currently supported.
It is possible to add support for new layers, though we don't have tutorials for this in place
just yet.

Does FINN only work with the example networks?
==============================================

FINN isn't restricted to the example networks;
rather, it's restricted to certain patterns (e.g. certain layer types and their combinations).
The current best practice for custom networks is to take a working network and gradually modify it.

What is the expected background for using FINN?
===============================================

Some general knowledge of Python, Docker, machine learning with neural networks and Jupyter notebooks
is expected.
Our goal is to make the tool in a shape and form so that no hardware/FPGA background
should be necessary, although having some knowledge would give better results.

What operating systems are supported by FINN?
=============================================

FINN should work fine under any Linux-based OS capable of running Vivado/Vitis, as long
as you install Docker (``docker-ce``) on your machine .


I am getting DocNav and Model_Composer errors when launching the Docker image.
==============================================================================

We do not mount those particular directories into the Docker container because they are not
used. The errors are Vivado related but you can safely ignore them.

What board do you recommend to start working with FINN?
=======================================================

Our preferred target platforms are those supported by  `PYNQ <http://www.pynq.io/board.html>`_.
For those boards we can offer end-to-end (DNN-to-bitstream) deployment,
see the `finn-examples <https://github.com/Xilinx/finn-examples>`_ repository for some examples.
However, FINN also supports Vivado IP Integrator designs. The IPs connect using AXI stream (FIFO)
in-and-out interfaces. This means that it can be integrated onto any Xilinx FPGA board,
though you will have to do the system integration manually.
