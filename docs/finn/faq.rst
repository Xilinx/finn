.. _faq:

***********************
Frequently Asked Questions
***********************

.. note:: **This page is under construction.**

Can I install FINN out of the Docker container?
===============================================

We do not support out of the Docker implementations at the moment. This is due 
to the high complexity of the FINN project dependencies. 

Can I deploy custom NNs using FINN? E.g. using arbitrary precisions and layers.
===============================================================================

FINN as it exists today is a network/hardware co-design tool targeting heavily 
quantized networks, and the existing compiler flows cannot support arbitrary 
networks such as the ones in the ONNX model zoo. Although we do support arbitrary 
precision, the way we create the hardware isn't typically practical for more than 
4 bits, or very large networks. FINN isn't really restricted to the tested networks; 
rather, it's restricted to certain patterns (e.g. certain layer types and their combinations). 
The current best practice for custom networks is to take a working network and gradually modify it. 

What is the expected background for using FINN?
===============================================

Our goal is to make the tool in a shape and form so that no hardware/FPGA background 
should be necessary (although having some knowledge would give better results). We 
are not there yet, but there will be some new features in the two coming releases that 
will help with the ease-of-use.

What OS is supported by FINN other than Ubuntu 18.04?
=====================================================

FINN should work fine under Linux based Operating Systems itself. However, FINN relies 
on Vivado(e.g. for Synthesis). Make sure the OS being used is supported by Vivado “2019.1” 
or “2020.1”. You should be able to install Docker engine in your machine (“docker-ce”).

I am getting DocNav and Model_Composer errors when launching the Docker image.
==============================================================================

We do not mount those particular directories into the Docker container because they are not
used. The errors are Vivado related but you can safely ignore them.

What board do you recommend to start working with FINN?
=============================================================================

FINN currently offers End-to-end deployment including the driver for PYNQ supported boards. 
You can have a look into what are the available PYNQ supported boards in (http://www.pynq.io/board.html). 
However, FINN now supports Vivado IP Integrator designs. The IPs connect using AXI stream (FIFO) 
in-and-out interfaces. This means that it can be integrated into any Xilinx FPGA board.
