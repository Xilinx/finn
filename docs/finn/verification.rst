.. _verification:

***********************
Functional Verification
***********************

.. note:: **This website is currently under construction.**

.. image:: /img/verification.png
   :scale: 70%
   :align: center

There are three ways to verify a network in FINN functionally. These ways of verification can be applied in different stages of the network inside FINN. All can be accessed using the execution function in module :py:mod:`finn.core.onnx_exec`. The execution happens in most cases node by node, which supports networks that have a mixture of standard ONNX nodes, custom nodes and HLS custom nodes. Because the single node can be executed using one of the following methods.

Simulation using Python
=======================

This simulation can be used right after the :ref:`brevitas_export` or when the network does not contain any HLS custom nodes, so right after the streamlining transformations and before the nodes are converted into HLS layers. 

Simulation using C++
====================

This simulation can be used for a model containing several HLS custom operations. Because they are based on finn-hlslib function, C++ code can be generated from this single nodes and they can be executed by compiling the code and running the resulting executables.


Emulation using PyVerilator
===========================

The emulation using PyVerilator can be used when IP blocks were generated, either node by node or of a whole design. For that purpose PyVerilator gets the generated verilog files.
