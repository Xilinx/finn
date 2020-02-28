.. _verification:

***********************
Functional Verification
***********************

.. note:: **This website is currently under construction.**

.. image:: /img/verification.png
   :scale: 70%
   :align: center

"* This part of the flow is covered by the `this notebook <https://github.com/Xilinx/finn/tree/master/notebooks/end2end_example/tfc_end2end_verification.ipynb>`_. *"

When the network is transformed it is important to verify the functionality to make sure the transformation did not change the behaviour of the model. There are multiple ways of verification that can be applied in different stages of the network inside FINN. All can be accessed using the execution function in module :py:mod:`finn.core.onnx_exec`. The execution happens in most cases node by node, which supports networks that have a mixture of standard ONNX nodes, custom nodes and HLS custom nodes. A single node can be executed using one or more of the following methods:

Simulation using Python
=======================

This simulation can be used right after the :ref:`brevitas_export` or when the network does not contain any HLS custom nodes, so right after the streamlining transformations and before the nodes are converted into HLS layers.

Simulation using C++
====================

This simulation can be used for a model containing several HLS custom operations. Because they are based on finn-hlslib function, C++ code can be generated from this single nodes and they can be executed by compiling the code and running the resulting executables.


Emulation using PyVerilator
===========================

The emulation using PyVerilator can be used when IP blocks were generated, either node by node or of a whole design. For that purpose PyVerilator gets the generated verilog files.
