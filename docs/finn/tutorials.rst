.. _tutorials:

*********
Tutorials
*********

FINN provides several Jupyter notebooks that can help to get familiar with the basics, the internals and the end-to-end flow in FINN.
All Jupyter notebooks can be found in the repo in the `notebook folder <https://github.com/Xilinx/finn/tree/main/notebooks>`_.

Basics
======

The notebooks in this folder should give a basic insight into FINN, how to get started and the basic concepts.

* 0_how_to_work_with_onnx

  * This notebook can help you to learn how to create and manipulate a simple ONNX model, also by using FINN

* 1_brevitas_network_import_via_QONNX

  * This notebook shows how to import a Brevitas network and prepare it for the FINN flow.

End-to-End Flow
===============

There are two groups of notebooks currently available under `the end2end_example directory <https://github.com/Xilinx/finn/tree/main/notebooks/end2end_example>`_ :

* ``cybersecurity`` shows how to train a quantized MLP with Brevitas and deploy it with FINN using the :ref:`command_line` build system.

* ``bnn-pynq`` shows the internal compiler steps that take pretrained Brevitas QNNs on MNIST and CIFAR-10 and generate the FPGA accelerator.


Advanced
========

The notebooks in this folder are more developer oriented. They should help you to get familiar with the principles in FINN and how to add new content regarding these concepts.

* 0_custom_analysis_pass

  * Explains what an analysis pass is and how to write one for FINN.

* 1_custom_transformation_pass

  * Explains what a transformation pass is and how to write one for FINN.

* 2_custom_op

  * Explains the basics of FINN custom ops and how to define a new one.

* 3_folding

  * Describes the use of FINN parallelization parameters (PE & SIMD), also called folding factors, to efficiently optimize models so as to extract the maximum performance out of them.

* 4_advanced_builder_settings

  * Provides a more detailed look into the FINN builder tool and explores different options to customize your FINN design.


FINN Example FPGA Flow Using MNIST Numerals
============================================

Next to the Jupyter notebooks above there is a tutorial about the command-line build_dataflow `here <https://github.com/Xilinx/finn/tree/main/tutorials/fpga_flow>`_ which shows how to bring a FINN compiled model into the Vivado FPGA design environment.
