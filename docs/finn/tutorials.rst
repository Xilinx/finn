.. _tutorials:

*********
Tutorials
*********

.. note:: **This website is currently under construction.**

FINN provides several Jupyter notebooks that can help to get familiar with the basics, the internals and the end-to-end flow in FINN. All Jupyter notebooks can be found in the repo in the `notebook folder <https://github.com/Xilinx/finn/tree/master/notebooks>`_.

Basics
======

The notebooks in this folder should give a basic insight into FINN, how to get started and the basic concepts.

* 0_how_to_work_with_onnx

  * This notebook can help you to learn how to create and manipulate a simple ONNX model, also by using FINN

* 1_brevitas_network_import

  * This notebook shows how to import a brevitas network and prepare it for the FINN flow.

Advanced
========

The notebooks in this folder are more developer oriented. They should help you to get familiar with the principles in FINN and how to add new content regarding these concepts.

* 0_custom_analysis_pass

  * This notebook explains what an analysis pass is and how to write one for FINN.

* 1_custom_transformation_pass

  * This notebook explains what a transformation pass is and how to write one for FINN.

End-to-End Flow
===============

This notebook shows the FINN end-to-end flow step by step using an example of a simple, binarized, fully-connected network trained on the MNIST data set. Starting with the brevitas export and taking this particular network all the way down to hardware by using a specific sequence of transformations.

* cnv_end2end_example

  * This notebook takes a simple convolutional model step-by-step from a trained Brevitas net to a running FPGA bitfile.

* tfc_end2end_example

  * This notebook takes a simple fully-connected  model step-by-step from a trained Brevitas net to a running FPGA bitfile.

* tfc_end2end_verification

  * This notebook runs parellel to the tfc_end2end_example notebook above, and shows how the output of each step can be verified.
