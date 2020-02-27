.. _tutorials:

*********
Tutorials
*********

.. note:: **This website is currently under construction.**

FINN provides several Jupyter notebooks that can help to get familiar with the basics, the internals and the end-to-end flow in FINN. All Jupyter notebooks can be found in the repo in the `notebook folder <https://github.com/Xilinx/finn/tree/dev/notebooks>`_.

Basics
======

The notebooks in this folder should give a basic insight into FINN, how to get started and the basic concepts.

* `0_getting_started <https://github.com/Xilinx/finn/blob/dev/notebooks/basics/0_getting_started.ipynb>`_
  
  * This notebook corresponds to the chapter :ref:`getting_started` and gives an overview how to start working with FINN.

* `1_how_to_work_with_onnx <https://github.com/Xilinx/finn/blob/dev/notebooks/basics/1_how_to_work_with_onnx.ipynb>`_

  * This notebook can help you to learn how to create and manipulate a simple ONNX model, also by using FINN

* `2_modelwrapper <https://github.com/Xilinx/finn/blob/dev/notebooks/basics/2_modelwrapper.ipynb>`_

  * This notebook corresponds to the section :ref:`modelwrapper` in the chapter about internals.

* `3_brevitas_network_import <https://github.com/Xilinx/finn/blob/dev/notebooks/basics/3_brevitas_network_import.ipynb>`_

  * This notebook shows how to import a brevitas network and prepare it for the FINN flow.

Internals
=========

The notebooks in this folder are more developer oriented. They should help you to get familiar with the principles in FINN and how to add new content regarding these concepts.

* `0_custom_analysis_pass <https://github.com/Xilinx/finn/blob/dev/notebooks/internals/0_custom_analysis_pass.ipynb>`_

  * This notebook explains what an analysis pass is and how to write one for FINN.

* `1_custom_transformation_pass <https://github.com/Xilinx/finn/blob/dev/notebooks/internals/1_custom_transformation_pass.ipynb>`_

  * This notebook explains what a transformation pass is and how to write one for FINN.

* `2_custom_op <https://github.com/Xilinx/finn/blob/dev/notebooks/internals/2_custom_op.ipynb>`_

  * This notebooks explains what a custom operation/node is and how to create one for FINN.

* `3_verify_hls_custom_op <https://github.com/Xilinx/finn/blob/dev/notebooks/internals/3_verify_hls_custom_op.ipynb>`_

  * This notebook shows the functional verification flow for hls custom operations/nodes.

End-to-End Flow
===============

This notebook shows the FINN end-to-end flow step by step using an example of a simple, binarized, fully-connected network trained on the MNIST data set. Starting with the brevitas export and taking this particular network all the way down to hardware by using a specific sequence of transformations.

* `tfc_end2end_example <https://github.com/Xilinx/finn/blob/dev/notebooks/end2end_example/tfc_end2end_example.ipynb>`_


