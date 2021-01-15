.. _example_networks:

****************
Example Networks
****************

Please visit the `finn-examples <https://github.com/Xilinx/finn-examples>`_
repository. This repo includes prebuilt bitfiles for various PYNQ and Alveo
platforms, as well as the scripts to rebuild these examples using the FINN
compiler.


End-to-end Integration tests
============================

The FINN compiler uses `several pre-trained QNNs <https://github.com/Xilinx/brevitas/tree/master/brevitas_examples/bnn_pynq>`_
that serve as both examples and testcases.

* TFC, SFC, LFC... are fully-connected networks trained on the MNIST dataset
* CNV is a convolutional network trained on the CIFAR-10 dataset
* w\_a\_ refers to the quantization used for the weights (w) and activations (a) in bits

These networks are built end-to-end as part of the `FINN integration tests <https://github.com/Xilinx/finn/blob/master/tests/end2end/test_end2end_bnn_pynq.py>`_ ,
and the key performance indicators (FPGA resource, frames per second...) are
automatically posted to the dashboard below.
To implement a new network, you can use the `integration test code <https://github.com/Xilinx/finn/blob/dev/tests/end2end/test_end2end_bnn_pynq.py>`_
as a starting point, as well as the `relevant Jupyter notebooks
<https://github.com/Xilinx/finn/tree/master/notebooks/end2end_example/bnn-pynq>`_.

.. image:: https://firebasestorage.googleapis.com/v0/b/drive-assets.google.com.a.appspot.com/o/Asset%20-%20Drive%20Icon512.png?alt=media
  :width: 50px
  :align: left

`FINN end-to-end integration tests dashboard on Google Drive <https://bit.ly/finn-end2end-dashboard>`_
