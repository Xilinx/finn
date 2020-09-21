.. _example_networks:

****************
Example Networks
****************

FINN uses <a href="https://github.com/Xilinx/brevitas/tree/master/brevitas_examples/bnn_pynq">
several pre-trained QNNs</a>
that serve as examples and testcases.

* TFC, SFC, LFC... are fully-connected networks trained on the MNIST dataset
* CNV is a convolutional network trained on the CIFAR-10 dataset
* w\_a\_ refers to the quantization used for the weights (w) and activations (a) in bits

These networks are built end-to-end as part of the <a href="https://github.com/Xilinx/finn/blob/master/tests/end2end/test_end2end_bnn_pynq.py">FINN integration tests</a>,
and the key performance indicators (FPGA resource, frames per second...) are
automatically posted to the dashboard below.
To implement a new network, you can use the <a href="https://github.com/Xilinx/finn/blob/dev/tests/end2end/test_end2end_bnn_pynq.py">
integration test code</a> as a starting point, as well as the relevant
<a href="https://github.com/Xilinx/finn/tree/master/notebooks/end2end_example">
Jupyter notebooks.</a>

<a href="https://bit.ly/finn-end2end-dashboard">
  <img src="https://firebasestorage.googleapis.com/v0/b/drive-assets.google.com.a.appspot.com/o/Asset%20-%20Drive%20Icon512.png?alt=media" width="50" align="center" />
  FINN end-to-end dashboard on Google Drive
</a>
