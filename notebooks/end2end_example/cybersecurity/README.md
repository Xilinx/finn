# Training and Deploying a Quantized MLP

In this folder you will find a series of notebooks that guide you through
the process of training a highly quantized neural network (QNN) and generating
a high-performance streaming dataflow accelerator from it using the FINN
compiler.
If you'd like to train your own QNNs and deploy them using FINN, this is a
good starting point.

Here, the example application is classifying network packets as malicious or
not by training a multi-layer perceptron (MLP) on the UNSW-NB15 dataset.
We recommend following these notebooks in the order they appear:

1. Training a few-bit MLP on the UNSW-NB15 dataset
2. Exporting the trained network and verify that it works as intended
3. Generating a streaming dataflow accelerator using the FINN compiler

Note: This tutorial abstract away the internal details of the steps to provide
a simpler introduction. If you'd like to understand more of the internal
details of what happens during the accelerator build, we recommend the
(BNN-PYNQ end-to-end notebooks)[../bnn-pynq].
