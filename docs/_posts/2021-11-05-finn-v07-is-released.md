---
layout: post
title:  "FINN v0.7 is released"
author: "Yaman Umuroglu"
---

*Important: The default branch has been renamed from `master` to `main`. You may want to make a fresh clone of the [FINN compiler repository](https://github.com/Xilinx/finn/) if you have a local copy.*

We are delighted to announce the release of FINN v0.7! Below is a summary of the highlights from this release.

### A new, flexible input representation: QONNX

FINN is taking another step towards increasing the flexibility of the framework by supporting a new input format for neural networks, called QONNX.
QONNX will enable FINN to be much more flexible in terms of representing weight and activation quantization, especially for higher precisions and fixed-point datatypes. For instance, this will enable future support for higher-precision quantized weights and activations, avoiding streamlining difficulties and expensive MultiThreshold-based activations.
QONNX is being developed in close collaboration with [hls4ml](https://github.com/fastmachinelearning/hls4ml) and will enable closer collaboration between FINN and hls4ml.
Our long-time collaborator and current intern Hendrik Borras wrote a [blog post](https://xilinx.github.io/finn//2021/11/03/qonnx-and-finn.html) about QONNX that has a lot more details, including a Netron interactive visualization of an example network in this new format.

### Three new models in finn-examples

Our examples repository [finn-examples](https://github.com/Xilinx/finn-examples) has been growing with new  contributions, showcasing what our community is doing with FINN and dataflow NN accelerators on FPGAs. In particular,
we have three new demos:

* **[Face mask wear and positioning:](https://github.com/Xilinx/finn-examples/blob/main/finn_examples/notebooks/3_binarycop_mask_detection.ipynb)** Low-power BNN classifier for Pynq-Z1 for correct face mask wear and positioning. Contributed by TU Munich and BMW.
* **[Radio signal modulation:](https://github.com/Xilinx/finn-examples/blob/main/finn_examples/notebooks/5_radioml_with_cnns.ipynb)** Classify RadioML 2018.1 at 250k inferences/second on a ZCU104. See also our [ITU challenge](https://bit.ly/brevitas-radioml-challenge-21) on the topic. Contributed by Felix Jentzsch.
* **[Keyword spotting:](https://github.com/Xilinx/finn-examples/blob/main/finn_examples/notebooks/4_keyword_spotting.ipynb)** Classify spoken commands from the Google Speech Commands dataset at 250k inferences/second on a Pynq-Z1. Utilizes the new QONNX build flow. Contributed by Hendrik Borras.

As always, you can get these accelerators alongside their Jupyter notebooks on your PYNQ board or Alveo U250 with `pip3 install finn-examples`. 

### Infrastructure improvements

As the FINN community grows, it becomes increasingly important to future-proof the various pieces of infrastructure that makes the FINN compiler work. Here are a few of the infrastructure improvements that went into this release:

* **Offline Docker image support:** Prebuilt FINN Docker images are now available on [DockerHub](https://hub.docker.com/r/maltanar/finn), so it's easier than ever to bring up your environment, or if you need a completely offline setup (with once-off image download). [Here](https://finn-dev.readthedocs.io/en/latest/getting_started.html#using-a-prebuilt-image) are the instructions for using prebuilt Docker images.
* **Improved `DataType` system:** Previously, FINN's `DataType` support was a not-quite-exhaustive enumeration of some possible values, which limits the flexibility of what the compiler can do. We now have a new system in place that supports arbitrary-precision integers as well as fixed-point types, allowing the expression of things like `DataType["UINT71"]` and `DataType["FIXED<9,3>"]`. The compiler flows to actually take advantage of these types in end-to-end flows will be coming in the near future.
* **New layer support:** We now HLS layer support for [upsampling](https://github.com/Xilinx/finn/blob/main/src/finn/custom_op/fpgadataflow/upsampler.py) and [embedded lookup](https://github.com/Xilinx/finn/blob/main/src/finn/custom_op/fpgadataflow/lookup.py) in place.


### New support channels

Nowadays we're getting lots of support requests, and though our [Gitter channel](https://gitter.im/xilinx-finn/community) is alive and well we wanted to make it easier to organize discussions, find answers and react to posts.
Towards that end, the primary support channel for FINN is now on [GitHub discussions](https://github.com/Xilinx/finn/discussions). The [Frequently Asked Questions](https://finn.readthedocs.io/en/latest/faq.html) and [Getting Started](https://finn.readthedocs.io/en/latest/getting_started.html) sections in the documentation have also seen major updates.


The release ([tag v0.7](https://github.com/Xilinx/finn/tree/v0.7)) is now available on GitHub.
We're continuously working to improve FINN in terms of layer, network and
infrastructure.
If you'd like to help out, please check out the <a href="https://github.com/Xilinx/finn/blob/master/CONTRIBUTING.md">contribution guidelines</a> and
share your ideas on the <a href="https://github.com/Xilinx/finn/discussions">FINN GitHub Discussions</a>!
