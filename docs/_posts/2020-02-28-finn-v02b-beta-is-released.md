---
layout: post
title:  "FINN v0.2b (beta) is released"
author: "Yaman Umuroglu"
---

We've been working on the new version of the FINN compiler for a while, and today we are excited to announce our first beta release to 
give you a taste of how things are shaping up! 

Here's a quick overview of the key features:

* <b>Train and export highly-quantized networks in PyTorch using Brevitas.</b> You can use <a href="https://github.com/Xilinx/brevitas">Brevitas</a>,
  our PyTorch library for quantization-aware training to train networks with few-bit weights and activations, then export them into 
  FINN-ONNX to be used by the FINN compiler.

* <b>Fully transparent end-to-end flow.</b> We support taking quantized networks (with limitations, see bottom of post) all the way down to a 
  customized FPGA bitstream. This happens across many steps ranging from streamlining to Vivado IPI stitching, and each step is fully 
  visible to the user. So if you are happy with just the threshold-activation (streamlined) QNN in ONNX, or if you want to take the 
  generated Vivado IP block and integrate it into your own IPI design, it's easy to break out of the flow at any step. 
  We also provide a variety of mechanisms to verify the design at different steps.

* <b>ONNX-based intermediate representation.</b> We use ONNX with some custom nodes and annotations as our intermediate representation. As the 
  FINN compiler transforms the network across many steps to produce an FPGA bitstream, you can view and explore the transformed network 
  using the excellent <a href="https://www.lutzroeder.com/ai/netron">Netron</a> viewer from the comfort of your web browser.

* Tutorials and documentation. We have prepared a set of <a href="https://github.com/Xilinx/finn/tree/master/notebooks">Jupyter notebooks</a> 
  to let you experiment with some of the things FINN can do, covering the basics, demonstrating the end-to-end flow on an example network, 
  and discussing some of the internals for more advanced users and developers. We also have Sphinx-generated documentation on 
  <a href="http://finn.readthedocs.io/">readthedocs</a> for more information on the FINN compiler and its API.

The release (tagged 0.2b) is now available on GitHub. Currently it's a beta release and only supports fully-connected layers in linear 
(non-branching) topologies, but we're actively working on the end-to-end convolution support for the next release. Further down the 
road, we hope to support more advanced topologies and provide end-to-end examples for MobileNet and ResNet-50.
