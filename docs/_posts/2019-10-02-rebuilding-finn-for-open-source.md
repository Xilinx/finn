---
layout: post
title:  "Rebuilding FINN for open source"
author: "Yaman Umuroglu"
---

We're happy to announce some exciting developments in the FINN project: we're rebuilding our solution stack from the ground up
to be more modular, more usable and more open-source!

### A quick retrospective

Over the past few years, the team at Xilinx Research Labs Ireland has done quite a bit of research of Quantized Neural Networks
(QNNs).
Starting with <a href="https://arxiv.org/abs/1612.07119">Binarized Neural Networks (BNNs) on FPGAs</a> back in 2016, we've since 
looked at many aspects of quantized deep learning, ranging from
at <a href ="https://arxiv.org/abs/1807.00301">better quantization methods</a> and
<a href="https://arxiv.org/abs/1709.06262">mixing quantization and pruning</a>, 
to <a href="https://arxiv.org/pdf/1807.10577.pdf">accuracy-throughput tradeoffs</a> and
<a href="https://arxiv.org/pdf/1807.04093.pdf">recurrent topologies</a>.

Although some <a href="https://github.com/Xilinx/BNN-PYNQ">demonstrators</a> of our work has been open source for some time, 
we want to take things a step further.
We love QNNs and the high-performance, high-efficiency dataflow accelerators we can build for them on Xilinx FPGAs, and we want you and 
the FPGA/ML community to be able to do the same.
The (co-)design process for making this happen is actually quite involved, starting from customizing a neural network in a machine
learning framework, going through multiple design steps that involve many optimizations, HLS code generation and Vivado synthesis, and 
ending up with an FPGA bitstream that you can deploy as part of some application.
Many of those steps require some manual effort, but having a modular, flexible solution stack to support you through this process is greatly 
helpful.
This is why we are rebulding our FINN solution stack from the ground-up to make it more modular, and we hope to build a community
around it that shares our excitement around QNNs for FPGAs.

### Making FINN modular

<img align="left" src="img/finn-stack.png" alt="drawing" style="margin-right: 20px" width="300"/>

The first step towards making this happen is to define what layers exist in the solution stack.
In many ways, this solution stack is inspired by the tested-and-tried frontend/backend software architecture found in compiler
frameworks like <a href="http://llvm.org">LLVM</a>.
This stack breaks down the complex co-design problem into parts, and each layer focuses on a different sub-problem, consuming
the artifacts produced by the previous one.
The diagram on the left illustrates this briefly, and over the next few months we hope to make a first few QNNs go through all
the layers of this stack to produce cool FPGA dataflow accelerators. 
In fact, some of these components are already available today for you to explore!

Let's have a look at the main parts:

* <b>Brevitas</b> is a PyTorch library that lets you do quantization-aware training. It gives you a set of `torch.nn` building
blocks to explore different forms of weight, activation and accumulator quantization schemes. You can also learn the bitwidths for 
different layers with backpropagation! See the <a href="https://xilinx.github.io/brevitas/">Brevitas page</a> for more information.
* <b>Frontend</b>. Once you are happy with the accuracy of your quantized neural network in Brevitas, you'll be able to export it into a custom 
<a href="https://onnx.ai">ONNX</a> representation that FINN uses internally to represent QNNs. More details about this custom ONNX
representation will be available in an upcoming blog post.
* The <b>FINN Compiler</b> will then import this ONNX representation, and go through several steps of optimizations such as the
  <a href="https://arxiv.org/pdf/1709.04060.pdf">streamlining transform</a> to make the QNN simpler. 
* The <b>FPGA dataflow backend</b> will then convert the optimized QNN into a series of streaming HLS library calls. An important
 part of the stack is the <a href="https://github.com/Xilinx/finn-hlslib">FINN HLS library</a>, which provides optimized Vivado HLS 
 descriptions of several common layer types (convolutions, thresholding, pooling...) found in QNNs.
 * <b>Synthesis</b>. Once the HLS calls are generated, the next steps are to call Vivado HLS and Vivado to generate a bitstream for the target
 Xilinx FPGA. We have plans to support Vivado IPI block design code generation as well for increased agility and modularity.
 * <b>PYNQ deployment</b>. Finally, you will be able to use any of the supported <a href="http://www.pynq.io/">PYNQ</a> platforms to directly call the
 generated accelerator from Python and integrate it with other functionality. Since FINN-generated dataflow accelerators expose
 streaming interfaces, we think it will be exciting to use streaming-oriented Python frameworks such as 
 <a href="https://github.com/ray-project/ray">Ray</a> to create heterogeneous, high-performance task graphs incorporating QNNs.
 
 ### Getting started
 
 More will be available in the coming weeks and months, but if you want to get your hands dirty there's already plenty to start with!
 If you haven't done so already, we recommend starting with <a href="https://github.com/Xilinx/BNN-PYNQ">BNN-PYNQ</a> to see what
 dataflow QNN accelerators look and feel like.
 You can also start experimenting with <a href="https://xilinx.github.io/brevitas/">Brevitas</a> to train some QNNs, or
 put together a streaming pipeline with the <a href="https://github.com/Xilinx/finn-hlslib">FINN HLS library</a>.
 We have also created a <a href="https://gitter.im/xilinx-finn/community">Gitter channel</a> to make it easier to get in touch with
 the community, and hope to see many of you there! :)
 
