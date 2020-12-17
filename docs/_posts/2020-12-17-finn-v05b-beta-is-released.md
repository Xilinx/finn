---
layout: post
title:  "FINN v0.5b (beta) and finn-examples released"
author: "Yaman Umuroglu"
---

We're happy to announce that FINN v0.5b is now available on GitHub. This is
one of the most feature-rich updates we've made so far, here are the highlights:


<img src="https://xilinx.github.io/finn/img/finn-examples-header.png" width="450" align="center"/>

**New finn-examples repo for PYNQ on Zynq and Alveo.** We now have a new repo
called [finn-examples](https://github.com/Xilinx/finn-examples), which hosts
several neural network accelerators built with the FINN compiler. It comes with
prebuilt bitfiles for several platforms and PYNQ Python drivers. It's also on
pypi, so can simply `pip3 install finn-examples` on your board to try it out.

<img src="https://xilinx.github.io/finn/img/imagenet.jpg" width="300" align="center"/>

**4-bit MobileNet-v1 for Alveo U250.** As [part of finn-examples](https://github.com/Xilinx/finn-examples/blob/main/finn_examples/notebooks/2_imagenet_with_mobilenet_v1.ipynb),
we are releasing a streaming dataflow accelerator for MobileNet-v1 on ImageNet.
It uses 4-bit weights and activations (first layer is 8-bit), gets 70.4% top-1
accuracy on ImageNet-1K and runs at 1800 FPS on the Alveo U250.
You can [rebuild it from source](https://github.com/Xilinx/finn-examples/tree/main/build/mobilenet-v1)
with the scripts provided in finn-examples.

<img src="https://xilinx.github.io/finn/img/cybsec.jpg" width="300" align="center"/>

**New tutorial: train and deploy a cybersecurity MLP.** v0.5b adds a new [three-part tutorial](https://github.com/Xilinx/finn/tree/master/notebooks/end2end_example/cybersecurity)
Jupyter on first training a quantized MLP with Brevitas, then deploying this
with the new `build_dataflow` system discussed below.  Here, the example application
is classifying network packets as malicious or not by training an MLP on the
UNSW-NB15 cybersecurity dataset.

<img src="https://xilinx.github.io/finn/img/build-cfg.png" width="400" align="center"/>

**New build system.** We have introduced a new build system
and command-line entrypoint called [`build_dataflow`](https://finn.readthedocs.io/en/latest/command_line.html) for productivity and ease-of-use. With this new system,
you can analyze and compile basic QNNs without having to set up a manual
sequence of graph transformations. To control the generated accelerator
performance, you can simply set a `target_fps` and a target clock frequency
and have FINN figure out the parallelization parameters and required FIFO sizes,
or you can set them manually for a hand-optimized design. Both `finn-examples`
and the new cybersecurity tutorials use this new build system.

**Other improvements.** Those were the highlights, but there are numerous other
new features and bugfixes that are included in v0.5b. Some of the larger ones
include preliminary support for [runtime-writable weights](https://github.com/Xilinx/finn/pull/234),
[automated FIFO sizing](https://github.com/Xilinx/finn/pull/232),
the [`SetFolding` transform for setting parallelization parameters](https://github.com/Xilinx/finn/pull/251),
and the separation of the FINN compiler's core components into a new repo
called [finn-base](https://github.com/Xilinx/finn-base/) for more modularity.

The release (tagged 0.5b) is now available on GitHub.
We're continuously working to improve FINN in terms of layer, network and
infrastructure.
If you'd like to help out, please check out the <a href="https://github.com/Xilinx/finn/blob/master/CONTRIBUTING.md">contribution guidelines</a> and
share your ideas on the <a href="https://gitter.im/xilinx-finn/community">FINN Gitter channel</a>!
