---
layout: post
title:  "FINN v0.4b (beta) is released"
author: "Yaman Umuroglu"
---

Version v0.4b (beta) of the FINN compiler is now available. As with the previous
release there's a whole lot of new features and bugfixes that have gone in,
but here are some highlights:

<img src="https://mma.prnewswire.com/media/752936/ALVEO_PRESS.jpg" width="300" align="center"/>

**Build support for Alveo/Vitis + more Zynq variants.** We now have a
`VitisBuild` transformation to provide a FINN flow that goes all the way to
bitfiles targeting Xilinx Alveo platforms. This transformation takes care of
FIFO, datawidth converter and DMA engine insertion so you can simply give it a
FINN model with HLS layers and let it run.
Similarly, we've simplified the Zynq build flow with `ZynqBuild` to provide a
similar experience, which should now be able to support most Zynq and Zynq
UltraScale+ platforms.
You can read more about the new hardware build transformations
<a href="https://finn.readthedocs.io/en/latest/hw_build.html">here</a>.

<img src="https://xilinx.github.io/finn/img/finn-dashboard.png" width="300" align="center"/>

**Fully-accelerated end-to-end examples + dashboard.** The FINN end-to-end example networks
are now fully accelerated on the FPGA, allowing raw images to be directly fed in
and top-1 indices to be retrieved.
We now also have a <a href="https://bit.ly/finn-end2end-dashboard">dashboard</a>
which gets automatically updated with the latest build results from end-to-end
examples, including FPGA resources and performance.
This also enables running full-performance accuracy validation on hardware,
which is now incorporated into the <a href="https://github.com/Xilinx/finn/blob/master/notebooks/end2end_example/tfc_end2end_example.ipynb#validation">end-to-end notebooks</a>.

<img src="https://xilinx.github.io/finn/img/finn-brevitas-debug.png" width="300" align="center"/>

**Brevitas-FINN co-debug support.** We can now export graphs from Brevitas with special DebugMarker nodes (like above) and PyTorch forward hooks to compare intermediate activations between the Brevitas version and FINN-ONNX exported version. This is handy for debugging especially larger networks when they don't export correctly. <a href="https://github.com/Xilinx/finn/blob/dev/tests/brevitas/test_brevitas_debug.py">Here</a> is an example of how to use this.

<img src="https://xilinx.github.io/finn/img/accumulator-minimization.png" width="300" align="center"/>

**Accumulator minimization.** When converting to HLS layers, FINN will now automatically try to pick a minimal bitwidth for each accumulator, based on the precision and size of the dot product it accumulates over. While prior accumulators were at a fixed bitwidth like
32-bits, the new approach can significantly save on resources by picking e.g. 10-bit accumulators (as per above) where possible. We've also expanded the range of DataTypes available in FINN to cover everything between 1-32 bits to provide more flexibility.

<img src="https://xilinx.github.io/finn/img/finn-cycle-estimate.png" width="300" align="center"/>

**New layers and cycle estimation.** We've been working on supporting more of the finn-hlslib layers in FINN and
<a href="https://github.com/Xilinx/finn/tree/dev/src/finn/custom_op/fpgadataflow">the list</a>  has expanded significantly.
Many of these layers (and their accompanying conversion transformations) will be utilized for new FINN end-to-end example networks,
like MobileNet-v1, ResNet-50 and a QuartzNet, over the course of the next few releases. These layers also support <a href="https://github.com/Xilinx/finn/blob/dev/src/finn/analysis/fpgadataflow/exp_cycles_per_layer.py">clock cycle estimation</a>
based on workload and parallelization parameters, allowing the user to estimate performance without having to go to synthesis.

The release (tagged 0.4b) is now available on GitHub.
We're continuously working to improve FINN in terms of layer, network and
infrastructure.
If you'd like to help out, please check out the <a href="https://github.com/Xilinx/finn/blob/master/CONTRIBUTING.md">contribution guidelines</a> and
share your ideas on the <a href="https://gitter.im/xilinx-finn/community">FINN Gitter channel</a>!
