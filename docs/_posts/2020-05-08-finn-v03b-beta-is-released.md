---
layout: post
title:  "FINN v0.3b (beta) is released"
author: "Yaman Umuroglu"
---

We're happy to announce the v0.3b (beta) release of the FINN compiler.
The full changelog is quite large as we've been working on a lot of exciting
new features, but here is a summary:

<img src="https://xilinx.github.io/finn/img/cnv-mp-fc.png" width="800" align="center"/>


**Initial support for ConvNets and end-to-end notebook example.** The
preliminary support for convolutions is now in place. Head over to the new
<a href="https://github.com/Xilinx/finn/blob/staging/v0.3b/notebooks/end2end_example/cnv_end2end_example.ipynb">
end-to-end notebook</a> to try out the end-to-end flow for convolutions
and build the demonstrator for a simple binarized CNN on CIFAR-10.

<img src="https://xilinx.github.io/finn/img/parallel-speedup.png" width="500" align="center"/>

**Parallel transformations.** When working with larger designs, HLS synthesis
and simulation compile times can be quite long. Thanks to a contribution by
@HenniOVP we now support multi-process parallelization several FINN transformations.
You can read more about those <a href="https://github.com/Xilinx/finn/blob/staging/v0.3b/notebooks/advanced/1_custom_transformation_pass.ipynb">here</a>.

<img src="https://xilinx.github.io/finn/finn/img/mem_mode.png" width="600" align="center"/>

**Decoupled memory mode for MVAUs.** To have more control over how the weight
memories are implemented, you can now specify the `mem_mode` and `ram_style`
attributes when instantiating compute engines. Read more <a href="https://finn.readthedocs.io/en/latest/internals.html#streamingfclayer-mem-mode">here.</a>

**Throughput testing and optimizations.** To do a quick assessment of the
customized accelerators you build, we now support a throughput test mode that
lets you benchmark the accelerator with a configurable number of samples.
To get better utilization from the heterogeneous streaming architectures FINN
builds, we have also introduced a FIFO insertion transformation.
You can see these in action in the updated <a href="https://github.com/Xilinx/finn/blob/staging/v0.3b/notebooks/end2end_example/tfc_end2end_example.ipynb">
TFC-w1a1 end2end notebook.</a>

We have a slew of other smaller features, bugfixes and various other improvements.
The release (tagged 0.3b) is now available on GitHub.
We're continuously working to improve FINN in terms of layer, network and
infrastructure.
If you'd like to help out, please check out the <a href="https://github.com/Xilinx/finn/blob/staging/v0.3b/CONTRIBUTING.md">contribution guidelines</a> and
share your ideas on the <a href="https://gitter.im/xilinx-finn/community">FINN Gitter channel</a>!
