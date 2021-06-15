---
layout: post
title:  "FINN v0.6 is released"
author: "Yaman Umuroglu, Hendrik Borras"
---

*Important: Due to changes in the board definitions, we recommend deleting the “board\_files” folder created by FINN after updating, or alternatively 
making a fresh clone of the repository. For more information check out the related [issue](https://github.com/Xilinx/finn/issues/341).*          

We are delighted to announce the release of FINN v0.6! Below is a summary of the highlights from this release.

### **Advertisement: AI for Good challenge**

Want to put your skills in creating hardware-efficient DNNs to good use for the future of wireless communications? 
Join our competition “Lightning-Fast Modulation Classification with Hardware-Efficient Neural Networks” organized as part of the ITU AI/ML in 5G Challenge! 
Attend the [challenge kickoff webinar](https://aiforgood.itu.int/events/lightning-fast-modulation-classification-with-hardware-efficient-neural-networks/) to find out more.

### New ImageNet networks in finn-examples

We have added two new CNNs on ImageNet to [finn-examples](https://github.com/Xilinx/finn-examples): a binary-weights, 2-bit-activations ResNet-50 on U250 
and a port of the 4-bit MobileNet-v1 to ZCU104. The U250 ResNet-50 delivers ~3000 FPS at ~68.9% top-1 accuracy, while the ZCU104 MobileNet-v1 achieves 
~450 FPS at 70.4%. The end-to-end build flows for these networks can be also found in the [build folder](https://github.com/Xilinx/finn-examples/tree/main/build)
of finn-examples, which is made possible by some of the new features listed below.

### Support for weights in URAM and DDR

FINN now supports significantly larger weights by being able to place them into URAMs or DDR, by marking the layer with
`mem_mode="decoupled" + ram_type="ultra"` for URAM and `mem_mode="external"` for DRAM.
For both of these cases, there is some runtime support necessary, which is incorporated into the generated Python PYNQ driver.
The URAM support is showcased by the MobileNet-v1 on the ZCU104 and the ResNet-50 on the U250.

### finn-experimental: double-packed DSPs and partitioning support

We now have a new FINN module called [finn-experimental](https://github.com/Xilinx/finn-experimental), which will house experimental features and
plugins for FINN that aren't fully tested but still useful for research.
Among others, this currently includes support for double-packing 8-bit operations into DSP slices and partitioning large graphs into either 
SLRs or multi-FPGAs for scaling to larger designs.
Both of these features are showcased by the ResNet-50 example.

### Updated Documentation, Tutorials and FAQ
Since good documentation is always important for newcomers and veterans alike, we are continuously improving our documentation at
[here](https://finn.readthedocs.io).
We’d like to the [FAQ section](https://finn.readthedocs.io/en/latest/faq.html) for quick answers to frequent questions,
and the [new tutorial notebook](https://github.com/Xilinx/finn/blob/master/notebooks/advanced/2_custom_op.ipynb) for custom operations,
where you can learn how to build a custom operation in FINN from the ground up.
We have also updated the [cybersecurity tutorial](https://github.com/Xilinx/finn/tree/master/notebooks/end2end_example/cybersecurity)  
to include feedback from our recent workshops. 

### Updates to the bundled versions of PyTorch and Brevitas
The bundled PyTorch and Brevitas versions for the FINN Docker are now updated to 1.7.0 and 0.5.1, respectively.
This will make it easier for users to train and deploy their networks directly within the same FINN docker container. 
As such the interoperability between Brevitas and FINN has become even better, with access to Brevitas features out of the box.

### Initial support for 1D convolutions

Thanks to our collaborators Mirza Mrahorovic and Jakoba Petri-Koenig from TU Delft, we now have initial support for 1D convolutions in FINN.
These convolutions find widespread use for machine learning with time-series data, such as digital signal processing. 
This now gives FINN a much wider range of possible applications, where low-latency and high-throughput are key.
We will be showcasing the support for 1D networks with new examples soon, but you can already see an example of how this works in 
[this testcase](https://github.com/Xilinx/finn/blob/master/tests/fpgadataflow/test_convert_to_hls_1d_conv_layer.py)

The release ([tag v0.6](https://github.com/Xilinx/finn/tree/v0.6)) is now available on GitHub.
We're continuously working to improve FINN in terms of layer, network and
infrastructure.
If you'd like to help out, please check out the <a href="https://github.com/Xilinx/finn/blob/master/CONTRIBUTING.md">contribution guidelines</a> and
share your ideas on the <a href="https://gitter.im/xilinx-finn/community">FINN Gitter channel</a>!
