---
layout: post
title:  "Line-rate intrusion detection demo"
author: "Yaman Umuroglu"
---

As we become more and more reliant on staying connected for both our work and personal lives, cybersecurity becomes increasingly important to protect our systems and data from attacks. Intrusion detection systems, which monitor a network or system for malicious activity or policy violations, are an important part of cybersecurity. These systems were traditionally crafted using sets of rules determined by experts, but the growth of network data volume and more complicated attacks make this increasingly difficult. 

[Machine learning](https://proceedings.neurips.cc/paper/1997/file/1abb1e1ea5f481b589da52303b091cbb-Paper.pdf) has been proposed as a solution, training neural networks and other ML algorithms on both real-life and synthetic datasets to better adapt the intrusion detection systems to the changing requirements.
However, this can come at the cost of performance. To avoid introducing bottlenecks on the network, the NN implementation must be capable of detecting malicious packets at line-rate, which can be hundreds of millions of packets per second, and is expected to increase further as next-generation networking solutions provide increased throughput. Can we build NN-based intrusion detection systems that can provide line-rate through neural-network hardware-codesign? With FINN, the answer is yes! 

In this demo, we take a [quantized version](https://www.researchgate.net/profile/Tadej-Murovic/publication/333563328_Massively_parallel_combinational_binary_neural_networks_for_edge_processing/links/5d01fd92299bf13a38511bca/Massively-parallel-combinational-binary-neural-networks-for-edge-processing.pdf) of the [UNSW-NB15 cybersecurity dataset](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/) and train a 2-bit quantized MLP on this dataset with [Brevitas](https://github.com/Xilinx/brevitas) that gets close to 92% accuracy. Then we use the FINN compiler to generate a fully-unrolled FPGA hardware implementation for this network that can classify incoming data at clock rate (300 million packets per second), demonstrating the performance on a Xilinx ZCU104 board.

You can find the links to the demo videos below:
 * Part 1: quantized MLP training and FINN compiler toolflow [https://www.youtube.com/watch?v=z49tzp3CBoM](https://www.youtube.com/watch?v=z49tzp3CBoM)
 * Part 2: hardware demo [https://www.youtube.com/watch?v=W35c5XmnlhA](https://www.youtube.com/watch?v=W35c5XmnlhA)
 
The line-rate (300 MHz inference throughput) version of the demo will be available as part of the FINN v0.6 release, but we already have tutorial material that explains all the steps involved for a simpler design (1 MHz inference throughput) that you can find [here](https://github.com/Xilinx/finn/tree/feature/tutorial_march21/notebooks/end2end_example/cybersecurity).
