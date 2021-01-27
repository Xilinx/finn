---
layout: post
title:  "FINN tutorial at FPGA'21"
author: "Yaman Umuroglu"
---

*Please check back regularly for updates*

We're delighted to announce a two-hour FINN tutorial as part of the [FPGA'21 conference](https://www.isfpga.org).
Details are as follows:

* **Date:** 28 February 2021 (Sunday)
* **Time:**
    * Pacific Standard Time (PST) 10:30 AM – 1:30 PM
    * Central European Time (CET) 19:30 PM – 21:30 PM
* **Format:** Virtual event, Zoom call with hands-on Jupyter notebook lab
* **Registration:**
    * FPGA'21 conference registration: https://www.isfpga.org/registration
    * **Important:** You must fill out [this form](https://forms.gle/Cm9LwoeSjeoetYqX7) in addition to the FPGA conference registration.
    * The hands-on part is limited to 40 participants.

## Description

Mixing machine learning into high-throughput, low-latency edge applications needs co-designed solutions to meet the performance requirements. Quantized Neural Networks (QNNs) combined with custom FPGA dataflow implementations offer a good balance of performance and flexibility, but building such implementations by hand is difficult and time-consuming.

In this tutorial, we will introduce FINN, an open-source experimental framework by Xilinx Research Labs to help the broader community explore QNN inference on FPGAs. Providing a full-stack solution from quantization-aware training to bitfile, FINN generates high-performance dataflow-style FPGA architectures customized for each network. Participants will be introduced to efficient inference with QNNs and streaming dataflow architectures, the components of the project’s open-source ecosystem, and gain hands-on experience training a quantized neural network with Brevitas and deploying it with FINN.

## Practical Information

Some prior knowledge of FPGAs, Vivado HLS, PyTorch and neural network training is recommended, but not required.

This will be a virtual event, with a Zoom video call and a hands-on Jupyter notebook lab.
Registered participants will get access to a FINN setup running in the cloud.
There are no special technical requirements besides a browser and Zoom client.

Connect with us and the other participants on the [tutorial Gitter channel](https://gitter.im/xilinx-finn/tutorial-fpga21),
or join the [FINN Gitter channel](https://gitter.im/xilinx-finn/community).

## Agenda

* Part I: Introduction
    * An introduction to FINN, QNNs and streaming dataflow architectures
    * The FINN open-source community
    * Tour of FINN GitHub repositories

* Part II: Hands-on lab: Training and deploying an MLP for network intrusion detection
    * Training a quantized MLP on the UNSW-NB15 dataset with Brevitas
    * Exporting the trained network to FINN-ONNX + verification with the FINN compiler
    * Design space exploration and accelerator generation with the FINN compiler

* Conclusion

## Presenters

* Michaela Blott, Xilinx Research Labs
* Zaid Al-Ars, TU Delft
* Yaman Umuroglu, Xilinx Research Labs
