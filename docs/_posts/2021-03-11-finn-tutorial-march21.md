---
layout: post
title:  "FINN tutorial, March 2021"
author: "Yaman Umuroglu"
---

*This event has now concluded. You can find the materials at the bottom of this page.*

We're delighted to announce a three-hour FINN tutorial. Details are as follows:

* **Date:** 24 March 2021 (Wednesday)
* **Time:**
    * Central European Time (CET) 17:30 – 20:30 PM
* **Format:** Virtual event, Zoom call with hands-on Jupyter notebook lab
    * Zoom and Jupyter links will be sent to confirmed participants via e-mail
* **Registration (closes 21 March 2021 17:00 CET):** [Google form](https://forms.gle/JixoUdDryi7jJgrr8)

## Description

Mixing machine learning into high-throughput, low-latency edge applications needs co-designed solutions to meet the performance requirements. Quantized Neural Networks (QNNs) combined with custom FPGA dataflow implementations offer a good balance of performance and flexibility, but building such implementations by hand is difficult and time-consuming.

In this tutorial, we will introduce FINN, an open-source experimental framework by Xilinx Research Labs to help the broader community explore QNN inference on FPGAs. Providing a full-stack solution from quantization-aware training to bitfile, FINN generates high-performance dataflow-style FPGA architectures customized for each network. Participants will be introduced to efficient inference with QNNs and streaming dataflow architectures, the components of the project’s open-source ecosystem, and gain hands-on experience training a quantized neural network with Brevitas and deploying it with FINN.

## Practical Information

Some prior knowledge of FPGAs, Vivado HLS, PyTorch and neural network training is recommended, but not required.

This will be a virtual event, with a Zoom video call and a hands-on Jupyter notebook lab.
Registered participants will get access to a FINN setup running in the cloud.
There are no special technical requirements besides a browser and Zoom client.

Connect with us and the FINN community on the [FINN Gitter channel](https://gitter.im/xilinx-finn/community).

## Agenda

* Part I: Introduction
    * An introduction to FINN, QNNs and streaming dataflow architectures
    * Tour of FINN GitHub repositories
    * The FINN open-source community
    * Q & A

* Part II: Hands-on lab: Training and deploying an MLP for network intrusion detection
    * Training a quantized MLP on the UNSW-NB15 dataset with Brevitas
    * Importing the MLP into FINN and verification
    * Design space exploration and accelerator generation with the FINN compiler

* Demo + Conclusion

## Organization

* Yaman Umuroglu, Michaela Blott, Jon Ander Lezeta and Felix Paul Jentzsch, Xilinx Research Labs
* Zaid Al-Ars and Jakoba Petri-Koenig, TU Delft
* Holger Froening and Hendrik Borras, Heidelberg University

## Materials

* Part I: Introduction
    * [Slides](https://github.com/Xilinx/finn/blob/github-pages/docs/finn-march21-tutorial.pdf)

* Part II: Hands-on lab: Training and deploying an MLP for network intrusion detection
   * [Slides](http://bit.ly/finn-tutorial-march21-hands-on-slides)
   * [Jupyter notebooks](http://bit.ly/finn-tutorial-march21-notebooks)
