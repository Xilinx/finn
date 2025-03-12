<img src=https://cs.uni-paderborn.de/fileadmin-eim/informatik/fg/ce/MiscImages/finn-plus_logo.png width=196/>

# Dataflow Compiler for Fast, Scalable Quantized Neural Network Inference on FPGAs

**FINN+** is a fork of **FINN**, an experimental framework from Integrated Communications and AI Lab of AMD Research & Advanced Development to explore deep neural network inference on FPGAs.
It specifically targets quantized neural networks, with emphasis on generating dataflow-style architectures customized for each network.
The resulting FPGA accelerators are highly efficient and can yield high throughput and low latency.
The framework is fully open-source in order to give a higher degree of flexibility, and is intended to enable neural network research spanning several layers of the software/hardware abstraction stack.

**For the time being, we refer to the original project [webpage](https://xilinx.github.io/finn/) for further information and documentation.**

## FINN+ Extensions
**FINN+** aims to incorporate all development from the upstream repository (dev branch) while extending **FINN** in all directions, including the following list of features that are either in progress or already completed:
- Transformer/Attention support
- Improved streamlining
- Improved automatic folding and FIFO-sizing
- Empirical quality-of-result (QoR) estimation
- Back-end extensions
    - Instrumentation for accurate performance profiling in simulation and on hardware
    - Improved Alveo build flow
    - Multi-FPGA support
    - Optimized C++ driver
- Quality-of-live improvements
    - Better logging and error handling
    - Type hinting/checking
    - Alternative YAML-based build configuration
    - Containerless setup

Please refer to our [**Feature Tracker**](https://github.com/orgs/eki-project/projects/1) for the current status of individual features.
While some items are already on-track to be merged into the upstream repository, we try to merge them into the **FINN+** dev branch as early as possible to increase development pace and drive our research forward.

## About Us
This repository is maintained by researchers from the [Computer Engineering Group](https://en.cs.uni-paderborn.de/ceg) (CEG) and [Paderborn Center for Parallel Computing](https://pc2.uni-paderborn.de/) (PC²) at Paderborn University, Germany as part of the [eki research project](https://www.eki-project.tech/).

<p align="left">
<a href="https://en.cs.uni-paderborn.de/ceg"><img align="top" src="https://cs.uni-paderborn.de/fileadmin-eim/informatik/fg/ce/MiscImages/UPB_Logo_ENG_coloured_RGB.jpg" alt="logo" style="margin-right: 20px" width="250"/></a>
<a href="https://pc2.uni-paderborn.de/"><img align="top" src="https://cs.uni-paderborn.de/fileadmin-eim/informatik/fg/ce/MiscImages/PC2_logo.png" alt="logo" style="margin-right: 20px" width="250"/></a>
</p>

<p align="left">
<a href="https://www.eki-project.tech/"><img align="top" src="https://cs.uni-paderborn.de/fileadmin-eim/informatik/fg/ce/MiscImages/eki-RGB-EN-s.png" alt="logo" style="margin-right: 20px" width="250"/></a>
<a href="https://www.bmuv.de/"><img align="top" src="https://cs.uni-paderborn.de/fileadmin-eim/informatik/fg/ce/MiscImages/BMUV_Fz_2021_Office_Farbe_en.png" alt="logo" style="margin-right: 20px" width="250"/></a>
</p>
