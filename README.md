## <img src=https://raw.githubusercontent.com/Xilinx/finn/master/docs/img/finn-logo.png width=128/> Fast, Scalable Quantized Neural Network Inference on FPGAs

[![Gitter](https://badges.gitter.im/xilinx-finn/community.svg)](https://gitter.im/xilinx-finn/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![ReadTheDocs](https://readthedocs.org/projects/finn/badge/?version=latest&style=plastic)](http://finn.readthedocs.io/)

FINN is an experimental framework from Xilinx Research Labs to explore deep neural network inference on FPGAs. It specifically targets quantized neural networks, with emphasis on generating dataflow-style architectures customized for each network.
For more general information about FINN, please visit the [project page](https://xilinx.github.io/finn/).

## Getting Started

Please see the [Getting Started](https://finn.readthedocs.io/en/latest/getting_started.html) page for more information on installation, requirements and how to run FINN in different modes.

## Old version

We previously released an early-stage prototype of a toolflow that took in Caffe-HWGQ binarized network descriptions and produced dataflow architectures. You can find it in the [v0.1](https://github.com/Xilinx/finn/tree/v0.1) branch in this repository.
Please be aware that this version is deprecated and unsupported, and the master branch does not share history with that branch so it should be treated as a separate repository for all purposes.
