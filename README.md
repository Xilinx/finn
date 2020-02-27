## <img src=https://raw.githubusercontent.com/Xilinx/finn/master/docs/img/finn-logo.png width=128/> Fast, Scalable Quantized Neural Network Inference on FPGAs



<img align="left" src="https://raw.githubusercontent.com/Xilinx/finn/master/docs/img/finn-stack.png" alt="drawing" style="margin-right: 20px" width="250"/>

[![Gitter](https://badges.gitter.im/xilinx-finn/community.svg)](https://gitter.im/xilinx-finn/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![ReadTheDocs](https://readthedocs.org/projects/finn/badge/?version=latest&style=plastic)](http://finn.readthedocs.io/)

FINN is an experimental framework from Xilinx Research Labs to explore deep neural network
inference on FPGAs.
It specifically targets <a href="https://github.com/maltanar/qnn-inference-examples" target="_blank">quantized neural
networks</a>, with emphasis on
generating dataflow-style architectures customized for each network.
The resulting FPGA accelerators can yield very high classification rates, or conversely be run with a slow clock for very low power consumption.
The framework is fully open-source in order to give a higher degree of flexibility, and is intended to enable neural network research spanning several layers of the software/hardware abstraction stack.

For more general information about FINN, please visit the [project page](https://xilinx.github.io/finn/), check out the [publications](https://xilinx.github.io/finn/publications) or some of the [demos](https://xilinx.github.io/finn/demos).

## Getting Started

Please see the [Getting Started](https://finn.readthedocs.io/en/latest/getting_started.html) page for more information on requirements, installation, and how to run FINN in different modes. Due to the complex nature of the dependencies of the project, we only support Docker-based deployment at this time.

## What's New in FINN?

* **2020-02-27:** FINN v0.2b (beta) is released, which is a clean-slate reimplementation of the framework. Currently only fully-connected networks are supported for the end-to-end flow. Please see the release blog post for a summary of the key features.

## Documentation

You can view the documentation on [readthedocs](https://finn.readthedocs.io) or build them locally using `python setup.py doc` from inside the Docker container. Additionally, there is a series of [Jupyter notebook tutorials](https://github.com/Xilinx/finn/tree/master/notebooks), which we recommend running from inside Docker for a better experience.

## Community

We have a [gitter channel](https://gitter.im/xilinx-finn/community) where you can ask questions. You can use the GitHub issue tracker to report bugs, but please don't file issues to ask questions as this is better handled in the gitter channel. We also heartily welcome contributors to the project but do not yet have guidelines in place for this, so if you are interested just get in touch over gitter.

## Citation

The current implementation of the framework is based on the following publications. Please consider citing them if you find FINN useful.

    @article{blott2018finn,
      title={FINN-R: An end-to-end deep-learning framework for fast exploration of quantized neural networks},
      author={Blott, Michaela and Preu{\ss}er, Thomas B and Fraser, Nicholas J and Gambardella, Giulio and O’brien, Kenneth and Umuroglu, Yaman and Leeser, Miriam and Vissers, Kees},
      journal={ACM Transactions on Reconfigurable Technology and Systems (TRETS)},
      volume={11},
      number={3},
      pages={1--23},
      year={2018},
      publisher={ACM New York, NY, USA}
    }

    @inproceedings{finn,
    author = {Umuroglu, Yaman and Fraser, Nicholas J. and Gambardella, Giulio and Blott, Michaela and Leong, Philip and Jahre, Magnus and Vissers, Kees},
    title = {FINN: A Framework for Fast, Scalable Binarized Neural Network Inference},
    booktitle = {Proceedings of the 2017 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays},
    series = {FPGA '17},
    year = {2017},
    pages = {65--74},
    publisher = {ACM}
    }

## Old version

We previously released an early-stage prototype of a toolflow that took in Caffe-HWGQ binarized network descriptions and produced dataflow architectures. You can find it in the [v0.1](https://github.com/Xilinx/finn/tree/v0.1) branch in this repository.
Please be aware that this version is deprecated and unsupported, and the master branch does not share history with that branch so it should be treated as a separate repository for all purposes.
