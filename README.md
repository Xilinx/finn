## <img src=https://raw.githubusercontent.com/Xilinx/finn/github-pages/docs/img/finn-logo.png width=128/> Fast, Scalable Quantized Neural Network Inference on FPGAs



<img align="left" src="https://raw.githubusercontent.com/Xilinx/finn/github-pages/docs/img/finn-stack.png" alt="drawing" style="margin-right: 20px" width="250"/>

[![Gitter](https://badges.gitter.im/xilinx-finn/community.svg)](https://gitter.im/xilinx-finn/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![ReadTheDocs](https://readthedocs.org/projects/finn/badge/?version=latest&style=plastic)](http://finn.readthedocs.io/)

FINN is an experimental framework from Xilinx Research Labs to explore deep neural network
inference on FPGAs.
It specifically targets <a href="https://github.com/maltanar/qnn-inference-examples" target="_blank">quantized neural
networks</a>, with emphasis on
generating dataflow-style architectures customized for each network.
The resulting FPGA accelerators are highly efficient and can yield high throughput and low latency.
The framework is fully open-source in order to give a higher degree of flexibility, and is intended to enable neural network research spanning several layers of the software/hardware abstraction stack.

We have a separate repository [finn-examples](https://github.com/Xilinx/finn-examples) that houses pre-built examples for several neural networks.
For more general information about FINN, please visit the [project page](https://xilinx.github.io/finn/) and check out the [publications](https://xilinx.github.io/finn/publications).

## Getting Started

Please see the [Getting Started](https://finn.readthedocs.io/en/latest/getting_started.html) page for more information on requirements, installation, and how to run FINN in different modes. Due to the complex nature of the dependencies of the project, **we only support Docker-based execution of the FINN compiler at this time**.

## What's New in FINN?

* **2021-11-05:** v0.7 is released, introducing QONNX support, three new example networks and many other improvements. Read more on the [v0.7 release blog post](https://xilinx.github.io/finn//2021/11/05/finn-v07-is-released.html).
* **2021-06-15:** v0.6 is released, with ResNet-50 on U250 and ZCU104 MobileNet-v1 in finn-examples showcasing new features plus a lot more. Read more on the [v0.6 release blog post](https://xilinx.github.io/finn//2021/06/15/finn-v06-is-released.html).
* **2020-12-17:** v0.5b (beta) is released, with a new [examples repo](https://github.com/Xilinx/finn-examples) including MobileNet-v1. Read more on the <a href="https://xilinx.github.io/finn/2020/12/17/finn-v05b-beta-is-released.html">release blog post</a>.

## Documentation

You can view the documentation on [readthedocs](https://finn.readthedocs.io) or build them locally using `python setup.py doc` from inside the Docker container. Additionally, there is a series of [Jupyter notebook tutorials](https://github.com/Xilinx/finn/tree/master/notebooks), which we recommend running from inside Docker for a better experience.

## Community

We have a [gitter channel](https://gitter.im/xilinx-finn/community) where you can ask questions. You can use the GitHub issue tracker to report bugs, but please don't file issues to ask questions as this is better handled in the gitter channel.

We also heartily welcome contributions to the project, please check out the [contribution guidelines](CONTRIBUTING.md) and the [list of open issues](https://github.com/Xilinx/finn/issues). Don't hesitate to get in touch over [Gitter](https://gitter.im/xilinx-finn/community) to discuss your ideas.

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
