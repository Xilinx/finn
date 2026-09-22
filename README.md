## <img src=https://raw.githubusercontent.com/Xilinx/finn/github-pages/docs/img/finn-logo.png width=128/> Fast, Scalable Quantized Neural Network Inference on FPGAs



<img align="left" src="https://raw.githubusercontent.com/Xilinx/finn/github-pages/docs/img/finn-stack.PNG" alt="drawing" style="margin-right: 20px" width="250"/>

[![GitHub Discussions](https://img.shields.io/badge/discussions-join-green)](https://github.com/Xilinx/finn/discussions)
[![ReadTheDocs](https://readthedocs.org/projects/finn/badge/?version=latest&style=plastic)](http://finn.readthedocs.io/)

FINN is an experimental framework from Integrated Communications and AI Lab of AMD Research & Advanced Development to explore deep neural network inference on FPGAs.
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

* Please find all news under [GitHub discussions Announcements](https://github.com/Xilinx/finn/discussions/categories/announcements).

## Documentation

You can view the documentation on [readthedocs](https://finn.readthedocs.io). Additionally, there is a series of [Jupyter notebook tutorials](https://github.com/Xilinx/finn/tree/main/notebooks), which we recommend running from inside Docker for a better experience.

## Community

We have [GitHub discussions](https://github.com/Xilinx/finn/discussions) where you can ask questions. You can use the GitHub issue tracker to report bugs, but please don't file issues to ask questions as this is better handled in GitHub discussions.

We also heartily welcome contributions to the project, please check out the [contribution guidelines](CONTRIBUTING.md) and the [list of open issues](https://github.com/Xilinx/finn/issues). Don't hesitate to get in touch over [GitHub discussions](https://github.com/Xilinx/finn/discussions) to discuss your ideas.

In the past, we also had a [Gitter channel](https://gitter.im/xilinx-finn/community). Please be aware that this is no longer maintained by us but can still be used to search for questions previous users had.


## Citation

If you use FINN in your research, please cite both the software and the foundational paper. See [CITATION.cff](CITATION.cff) for citation metadata that can be imported into reference managers.

GitHub also provides a "Cite this repository" option in the sidebar which uses this information.

## Old version

We previously released an early-stage prototype of a toolflow that took in Caffe-HWGQ binarized network descriptions and produced dataflow architectures. You can find it in the [v0.1](https://github.com/Xilinx/finn/tree/v0.1) branch in this repository.
Please be aware that this version is deprecated and unsupported, and the main branch does not share history with that branch so it should be treated as a separate repository for all purposes.
