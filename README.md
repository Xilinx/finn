# FINN

**Warning: this early preview version of FINN is deprecated and unsupported as there are many bugs lurking about. A new clean-slate version is under development here on GitHub, see the dev branch.**

FINN is an end to end framework for generating high performance FPGA hardware implementations of neural networks. 

## Installation
FINN can be used within a Docker container (recommended flow) or locally.
### Docker Install flow

```git clone <repo>```
```cd FINN```
```docker build . --tag=finn```
```docker run -it finn```

### Local Install flow
#### Prerequisites

- Install HWGQ Caffe in the same directory as FINN: https://github.com/zhaoweicai/hwgq
- Install git lfs: https://git-lfs.github.com/
- Install numpy and google protobuf packages
    - https://www.scipy.org/install.html
    - https://github.com/google/protobuf
- Install pandas and lmdb packages
    - `sudo pip install pandas lmdb`

```git clone <repo>```

## Quick Start

```cd FINN```
```source env.sh```
```git lfs pull```

### Estimate performance of LFC MLP network

```python FINN/bin/finn --device=pynqz1 --prototxt=FINN/inputs/lfc-w1a1.prototxt --mode=estimate```

### Estimate performance and synthesize CNV network

```python FINN/bin/finn --device=pynqz1 --prototxt=FINN/inputs/cnv-w1a1.prototxt --caffemodel=FINN/inputs/cnv-w1a1.caffemodel --mode=synth```

## User guide

FINN can be called in two modes, estimate and synth. "Estimate" mode reads in the prototxt of the network, constructs a hardware model of the network and scales up the hardware to optimally utilize the chosen device. This process produces a report of estimated performance and hardware utilisation. "Synth" mode performs the same actions as estimate, but additionally calls Xilinx Vivado tools to synthesize the design for the target device.  


## Reference Designs

Two reference designs are included. Both have binarized weights and activations.

### MLP
A multilayer perceptron, trained for MNIST digit recognition. 

- prototxt = FINN/inputs/lfc-w1a1.prototxt
- caffemodel = FINN/inputs/lfc-w1a1.caffemodel

### CNV
A convolutional neural network, trained for CIFAR10 image classification.

- prototxt = FINN/inputs/cnv-w1a1.prototxt
- caffemodel = FINN/inputs/cnv-w1a1.caffemodel

## Citation
If you find FINN useful, please cite the [FINN paper](https://arxiv.org/abs/1612.07119).

    @inproceedings{finn,
    author = {Umuroglu, Yaman and Fraser, Nicholas J. and Gambardella, Giulio and Blott, Michaela and Leong, Philip and Jahre, Magnus and Vissers, Kees},
    title = {FINN: A Framework for Fast, Scalable Binarized Neural Network Inference},
    booktitle = {Proceedings of the 2017 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays},
    series = {FPGA '17},
    year = {2017},
    pages = {65--74},
    publisher = {ACM}
    }

