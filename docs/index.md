# Machine Learning on Xilinx FPGAs with FINN

## What is FINN?

<img align="left" src="img/finn-example.png" alt="drawing" width="300"/>

FINN is an
experimental framework from Xilinx Research Labs to explore deep neural network
inference on FPGAs. 
It specifically targets <a href="https://github.com/maltanar/qnn-inference-examples" target="_blank">quantized neural
networks</a>, with emphasis on
generating dataflow-style architectures customized for each network.
It is not
intended to be a generic DNN accelerator like xDNN, but rather a tool for
exploring the design space of DNN inference accelerators on FPGAs.
<br>
## Features

* **Templated Vivado HLS library of streaming components:** FINN comes with an
HLS hardware library that implements convolutional, fully-connected, pooling and
LSTM layer types as streaming components. The library uses C++ templates to
support a wide range of precisions.
* **Ultra low-latency and high performance
with dataflow:** By composing streaming components for each layer, FINN can
generate accelerators that can classify images at sub-microsecond latency.
* **Many end-to-end example designs:** We provide examples that start from training a
quantized neural network, all the way down to an accelerated design running on
hardware. The examples span a range of datasets and network topologies.
* **Toolflow for rapid design generation:** The FINN toolflow supports allocating
separate compute resources per layer, either automatically or manually, and
generating the full design for synthesis. This enables rapid exploration of the
design space.

## Quickstart

Depending on what you would like to do, we have
different suggestions on where to get started:

* **I want to try out premade accelerators on real hardware.** Head over to <a href="https://github.com/Xilinx/BNN-PYNQ" target="_blank">BNN-PYNQ</a> repository to try out some image
classification accelerators, or to <a href="https://github.com/Xilinx/LSTM-PYNQ" target="_blank">LSTM-PYNQ</a>
to try optical character recognition with LSTMs.
* **I want to try the full design flow.** The <a href="https://github.com/Xilinx/FINN" target="_blank">FINN</a> repository
contains the Python toolflow that goes from a trained, quantized Caffe network
to an accelerator running on real hardware.
* **I want to train new quantized networks for FINN.** Have a look <a href="https://github.com/Xilinx/BNN-PYNQ/tree/master/bnn/src/training" target="_blank">here</a>, at 
[this presentation](https://drive.google.com/open?id=17oorGvtUbdFd-o1OzSuxGCSrWsvm_S2ftC1UC2FLtuE) 
for an example with Fashion-MNIST, or <a href="https://github.com/Xilinx/pytorch-ocr" target="_blank">here</a> for quantized 
LSTMs with PyTorch.
* **I want to understand how it all fits together.** Check out our [publications](#publications), 
particularly the <a href="https://arxiv.org/abs/1612.07119" target="_blank">FINN paper at FPGA'17</a> and the <a href="https://arxiv.org/abs/1809.04570" target="_blank">FINN-R paper in ACM TRETS</a>.

## Neural Network Demos

Multiple Jupyter notebooks examples are provided, with different datasets and two architectures:

* **Feed-forward Dataflow**: all layers of the network are
implemented in the hardware, the output of one layer is the input of the
following one that starts processing as soon as data is available. The network
parameters for all layers are cached in the on-chip memory. For each network
topology, a customized hardware implementation is generated that provides low
latency and high throughput.

* **Dataflow with loopback**: a fixed hardware
architecture is implemented, being able to compute multiple layers in a single
call. The complete network is executed in multiple calls, which are scheduled on
the same hardware architecture. Changing the network topology implies changing
the runtime scheduling, but not the hardware architecture. This provides a
flexible implementation but features slightly higher latency.

Our design
examples are mostly for the <a href="http://www.pynq.io/" target="_blank">PYNQ</a> Z1 and Z2 boards, and a
few for the Ultra96. Future support for AWS F1 and other Xilinx platforms is
also planned.

### Demos with Dataflow Architecture 

| Thumbnail | Dataset | Neural Network | Task | Link |
|-----------|---------|-------------|------|--------|
|<img src="img/cifar-10.png" alt="drawing" width="200"/>|<a href="https://www.cs.toronto.edu/~kriz/cifar.html" target="_blank">CIFAR-10</a>|6 convolutional, 3 max pool and 3 fully connected layers|Image classification (animals and vehicles)|<a href="https://github.com/Xilinx/BNN-PYNQ/blob/master/notebooks/CNV-BNN_Cifar10.ipynb" target="_blank">Cifar10</a>|
|<img src="img/svhn.png" alt="drawing" width="200"/>|<a href="http://ufldl.stanford.edu/housenumbers/" target="_blank">Street View House Numbers</a>|6 convolutional, 3 max pool and 3 fully connected layers|Image classification (house numbers)|<a href="https://github.com/Xilinx/BNN-PYNQ/blob/master/notebooks/CNV-BNN_SVHN.ipynb" target="_blank">SVHN</a>|
|<img src="img/gtsrb.png" alt="drawing" width="200"/>|<a href="http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset" target="_blank">German Road Signs</a>|6 convolutional, 3 max pool and 3 fully connected layers|Image classification (road signs)|<a href="https://github.com/Xilinx/BNN-PYNQ/blob/master/notebooks/CNV-BNN_Road-Signs.ipynb" target="_blank">GTRSB</a>|
|<img src="img/mnist.jpg" alt="drawing" width="200"/>|<a href="http://yann.lecun.com/exdb/mnist/" target="_blank">MNIST</a>|3 fully connected layers|Image classification (handwritten digits)|<a href="https://github.com/Xilinx/BNN-PYNQ/blob/master/notebooks/LFC-BNN_MNIST_Webcam.ipynb" target="_blank">MNIST</a>|
|<img src="img/fraktur.png" alt="drawing" width="200"/>|Fraktur|Bi-LSTM|Optical Character Recognition|<a href="https://github.com/Xilinx/LSTM-PYNQ/blob/master/notebooks/Fraktur_OCR.ipynb" target="_blank">Fraktur</a>|

### Demos with Loopback Architecture

* <a href="https://github.com/Xilinx/QNN-MO-PYNQ/blob/master/notebooks/dorefanet-classification.ipynb" target="_blank">ImageNet Classification</a>: shows an example
on how to classify a non-labelled image (e.g., downloaded from the web, your
phone etc) in one of the 1000 classes available on the <a href="http://image-
net.org/challenges/LSVRC/2014/browse-synsets" target="_blank"> ImageNet </a>
dataset.  

* <a href="https://github.com/Xilinx/QNN-MO-PYNQ/blob/master/notebooks/dorefanet-imagenet-samples.ipynb" target="_blank">ImageNet - Dataset validation</a>: shows an example classifying labelled image (i.e.,  extracted
from the dataset) in one of the 1000 classes available on the <a href="http
://image-net.org/challenges/LSVRC/2014/browse-synsets" target="_blank"> ImageNet
</a> dataset.  

* <a href="https://github.com/Xilinx/QNN-MO-PYNQ/blob/master/notebooks/dorefanet-imagenet-loop.ipynb" target="_blank">ImageNet - Dataset validation in a loop</a>: shows an example classifying labelled image
(i.e.,  extracted from the dataset) in one of the 1000 classes available on the
<a href="http://image-net.org/challenges/LSVRC/2014/browse-synsets"
target="_blank"> ImageNet </a> dataset in a loop.

* <a href="https://github.com/Xilinx/QNN-MO-PYNQ/blob/master/notebooks/tiny-yolo-image.ipynb" target="_blank">Object Detection - from image</a>: shows object detection in a image
(e.g., downloaded from the web, your phone etc), being able to identify objects
in a scene and drawing bounding boxes around them. The objects can be one of the
20 available in the  <a href="http://host.robots.ox.ac.uk/pascal/VOC/"
target="_blank"> PASCAL VOC </a> dataset

## Other Repositories

* The <a href="https://github.com/Xilinx/FINN" target="_blank">FINN toolflow repository</a> contains a end-to-end Python
"compiler" flow to import a trained <a href="https://github.com/zhaoweicai/hwgq/" target="_blank">quantized Caffe</a> network, perform simplifications and
resource allocation per layer, emit and synthesize the resulting HLS design.

* The <a href="https://github.com/Xilinx/pytorch-quantization" target="_blank">pytorch-quantization repository</a> provides primitives for **LSTM** quantization at training time
using **PyTorch**.

* The <a href="https://github.com/Xilinx/pytorch-ocr" target="_blank">pytorch-ocr repository</a> provides tools for training and exporting quantized LSTMs for
**OCR** in PyTorch, targeting **LSTM-PYNQ**.

## Publications

* FPL'18: <a href="https://arxiv.org/pdf/1807.04093.pdf" target="_blank">FINN-L:Library Extensions and Design Trade-off Analysis for Variable Precision LSTM Networks on FPGAs</a>
* FPL'18: <a href="https://arxiv.org/pdf/1806.08862.pdf" target="_blank">BISMO: A Scalable Bit-Serial Matrix Multiplication Overlay for Reconfigurable Computing</a>
* FPL'18: <a href="http://kalman.mee.tcd.ie/fpl2018/content/pdfs/FPL2018-43iDzVTplcpussvbfIaaHz/XZmyRhWvHACdwHRVTCTVB/6jfImwD836ibhOELmms0Ut.pdf" target="_blank">Customizing Low-Precision Deep Neural Networks For FPGAs</a>
* ACM TRETS, Special Issue on Deep Learning: <a href="https://arxiv.org/abs/1809.04570" target="_blank">FINN-R: An End-to-End Deep-Learning Framework for Fast Exploration of Quantized Neural Networks</a>
* ARC'18: <a href="https://arxiv.org/pdf/1807.10577.pdf" target="_blank">Accuracy to Throughput Trade-Offs for Reduced Precision Neural Networks on Reconfigurable Logic</a>
* CVPR’18: <a href="https://arxiv.org/abs/1807.00301" target="_blank">SYQ: Learning Symmetric Quantization For Efﬁcient Deep Neural Networks</a>
* DATE'18: <a href="https://ieeexplore.ieee.org/abstract/document/8342121/" target="_blank">Inference of quantized neural networks on heterogeneous all-programmable devices</a>
* ICONIP’17: <a href="https://arxiv.org/abs/1709.06262" target="_blank">Compressing Low Precision Deep Neural Networks Using Sparsity-Induced Regularization in Ternary Networks</a>
* ICCD'17: <a href="https://ieeexplore.ieee.org/abstract/document/8119246/" target="_blank">Scaling Neural Network Performance through Customized Hardware Architectures on Reconfigurable Logic</a>
* PARMA-DITAM'17: <a href="https://arxiv.org/abs/1701.03400" target="_blank">Scaling Binarized Neural Networks on Reconfigurable Logic</a>
* FPGA'17: <a href="https://arxiv.org/abs/1612.07119" target="_blank">FINN: A Framework for Fast, Scalable Binarized Neural Network Inference</a>
* H2RC'16: <a href="https://h2rc.cse.sc.edu/2016/papers/paper_25.pdf" target="_blank">A C++ Library for Rapid Exploration of Binary Neural Networks on Reconfigurable Logic</a>

## External Publications and Projects Based on FINN

If you are using FINN in your
work and would like to be listed here, please contact us!

* <a href="https://coefs.uncc.edu/htabkhiv/teaching/hardware-software-co-design-real-time-ai/" target="_blank">Hardware-Software Co-Design Real-time AI (UNC Charlotte)</a>
* <a href="https://ieeexplore.ieee.org/abstract/document/8442108" target="_blank">BinaryEye: A 20 kfps Streaming Camera System on FPGA with Real-Time On-Device Image Recognition Using Binary Neural Networks</a>
* <a href="https://qiita.com/ykshr/items/08147098516a45203761" target="_blank">Cucumber sorting with FINN (in Japanese)</a>
* <a href="https://github.com/mohaghasemzadeh/ReBNet" target="_blank">ReBNet: Residual Binarized Neural Network, FCCM'18 best paper</a>

## Events, Tutorials and Keynotes

* Future of AI'2019 keynote:  <a href="https://github.com/Xilinx/FINN/blob/master/docs/FutureofAI2019_Blott.pdf" target="_blank">Future of AI: Unconventional Compute Architectures</a>
* BigData Belfast'2018 talk: <a href="https://github.com/Xilinx/FINN/blob/master/docs/BigDataBelfast2018.pdf" target="_blank">Unconventional Compute Architectures for Enabling the Roll-Out of Deep Learning</a>
* CLUSTER'2018 keynote: <a href="https://github.com/Xilinx/FINN/blob/master/docs/IEEECluster2018.pdf" target="_blank">Unconventional Compute Architectures with Reconfigurable Devices in the Cloud</a>
* RCML'2018 invited talk: <a href="https://github.com/Xilinx/FINN/blob/master/docs/ARC2018.pdf" target="_blank">The Emerging Computational Landscape of Neural Networks</a>
* HotChips'2018 ML tutorial: <a href="https://github.com/Xilinx/FINN/blob/master/docs/Hotchips2018_Tutorial.pdf" target="_blank">Overview of Deep Learning and Computer Architectures for Accelerating DNNs</a>
  + <a href="https://youtu.be/ydsZ7A0FF0I" target="_blank">Video</a>
* ASAP'2018 keynote: <a href="https://github.com/Xilinx/FINN/blob/master/docs/ASAP2018.pdf" target="_blank">Design Trade-offs for Machine Learning Solutions on Reconfigurable Devices</a>
* ARC'2018 keynote: <a href="https://github.com/Xilinx/FINN/blob/master/docs/ARC2018.pdf" target="_blank">Scalable Machine Learning with Reconfigurable Devices</a>
* FPGA'2018 tutorial: <a href="https://github.com/Xilinx/FINN/blob/master/docs/FPGA2018_tutorial.pdf" target="_blank">Training Quantized Neural Networks</a>
* MPSoC 2017 talk: <a href="https://github.com/Xilinx/FINN/blob/master/docs/MPSOC2018.pdf" target="_blank">A Framework for Reduced Precision Neural Networks on FPGAs</a>
* TCD 2017 guest lecture on ML: <a href="https://www.youtube.com/watch?v=pIVh-4tqjPc" target="_blank">Machine Learning for Embedded Systems (Video)</a>
* QPYNQ'2017 tutorial: <a href="https://www.ntnu.edu/ie/eecs/qpynq" target="_blank">Quantized Neural Networks with Xilinx PYNQ</a>

## People

### The FINN Team

We are part of Xilinx's CTO group under Ivo Bolsens (CTO) and Kees Vissers (Fellow) and working very closely with the Pynq team and Kristof Denolf and Jack Lo for integration with video processing.

<img src="img/finn-team.jpg" alt="The FINN Team" width="400"/>

From left to right: Lucian Petrica, Giulio Gambardella,
Alessandro Pappalardo, Ken O’Brien, Michaela Blott, Nick Fraser, Yaman Umuroglu


### External Collaborators
* NTNU, Norway: Magnus Jahre, Magnus Sjalander
* University of Sydney, Australia: Julian Faraone, Philip Leong
* ETH Zurich, Switzerland: Kaan Kara, Ce Zhang, Lois Orosa, Onur Mutlu
* University of Kaiserslautern, Germany: Vladimir Rybalkin, Mohsin Ghaffar, Nobert Wehn
* Imperial College, UK: Alex (Jiang) Su and Peter Cheung
* Northeastern University, USA: Miriam Leeser
* Trinity College Dublin, Ireland: Linda Doyle
* Missing Link Electronics, Germany
