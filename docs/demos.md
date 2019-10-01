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
