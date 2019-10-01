# FINN
<img align="left" src="img/finn-stack.png" alt="drawing" style="margin-right: 20px" width="300"/>

FINN is an
experimental framework from Xilinx Research Labs to explore deep neural network
inference on FPGAs.
It specifically targets <a href="https://github.com/maltanar/qnn-inference-examples" target="_blank">quantized neural
networks</a>, with emphasis on
generating dataflow-style architectures customized for each network.
It is not
intended to be a generic DNN accelerator like xDNN, but rather a tool for
exploring the design space of DNN inference accelerators on FPGAs.
<br><br>
A new, more modular version of FINN is currently under development on GitHub, and we welcome contributions from the community!
<br>

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
