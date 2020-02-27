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
A new, more modular version of FINN is currently under development <a href="https://github.com/Xilinx/finn">on GitHub</a>, and we welcome contributions from the community!


## Quickstart

Depending on what you would like to do, we have different suggestions on where to get started:

* **I want to try out prebuilt QNN accelerators on real hardware.** Head over to <a href="https://github.com/Xilinx/BNN-PYNQ" target="_blank">BNN-PYNQ</a> repository to try out some image
classification accelerators, or to <a href="https://github.com/Xilinx/LSTM-PYNQ" target="_blank">LSTM-PYNQ</a>
to try optical character recognition with LSTMs.
* **I want to train new quantized networks for FINN.** Check out <a href="https://github.com/Xilinx/brevitas">Brevitas</a>,
our PyTorch library for training quantized networks. The Brevitas-to-FINN part of the flow is coming soon!
* **I want to understand the computations involved in quantized inference.** Check out these Jupyter notebooks on <a href="https://github.com/maltanar/qnn-inference-examples">QNN inference</a>. This repo contains simple Numpy/Python layer implementations and a few pretrained QNNs for instructive purposes.
* **I want to understand how it all fits together.** Check out our [publications](#publications),
particularly the <a href="https://arxiv.org/abs/1612.07119" target="_blank">FINN paper at FPGA'17</a> and the <a href="https://arxiv.org/abs/1809.04570" target="_blank">FINN-R paper in ACM TRETS</a>.
