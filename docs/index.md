# FINN
<img align="left" src="img/finn-stack.PNG" alt="drawing" style="margin-right: 20px" width="300"/>

FINN is a machine learning framework by the Integrated Communications and AI Lab of AMD Research & Advanced Development.
It provides an end-to-end flow for the exploration and implementation of quantized neural network inference solutions on FPGAs.
FINN generates dataflow architectures as a physical representation of the implemented custom network in space.
It is not a generic DNN acceleration solution but relies on co-design and design space exploration for quantization and parallelization tuning so as to optimize a solutions with respect to resource and performance requirements.

<br><br>
The FINN compiler is under active development <a href="https://github.com/Xilinx/finn">on GitHub</a>, and we welcome contributions from the community!

## Quickstart

Depending on what you would like to do, we have different suggestions on where to get started:

* **I want to try out prebuilt QNN accelerators on my FPGA board.** Head over to [finn-examples](https://github.com/Xilinx/finn-examples)
to try out some FPGA accelerators built with the FINN compiler. We have more examples in the [BNN-PYNQ](https://github.com/Xilinx/BNN-PYNQ)
and the [LSTM-PYNQ](https://github.com/Xilinx/LSTM-PYNQ) repos, although these are not built with the FINN compiler.
* **I want to train new quantized networks for FINN.** Check out <a href="https://github.com/Xilinx/brevitas">Brevitas</a>,
our PyTorch library for quantization-aware training.
* **I want to understand the computations involved in quantized inference.** Check out these Jupyter notebooks on <a href="https://github.com/maltanar/qnn-inference-examples">QNN inference</a>. This repo contains simple Numpy/Python layer implementations and a few pretrained QNNs for instructive purposes.
* **I want to understand how it all fits together.** Check out our [publications](publications.md),
particularly the <a href="https://arxiv.org/abs/1612.07119" target="_blank">FINN paper at FPGA'17</a> and the <a href="https://arxiv.org/abs/1809.04570" target="_blank">FINN-R paper in ACM TRETS</a>.
