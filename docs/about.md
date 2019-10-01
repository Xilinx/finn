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

## Who are we?

The FINN team is part of Xilinx's CTO group under Ivo Bolsens (CTO) and Kees Vissers (Fellow) and working very closely with the Pynq team and Kristof Denolf and Jack Lo for integration with video processing.

<img src="img/finn-team.jpg" alt="The FINN Team" width="400"/>

From left to right: Lucian Petrica, Giulio Gambardella,
Alessandro Pappalardo, Ken O’Brien, Michaela Blott, Nick Fraser, Yaman Umuroglu
