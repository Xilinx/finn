# FINN
<img src="img/finn-example.png" alt="drawing" width="400"/>

FINN is a machine learning framework by the Integrated Communications and AI Lab of AMD Research & Advanced Development.
It provides an end-to-end flow for the exploration and implementation of quantized neural network inference solutions on FPGAs.
FINN generates dataflow architectures as a physical representation of the implemented custom network in space.
It is not a generic DNN acceleration solution but relies on co-design and design space exploration for quantization and parallelization tuning so as to optimize a solutions with respect to resource and performance requirements.

<br><br>
The FINN compiler is under active development <a href="https://github.com/Xilinx/finn">on GitHub</a>, and we welcome contributions from the community!

<br>
## Features

* **Templated Vitis HLS library of streaming components:** FINN comes with an
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

The FINN team consists of members of AMD Research under Ralph Wittig (AMD Research & Advanced Development) and members of Custom & Strategic Engineering under Allen Chen, working very closely with the Pynq team.

<img src="img/finn-team.png" alt="The FINN Team (AMD Research and Advanced Development)" width="400"/>

From top left to bottom right: Yaman Umuroglu, Michaela Blott, Alessandro Pappalardo, Lucian Petrica, Nicholas Fraser,
Thomas Preusser, Jakoba Petri-Koenig, Ken O’Brien

<img src="img/finn-team1.png" alt="The FINN Team (Custom & Strategic Engineering)" width="400"/>

From top left to bottom right: Eamonn Dunbar, Kasper Feurer, Aziz Bahri, John Monks, Mirza Mrahorovic
