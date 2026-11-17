This repo contains several Jupyter notebook examples on how to use FINN.
These are intended as tutorials

* `cybersecurity` is a three-part tutorial that shows how to train a simple
quantized MLP in Brevitas and deploy that with FINN using the FINN builder.
This is the recommended starting point for building your own accelerator.

* `bnn-pynq` explains how FINN works internally. One notebook goes through the
intermediate models of a builder run for a simple MLP and a convolutional net
and explains what each builder step does, the other one shows how to simulate
these models in Python, C++ and RTL to verify and debug them. These notebooks
are a reference for understanding and debugging the compiler, not a template:
to build your own network, start from a builder configuration as in the
`cybersecurity` notebooks and the advanced builder settings notebook.

In addition to these notebooks, you may want to check out the [finn-examples](https://github.com/Xilinx/finn-examples) repo, which contains
several prebuilt bitfiles (including a MobileNet-v1) as well as the Python scripts to rebuild them with
FINN.
