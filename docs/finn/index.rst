.. finn documentation master file, created by
   sphinx-quickstart on Mon Feb 24 14:55:45 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

****
FINN
****
.. note:: **This website is currently under construction.**

Welcome to the FINN Read the Docs website. This website is about the new, more modular version of FINN, which is currently under development on GitHub, and we welcome contributions from the community! Stay tuned for more updates.

What is FINN?
=============
'FINN' is colloquially used to refer to two separate but highly related things:

* The FINN project, which is an experimental framework from Xilinx Research Labs to explore deep neural network inference on FPGAs. It specifically targets quantized neural networks, with emphasis on generating dataflow-style architectures customized for each network. It includes tools for training quantized neural networks such as Brevitas, the FINN compiler, and the finn-hlslib Vivado HLS library of FPGA components for QNNs. An overview of the project can be taken from the following graphic and details can be seen on the `FINN project homepage <https://xilinx.github.io/finn/>`_.

.. image:: ../img/finn-stack.png
   :scale: 40%
   :align: center

* The FINN compiler, which this Read the Docs website corresponds to and is the centerpiece of the FINN project. Details can be looked up directly in the `FINN GitHub repository <https://github.com/Xilinx/finn>`_. To learn more about the FINN compiler, use this website and for a hands-on experience the repository contains some Jupyter notebooks which can be found under this `link <https://github.com/Xilinx/finn/tree/dev/notebooks>`_.

More FINN Resources
===================

* `List of publications <https://github.com/Xilinx/finn/blob/master/docs/publications.md>`_

* `Roadmap <https://github.com/Xilinx/finn/projects/1>`_

.. toctree::
   :maxdepth: 5
   :hidden:

   getting_started
   tutorials
   end_to_end_flow
   example_networks
   internals
   source_code/finn
   genindex
