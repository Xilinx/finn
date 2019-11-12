.. Copyright (c) 2019, Xilinx, Inc.
.. All rights reserved.

.. Redistribution and use in source and binary forms, with or without
.. modification, are permitted provided that the following conditions are met:

.. 1.  Redistributions of source code must retain the above copyright notice,
..    this list of conditions and the following disclaimer.

.. 2.  Redistributions in binary form must reproduce the above copyright
..     notice, this list of conditions and the following disclaimer in the
..     documentation and/or other materials provided with the distribution.

.. 3.  Neither the name of the copyright holder nor the names of its
..     contributors may be used to endorse or promote products derived from
..     this software without specific prior written permission.

.. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
.. AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
.. THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
.. PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
.. CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
.. EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
.. PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
.. OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
.. WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
.. OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
.. ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

====================================
Introduction
====================================
FINN is an experimental framework from Xilinx Research Labs to explore deep neural network inference on FPGAs. It specifically targets quantized neural networks, with emphasis on generating dataflow-style architectures customized for each network. It is not intended to be a generic DNN accelerator like xDNN, but rather a tool for exploring the design space of DNN inference accelerators on FPGAs. 

====================================
FINN-HLS content
====================================

The FINN-HLS repository contains the C++ description of multiple layers for the implementation of quantized neural networks using dataflow architecture. 
The library serves as a hardware backend for the FINN project, and it integrates in Xilinx Vivado HLS tool.


.. toctree::
   :maxdepth: 2


   
  library/activations 
  library/weights 
  library/matrixvector 
  library/dma
  library/maxpool 
  library/fclayer 
  library/convlayer 
  library/swg
  library/streamtools 
  library/mac
  library/mmv

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
