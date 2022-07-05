.. _internals:

*********
Internals
*********

Intermediate Representation: QONNX and FINN-ONNX
================================================

FINN uses `ONNX <https://github.com/onnx/onnx>`_ as an intermediate representation (IR) for neural networks. As such, almost every component inside FINN uses ONNX and its `Python API <https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md>`_, so you may want to familiarize yourself with how ONNX represents DNNs. Specifically, the `ONNX protobuf description <https://github.com/onnx/onnx/blob/master/onnx/onnx.proto>`_ (or its `human-readable documentation <https://github.com/onnx/onnx/blob/master/docs/IR.md>`_ and the `operator schemas <https://github.com/onnx/onnx/blob/master/docs/Operators.md>`_ are useful as reference documents. We also provide a Jupyter notebook that can help to get familiar with ONNX by showing how to work with a simple ONNX model in FINN, see chapter :ref:`tutorials` for details.

.. note:: FINN supports two specialized variants of ONNX called QONNX and FINN-ONNX, and not all ONNX graphs are supported by FINN (and vice versa).

Custom Quantization Annotations
===============================

ONNX does not support datatypes smaller than 8-bit integers, whereas in FINN we are interested in smaller integers down to ternary and bipolar. To make this work, FINN uses the quantization_annotation field in ONNX to annotate tensors with their FINN DataType (:py:mod:`qonnx.core.datatype.DataType`) information. However, all tensors are expected to use single-precision floating point (float32) storage in FINN. This means we store even a 1-bit value as floating point for the purposes of representation. The FINN compiler flow is responsible for eventually producing a packed representation for the target hardware, where the 1-bit is actually stored as 1-bit.

Note that FINN uses floating point tensors as a carrier data type to represent integers. Floating point arithmetic can introduce rounding errors, e.g. (int_num * float_scale) / float_scale is not always equal to int_num.
When using the custom ONNX execution flow, FINN will attempt to sanitize any rounding errors for integer tensors. See (:py:mod:`qonnx.util.basic.sanitize_quant_values`) for more information.
This behavior can be disabled (not recommended!) by setting the environment variable SANITIZE_QUANT_TENSORS=0.

Custom Operations/Nodes
=======================

FINN uses many custom operations (op_type in ONNX NodeProto) that are not defined in the ONNX operator schema. These custom nodes are marked with domain="finn.*" in the protobuf to identify them as such. These nodes can represent specific operations that we need for low-bit networks, or operations that are specific to a particular hardware backend. To get more familiar with custom operations and how they are created, please take a look in the Jupyter notebook about CustomOps (see chapter :ref:`tutorials` for details) or directly in the module :py:mod:`finn.custom_op`.

.. note:: See the description of `this PR <https://github.com/Xilinx/finn-base/pull/6>`_ for more on how the operator wrapper library is organized.

Custom ONNX Execution Flow
==========================

To verify correct operation of FINN-ONNX graphs, FINN provides its own ONNX execution flow (:py:mod:`finn.core.onnx_exec`). This flow supports the standard set of ONNX operations as well as the custom FINN operations.

.. warning:: This execution flow is only meant for checking the correctness of models after applying transformations, and not for high performance inference.

.. _modelwrapper:

ModelWrapper
============

FINN provides a ModelWrapper class (:py:mod:`qonnx.core.modelwrapper.ModelWrapper`) as a thin wrapper around ONNX to make it easier to analyze and manipulate ONNX graphs. This wrapper provides many helper functions, while still giving full access to the ONNX protobuf representation.

Some of the helper functions are described in more detail below.

Create a ModelWrapper instance
------------------------------
The ModelWrapper instance can be created using a model in .onnx format or by directly passing a ModelProto instance to the wrapper. The code block below gives an example of how to use the wrapper on a model in .onnx format.
::

  from qonnx.core.modelwrapper import ModelWrapper
  model = ModelWrapper("model.onnx")

Access the ONNX GraphProto through ModelWrapper
-----------------------------------------------
The ONNX ModelProto can be accessed with following command:
::

  modelproto = model.model

The graph can be accessed using:
::

  graphproto = model.graph

The node list is accessed by:
::

  nodes = model.graph.node

The individual nodes can be selected via their indices.
::

  # first node
  nodes[0]

The number of all nodes can be determined with the len() function in Python.
::

  # number of nodes in the graph
  len(nodes)

Helper functions for tensors
----------------------------

A list of all tensors (names) can easily be accessed using:
::

  tensor_list = model.get_all_tensor_names()

If we take a single tensor from that list (by index), we can determine their producer or consumer node by using one of the following functions. Note that it may be that a tensor does not have a producer or consumer node, for example if the tensor represents a constant that is already set. In that case `None` will be returned.
::

  # find producer of third tensor in model tensor list
  model.find_producer(tensor_list[2])

  # find consumer of third tensor in model tensor list
  model.find_consumer(tensor_list[2])

Every tensor has a specific shape, to get or to set this shape these functions can be used:
::

  # get tensor shape of third tensor in model tensor list
  model.get_tensor_shape(tensor_list[2])

  # set tensor shape of third tensor in model tensor list
  tensor_shape = [1, 1, 28, 28]
  model.set_tensor_shape(tensor_list[2], tensor_shape)

Optionally, the dtype (container datatype) of the tensor can also be specified as third argument in the set function. By default it is set to TensorProto.FLOAT.

As mentioned above there are FINN DataTypes additional to the container datatype, these can be accessed and set for a tensor with the following functions:
::

  # get tensor dataype of third tensor in model tensor list
  model.get_tensor_datatype(tensor_list[2])

  # set tensor datatype of third tensor in model tensor list
  from qonnx.core.datatype import DataType

  finn_dtype = DataType.BIPOLAR
  model.set_tensor_datatype(tensor_list[2], finn_dtype)

ModelWrapper contains two helper functions for tensor initializers, one to determine the current initializer and one to set the initializer of a tensor. If there is no initializer, None is returned.
::

  # get tensor initializer of third tensor in model tensor list
  model.get_initializer(tensor_list[2])

ModelWrapper contains more useful functions, if you are interested please have a look at the ModelWrapper module (:py:mod:`qonnx.core.modelwrapper.ModelWrapper`) directly.


.. _analysis_pass:

Analysis Pass
=============

An analysis pass traverses the graph structure and produces information about certain properties. It gets the model in the ModelWrapper as input and returns a dictionary of the properties the analysis extracts. If you are interested in how to write an analysis pass for FINN, please take a look at the Jupyter notebook about how to write an analysis pass, see chapter :ref:`tutorials` for details. For more information about existing analysis passes in FINN, see module :py:mod:`finn.analysis`.

.. _transformation_pass:

Transformation Pass
===================

A transformation passes changes (transforms) the given model, it gets the model in the ModelWrapper as input and returns the changed model (ModelWrapper) to the FINN flow. Additional the flag *model_was_changed* which indicates if a transformation has to be performed more than once, is returned. If you are interested in how to write a transformation pass for FINN, please take a look at the Jupyter notebook about how to write a transformation pass, see chapter :ref:`tutorials` for details. For more information about existing transformation passes in FINN, see module :py:mod:`finn.transformation`.

.. _mem_mode:

MatrixVectorActivation *mem_mode*
===========================

FINN supports two types of the so-called *mem_mode* attrıbute for the node MatrixVectorActivation. This mode controls how the weight values are accessed during the execution. That means the mode setting has direct influence on the resulting circuit. Currently two settings for the *mem_mode* are supported in FINN:

* "const"

* "decoupled"

The following picture shows the idea behind the two modes.

.. image:: img/mem_mode.png
   :scale: 55%
   :align: center

Const mode
----------
In *const* mode the weights are "baked in" into the Matrix-Vector-Activate-Unit (MVAU), which means they are part of the HLS code. During the IP block generation the weight values are integrated as *params.h* file in the HLS code and synthesized together with it. For the *const* mode IP block generation the `Matrix_Vector_Activate_Batch function <https://github.com/Xilinx/finn-hlslib/blob/19fa1197c09bca24a0f77a7fa04b8d7cb5cc1c1d/mvau.hpp#L93>`_ from the finn-hls library is used, which implements a standard MVAU. The resulting IP block has an input and an output stream, as shown in the above picture on the left. FIFOs in the form of verilog components are connected to these.

Advantages:

* smaller resource footprint

* easier to debug layer in cppsim since no additional components

* well-tested and mature components

Disadvantages:

* can lead to very long HLS synthesis times for certain weight array shapes

* less control over the weight memory FPGA primitives, Vivado HLS doesn't always make the best resource allocation decisions

Decoupled mode
--------------
In *decoupled* mode a different variant of the MVAU with three ports is used. Besides the input and output streams, which are fed into the circuit via Verilog FIFOs, there is another input, which is used to stream the weights. For this the `streaming MVAU <https://github.com/Xilinx/finn-hlslib/blob/07a8353f6cdfd8bcdd81e309a5581044c2a93d3b/mvau.hpp#L213>`_ from the finn-hls library is used. To make the streaming possible a Verilog weight streamer component accesses the weight memory and sends the values via another FIFO to the MVAU. This component can be found in the `finn-rtllib <https://github.com/Xilinx/finn/tree/dev/finn-rtllib>`_ under the name *memstream.v*. For the IP block generation this component, the IP block resulting from the synthesis of the HLS code of the streaming MVAU and a FIFO for the weight stream are combined in a verilog wrapper. The weight values are saved in .dat files and stored in the weight memory from which the weight streamer reads. The resulting verilog component, which is named after the name of the node and has the suffix "_memstream.v", exposes only two ports to the outside, the data input and output. It therefore behaves externally in the same way as the MVAU in *const* mode.

Advantages:

* better control over the used memory primivites used (see the ram_style attribute in MatrixVectorActivation)

* potentially faster HLS synthesis time since weight array shape is no longer part of HLS synthesis

* (future work) will enable placing memory and compute into different clock domains, combining different layers into same weight memory for higher packing efficiency, sourcing the weight stream from other sources such as DRAM

Disadvantages:

* somewhat less well-tested compared to the const mode

* higher resource footprint due to additional weight streamer and weight FIFO


How to set *mem_mode*
---------------------
When the nodes in the network are converted to HLS layers, the *mem_mode* can be passed. More detailed information about the transformations that prepare the network and the transformation that performs the conversion to HLS layers can be found in chapter :ref:`nw_prep`. The *mem_mode* is passed as argument. Note that if no argument is passed, the default is *const*.
