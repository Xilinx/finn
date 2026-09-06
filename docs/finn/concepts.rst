.. _concepts:

********
Concepts
********

This page introduces the core concepts and abstractions used throughout FINN.

Intermediate Representation: QONNX and FINN-ONNX
=================================================

FINN uses `ONNX <https://github.com/onnx/onnx>`_ as an intermediate representation (IR) for neural networks. Almost every component inside FINN uses ONNX and its `Python API <https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md>`_.

Key ONNX resources:

- `ONNX protobuf description <https://github.com/onnx/onnx/blob/main/onnx/onnx.proto>`_
- `Human-readable IR documentation <https://github.com/onnx/onnx/blob/main/docs/IR.md>`_
- `Operator schemas <https://github.com/onnx/onnx/blob/main/docs/Operators.md>`_

See the :ref:`tutorials` chapter for a Jupyter notebook that demonstrates working with ONNX models in FINN.

.. note:: FINN supports two specialized variants of ONNX called **QONNX** and **FINN-ONNX**, and not all ONNX graphs are supported by FINN (and vice versa).

QONNX vs FINN-ONNX
-------------------

**QONNX** represents quantization using explicit Quant and BipolarQuant nodes. This format is used for models exported from Brevitas and during the early stages of the FINN flow.

**FINN-ONNX** uses quantization annotations (via the quantization_annotation field in ONNX) to annotate tensors with their FINN DataType information. This is FINN's internal representation during compilation.

See the `QONNX repository <https://github.com/fastmachinelearning/qonnx>`_ for details on QONNX.

Custom Quantization Annotations
================================

Standard ONNX does not support arbitrary-precision integer datatypes. FINN supports arbitrary integer quantization (e.g., 1-bit bipolar, 3-bit, 5-bit, up to 32-bit and beyond). To support this, FINN-ONNX uses quantization annotations to attach FINN DataType (:py:mod:`qonnx.core.datatype.DataType`) information to tensors.

**Key principle**: All tensors use single-precision floating point (float32) as the **container datatype**, even for 1-bit values. The FINN DataType annotation specifies the actual bit width and signedness. The FINN compiler flow produces packed representations for target hardware.

Floating Point as Carrier Datatype
-----------------------------------

FINN uses floating point tensors as a carrier data type to represent integers. Floating point arithmetic can introduce rounding errors, e.g., ``(int_num * float_scale) / float_scale`` is not always equal to ``int_num``.

When using the custom ONNX execution flow, FINN will attempt to sanitize rounding errors for integer tensors. See :py:mod:`qonnx.util.basic.sanitize_quant_values` for more information.

This behavior can be disabled (not recommended) by setting the environment variable ``SANITIZE_QUANT_TENSORS=0``.

Custom Operations (CustomOps)
==============================

FINN uses many custom operations (``op_type`` in ONNX NodeProto) that are not defined in the ONNX operator schema. These custom nodes are marked with ``domain="finn.*"`` or ``domain="qonnx.*"`` in the protobuf to identify them as such.

Custom operations can represent:

- Specific operations needed for low-bit networks (e.g., MultiThreshold, Bipolar quantization)
- Operations specific to a particular hardware backend (e.g., MatrixVectorActivation, ConvolutionInputGenerator)
- Graph organization nodes (e.g., StreamingDataflowPartition)

See the CustomOps tutorial in :ref:`tutorials` or the :py:mod:`finn.custom_op` module for details. For implementing new CustomOps, see the :doc:`/implementation/index`.

Custom ONNX Execution Flow
===========================

To verify correct operation of FINN-ONNX graphs, FINN provides its own ONNX execution flow (:py:mod:`finn.core.onnx_exec`). This flow supports the standard set of ONNX operations as well as the custom FINN operations.

.. warning:: This execution flow is only meant for checking the correctness of models after applying transformations, and not for high performance inference.

.. _modelwrapper:

ModelWrapper
============

FINN provides a ModelWrapper class (:py:mod:`qonnx.core.modelwrapper.ModelWrapper`) as a thin wrapper around ONNX to make it easier to analyze and manipulate ONNX graphs. This wrapper provides many helper functions, while still giving full access to the ONNX protobuf representation.

Creating a ModelWrapper
------------------------

The ModelWrapper instance can be created from a ``.onnx`` file or by directly passing a ModelProto instance:

.. code-block:: python

    from qonnx.core.modelwrapper import ModelWrapper
    model = ModelWrapper("model.onnx")

Accessing the Graph
--------------------

Access the ONNX ModelProto:

.. code-block:: python

    modelproto = model.model

Access the graph:

.. code-block:: python

    graphproto = model.graph

Access the node list:

.. code-block:: python

    nodes = model.graph.node
    first_node = nodes[0]
    num_nodes = len(nodes)

Tensor Operations
-----------------

**List all tensor names:**

.. code-block:: python

    tensor_list = model.get_all_tensor_names()

**Find producer/consumer nodes:**

.. code-block:: python

    # Find producer of third tensor
    model.find_producer(tensor_list[2])

    # Find consumer of third tensor
    model.find_consumer(tensor_list[2])

If a tensor does not have a producer or consumer node (e.g., it's a constant), ``None`` is returned.

**Get/set tensor shape:**

.. code-block:: python

    # Get tensor shape
    shape = model.get_tensor_shape(tensor_list[2])

    # Set tensor shape
    tensor_shape = [1, 1, 28, 28]
    model.set_tensor_shape(tensor_list[2], tensor_shape)

Optionally, the dtype (container datatype) can be specified as a third argument. By default it is set to ``TensorProto.FLOAT``.

**Get/set FINN DataType:**

.. code-block:: python

    from qonnx.core.datatype import DataType

    # Get FINN DataType
    finn_dtype = model.get_tensor_datatype(tensor_list[2])

    # Set FINN DataType
    model.set_tensor_datatype(tensor_list[2], DataType["BIPOLAR"])

**Get tensor initializer:**

.. code-block:: python

    # Get initializer (returns None if no initializer exists)
    initializer = model.get_initializer(tensor_list[2])

See :py:mod:`qonnx.core.modelwrapper.ModelWrapper` for the complete API.

.. _analysis_pass:

Analysis Passes
===============

An analysis pass traverses the graph structure and produces information about certain properties. It receives a ModelWrapper as input and returns a dictionary of extracted properties.

**Purpose**: Extract information without modifying the model (e.g., resource estimates, performance metrics, node counts).

**Examples**:

- ``op_and_param_counts`` - Counts operations and parameters
- ``exp_cycles_per_layer`` - Reports expected cycles per layer
- ``res_estimation`` - Estimates FPGA resource usage

See the `custom analysis pass notebook <https://github.com/Xilinx/finn/blob/main/notebooks/advanced/0_custom_analysis_pass.ipynb>`_ for a tutorial on writing analysis passes, and :py:mod:`finn.analysis` for existing implementations.

.. _transformation_pass:

Transformation Passes
=====================

A transformation pass changes (transforms) the given model. It receives a ModelWrapper as input and returns:

1. The modified ModelWrapper
2. A ``model_was_changed`` flag indicating if the transformation should be applied again

**Purpose**: Progressively lower the model from high-level operations to hardware-ready operators.

**Examples**:

- ``InferShapes`` - Propagates tensor shapes through the graph
- ``InferDataTypes`` - Propagates FINN DataTypes through the graph
- ``ConvertToHWLayers`` - Converts high-level operations to hardware layers
- ``SpecializeLayers`` - Selects HLS vs RTL backend for each layer
- ``InsertFIFO`` - Inserts streaming FIFOs between layers

See the `custom transformation pass notebook <https://github.com/Xilinx/finn/blob/main/notebooks/advanced/1_custom_transformation_pass.ipynb>`_ for a tutorial on writing transformation passes, and :py:mod:`finn.transformation` for existing implementations.

Transformation Patterns
-----------------------

Transformations in FINN typically fall into these categories:

**Graph rewriting**
    Replace subgraphs with equivalent but more optimized representations (e.g., ``AbsorbAddIntoMultiThreshold``, ``MoveFlattenPastAffine``)

**Shape/datatype inference**
    Propagate tensor properties through the graph (e.g., ``InferShapes``, ``InferDataTypes``)

**Hardware conversion**
    Convert high-level operations to hardware-specific implementations (e.g., ``ConvertToHWLayers``, ``SpecializeLayers``)

**Optimization**
    Apply performance or resource optimizations (e.g., ``SetFolding``, ``MinimizeAccumulatorWidth``)

**IP generation**
    Generate HDL code for hardware layers (e.g., ``PrepareIP``, ``HLSSynthIP``, ``CreateStitchedIP``)

See :doc:`/implementation/index` for guidance on implementing new transformation passes.

See Also
========

- :doc:`/implementation/index` - Extending FINN with new operators and transformations
- :ref:`tutorials` - Jupyter notebooks with hands-on examples
- :py:mod:`qonnx.core.modelwrapper.ModelWrapper` - ModelWrapper API documentation
- :py:mod:`finn.transformation` - Transformation pass implementations
- :py:mod:`finn.analysis` - Analysis pass implementations
