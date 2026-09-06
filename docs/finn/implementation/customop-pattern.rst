.. _customop_pattern:

*************************************
CustomOp Class Hierarchy
*************************************

FINN uses a class hierarchy for hardware operators that separates backend-agnostic functionality from backend-specific code generation.

Typical Pattern
================

Most FINN hardware operators follow this structure:

.. code-block:: text

    ┌─────────────────┐
    │   CustomOp      │  (from qonnx - abstract base)
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │  HWCustomOp     │  (FINN abstract base for HW operators)
    └────────┬────────┘
             │
             │                  ┌──────────────┐
             │                  │  HLSBackend  │  (abstract mixin)
             │                  └──────────────┘
             │
    ┌────────▼────────┐                ┌──────────────┐
    │  LayerNorm      │                │  RTLBackend  │  (abstract mixin)
    │ (Base Layer)    │                └──────────────┘
    └────────┬────────┘
             │
             ├──────────────────┐
             │                  │
    ┌────────▼────────┐  ┌──────▼──────┐
    │ LayerNorm_hls   │  │LayerNorm_rtl│
    │ (LayerNorm+     │  │ (LayerNorm+ │
    │  HLSBackend)    │  │  RTLBackend)│
    └─────────────────┘  └─────────────┘

**Four classes involved per operator**:

1. **HWCustomOp** - Abstract base class providing common hardware operator interface
2. **HLSBackend / RTLBackend** - Abstract mixin classes for code generation
3. **Base Layer** (e.g., ``LayerNorm``) - Concrete backend-agnostic implementation
4. **Backend Variants** (e.g., ``LayerNorm_hls``, ``LayerNorm_rtl``) - Backend-specific code generation

This separation allows:

- Sharing common logic across backends (shape calculations, execution semantics)
- Adding new backends without duplicating functionality
- Testing operator semantics independently of hardware generation
- Operators can have one or both backend implementations

Base Layer (Backend-Agnostic)
==============================

**Location**: ``src/finn/custom_op/fpgadataflow/<layer>.py``

**Naming**: PascalCase (e.g., ``LayerNorm``, ``MatrixVectorActivation``, ``FMPadding``)

**Inherits from**: ``HWCustomOp``

**Responsibilities**:

- Define node attributes via ``get_nodeattr_types()``
- Implement shape calculations (``get_normal_input_shape()``, ``get_folded_output_shape()``)
- Calculate stream widths (``get_instream_width()``, ``get_outstream_width()``)
- Provide Python golden reference execution (``execute_node()``)
- Define number of inputs/outputs
- Implement any backend-agnostic helper methods

**Example**: ``src/finn/custom_op/fpgadataflow/layernorm.py``

.. code-block:: python

    from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp

    class LayerNorm(HWCustomOp):
        """Base class for LayerNorm operator."""

        def get_nodeattr_types(self):
            """Define node attributes for LayerNorm layer."""
            my_attrs = {
                "N": ("i", True, 0),  # Number of elements to normalize
                "SIMD": ("i", True, 0),  # Parallelism factor
                "InputDataType": ("s", True, ""),
                "WeightDataType": ("s", True, ""),
            }
            my_attrs.update(super().get_nodeattr_types())
            return my_attrs

        def get_folded_output_shape(self, ind=0):
            """Return folded output shape with SIMD dimension."""
            n = self.get_nodeattr("N")
            simd = self.get_nodeattr("SIMD")
            folded_oshape = (1, n // simd, simd)
            return folded_oshape

        def execute_node(self, context, graph):
            """Execute this node in Python (golden reference)."""
            # Implementation of layer normalization in numpy for verification
            ...

**Key Methods to Implement**:

- ``get_nodeattr_types()`` - Define all node attributes
- ``get_normal_input_shape()`` / ``get_folded_input_shape()`` - Input tensor shapes
- ``get_normal_output_shape()`` / ``get_folded_output_shape()`` - Output tensor shapes
- ``get_instream_width()`` / ``get_outstream_width()`` - Stream widths in bits
- ``execute_node()`` - Python execution for verification

Node Attribute Best Practices
------------------------------

**When to add node attributes:**

- Only add node attributes for information that **cannot be easily computed** from other attributes or the graph
- Computed values should be **methods**, not stored attributes
- Choose appropriate scope: node attributes are layer-specific; use transformation parameters for global config

**Example:**

- **Store**: ``NumChannels``, ``PE`` (fundamental layer-specific parameters)
- **Compute**: ``TMEM = NumChannels / PE`` (implement as ``get_tmem()`` method)
- **Don't store**: Clock period as a node attribute (global parameter, pass to transformations instead)

HLS Backend Variant
====================

**Location**: ``src/finn/custom_op/fpgadataflow/hls/<layer>_hls.py``

**Naming**: Base name + ``_hls`` suffix (e.g., ``LayerNorm_hls``, ``MVAU_hls``)

**Inherits from**: Base layer + ``HLSBackend``

**Responsibilities**:

- Generate HLS C++ code that calls finn-hlslib templates
- Define include directives, template parameters, function calls
- Add HLS pragmas
- Generate weight/threshold parameters if needed

See :doc:`hls-layers` for detailed implementation guide.

RTL Backend Variant
====================

**Location**: ``src/finn/custom_op/fpgadataflow/rtl/<layer>_rtl.py``

**Naming**: Base name + ``_rtl`` suffix (e.g., ``LayerNorm_rtl``, ``MVAU_rtl``)

**Inherits from**: Base layer + ``RTLBackend``

**Responsibilities**:

- Generate SystemVerilog/Verilog HDL code
- Instantiate finn-rtllib modules or generate custom HDL
- Define HDL file lists and Vivado IPI TCL commands
- Provide rtlsim execution if applicable

See :doc:`rtl-layers` for detailed implementation guide.

Alternative Patterns
====================

While most operators follow the typical pattern above, some special cases exist:

Backend-Specific Operators
---------------------------

Some operators only have one backend implementation and are infrastructure ops rather than compute ops. They combine the base layer with the backend in a single class:

**HLS-only operators**:

- ``IODMA_hls(HWCustomOp, HLSBackend)`` - DMA operator
- ``CheckSum_hls(HWCustomOp, HLSBackend)`` - Checksum verification
- ``TLastMarker_hls(HWCustomOp, HLSBackend)`` - AXI stream TLAST marker

**RTL-only operators**:

- ``FINNLoop(HWCustomOp, RTLBackend)`` - Loop control operator


Non-Hardware Operators
-----------------------

Some custom operators don't represent synthesizable hardware:

- ``StreamingDataflowPartition(CustomOp)`` - Graph partitioning marker

These inherit directly from ``CustomOp`` (from qonnx) rather than ``HWCustomOp``, as they're used for graph organization rather than hardware generation.

Specialization: Choosing HLS vs RTL
====================================

The ``SpecializeLayers`` transformation converts base layers to specific HLS or RTL variants based on:

- FPGA part (determines available DSP primitives)
- Datatype constraints (bit widths, signed/unsigned)
- User preference via ``preferred_impl_style`` node attribute

See :doc:`specialization-rules` for details on the selection logic.

Adding a New CustomOp
======================

**Step 1: Create Base Layer**

1. Create ``src/finn/custom_op/fpgadataflow/<layer>.py``
2. Inherit from ``HWCustomOp``
3. Define ``get_nodeattr_types()`` with all configuration parameters
4. Implement shape calculation methods
5. Implement ``execute_node()`` for Python golden reference
6. Add import to ``src/finn/custom_op/fpgadataflow/__init__.py``

**Step 2: Add HLS Variant (Optional)**

1. Create ``src/finn/custom_op/fpgadataflow/hls/<layer>_hls.py``
2. Inherit from base layer + ``HLSBackend``
3. Implement code generation methods
4. Ensure finn-hlslib has the required C++ template (or add it)
5. Add import to ``src/finn/custom_op/fpgadataflow/hls/__init__.py``

**Step 3: Add RTL Variant (Optional)**

1. Create ``src/finn/custom_op/fpgadataflow/rtl/<layer>_rtl.py``
2. Inherit from base layer + ``RTLBackend``
3. Implement HDL generation methods
4. Ensure finn-rtllib has the required SystemVerilog module (or add it)
5. Add import to ``src/finn/custom_op/fpgadataflow/rtl/__init__.py``

**Step 4: Add Specialization Rules**

Update ``src/finn/transformation/fpgadataflow/specialize_layers.py`` to include rules for when to use HLS vs RTL for your new layer.

**Step 5: Add Tests**

Create tests in ``tests/fpgadataflow/test_<layer>.py`` covering:

- Base layer execution (Python golden reference)
- HLS variant (cppsim, rtlsim)
- RTL variant (rtlsim)

See Also
========

- :doc:`hls-layers` - Detailed HLS code generation guide
- :doc:`rtl-layers` - Detailed RTL code generation guide
- :doc:`specialization-rules` - HLS vs RTL selection rules
