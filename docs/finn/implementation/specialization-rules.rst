.. _specialization_rules:

*************************************
SpecializeLayers: HLS vs RTL Selection
*************************************

The ``SpecializeLayers`` transformation converts backend-agnostic base layers into specific HLS or RTL variants based on constraints and preferences.

**Location**: ``src/finn/transformation/fpgadataflow/specialize_layers.py``

**When it runs**: During the build flow, after hardware layer conversion

Overview
========

SpecializeLayers replaces base layers (e.g., ``LayerNorm``) with backend-specific variants (``LayerNorm_hls`` or ``LayerNorm_rtl``) by checking:

1. User preference (``preferred_impl_style`` node attribute)
2. FPGA part capabilities
3. Datatype constraints
4. Layer-specific requirements

Selection Logic
===============

For each base layer node:

1. **Check if already specialized** → Skip if node is already ``*_hls`` or ``*_rtl``
2. **Check user preference** → Use ``preferred_impl_style`` if set and valid
3. **Apply constraints** → Determine which backends are available
4. **Choose default** → RTL preferred if available and constraints satisfied

**Auto-detection**: If a layer has both HLS and RTL variants available, RTL is preferred when constraints are satisfied. Otherwise, HLS is used as fallback.

RTL Constraint Table
=====================

Layers with conditional RTL availability:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Layer
     - RTL Constraints
   * - MVAU
     - 2-8 bit signed weights/activations, no embedded threshold
   * - VVAU
     - Versal only, ≤8 bit signed weights
   * - LayerNorm
     - Versal only, FLOAT32 only
   * - ElementwiseBinary
     - Versal only, FLOAT32 only
   * - StreamingDataWidthConverter
     - Integer width ratio only

Example: LayerNorm Specialization
==================================

**Base layer**: ``LayerNorm(HWCustomOp)``

**Available variants**:

- ``LayerNorm_hls(LayerNorm, HLSBackend)`` - Always available
- ``LayerNorm_rtl(LayerNorm, RTLBackend)`` - Versal + FLOAT32 only

**Constraint check**:

.. code-block:: python

    def _suitable_for_layernorm_rtl(node, fpgapart):
        """Check if LayerNorm can use RTL backend."""
        # Check FPGA part
        if not is_versal(fpgapart):
            return False

        # Check datatype
        idt = DataType[node.get_nodeattr("InputDataType")]
        if idt != DataType["FLOAT32"]:
            return False

        return True

**Selection flow**:

1. User sets ``preferred_impl_style = "rtl"`` on LayerNorm node
2. SpecializeLayers checks: Is FPGA Versal? Is datatype FLOAT32?
3. If both true → Replace with ``LayerNorm_rtl``
4. If either false → Fall back to ``LayerNorm_hls`` with warning

Adding Specialization for New Layers
=====================================

When adding a new layer with multiple backends:

**Step 1**: Implement constraint check function

.. code-block:: python

    def _suitable_for_mylayer_rtl(node, fpgapart):
        """Check if MyLayer satisfies RTL constraints."""
        # Check FPGA capabilities
        if not meets_fpga_requirements(fpgapart):
            return False

        # Check datatypes
        idt = DataType[node.get_nodeattr("InputDataType")]
        if not idt.is_integer():
            return False

        # Add other layer-specific checks
        return True

**Step 2**: Add to SpecializeLayers transformation

.. code-block:: python

    elif node.op_type == "MyLayer":
        # Check user preference
        preferred = node.get_nodeattr("preferred_impl_style")

        # Determine backend
        if preferred == "rtl" and _suitable_for_mylayer_rtl(node, fpgapart):
            new_op_type = "MyLayer_rtl"
        elif preferred == "hls":
            new_op_type = "MyLayer_hls"
        elif _suitable_for_mylayer_rtl(node, fpgapart):
            new_op_type = "MyLayer_rtl"  # RTL default if available
        else:
            new_op_type = "MyLayer_hls"  # HLS fallback

        # Replace node
        new_node = make_new_node(new_op_type, ...)

See Also
========

- :doc:`customop-pattern` - CustomOp class hierarchy
- :doc:`hls-layers` - Implementing HLS variants
- :doc:`rtl-layers` - Implementing RTL variants
