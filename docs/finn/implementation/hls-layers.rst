.. _hls_layers:

*************************************
Implementing HLS Variants
*************************************

This guide explains how to implement HLS backend variants for FINN hardware layers.

Overview
========

HLS variants generate C++ code that calls finn-hlslib template functions. The generated code is synthesized by Vitis HLS to create RTL IP blocks.

**Location**: ``src/finn/custom_op/fpgadataflow/hls/<layer>_hls.py``

**Inherits from**: Base layer + ``HLSBackend``

**Key responsibility**: Populate the ``code_gen_dict`` dictionary with code fragments that are assembled into a complete HLS C++ file.

Code Generation Dictionary
===========================

The HLS backend infrastructure uses a template-based system. Your HLS variant populates ``self.code_gen_dict`` with code fragments that are substituted into templates defined in ``src/finn/custom_op/fpgadataflow/templates.py``.

**Templates**:

- ``ipgen_template`` - Used for IP generation (synthesis)
- ``docompute_template`` - Used for cppsim (C++ simulation)
- ``docompute_template_timeout`` - Used for cppsim with freerunning HLS operations

Required Methods
================

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Method
     - Template Key
     - Purpose
   * - ``global_includes()``
     - ``$GLOBALS$``
     - Add #include directives for finn-hlslib headers
   * - ``defines(var)``
     - ``$DEFINES$``
     - Define constants for HLS template parameters (#define statements)
   * - ``docompute()``
     - ``$DOCOMPUTE$``
     - Generate main function call to finn-hlslib template
   * - ``blackboxfunction()``
     - ``$BLACKBOXFUNCTION$``
     - Generate function signature with stream parameters
   * - ``pragmas()``
     - ``$PRAGMAS$``
     - Add HLS interface pragmas (axis ports, control protocol)

Optional Methods
================

- ``generate_params(model, path)`` - Generate weight/threshold parameter files (needed for MVAU, VVAU, Thresholding)

**Reference implementation**: See ``src/finn/custom_op/fpgadataflow/hls/layernorm_hls.py`` for a complete example.

HLS-Specific Node Attributes
=============================

The ``HLSBackend`` base class defines node attributes that control code generation:

cpp_interface
-------------

**Purpose**: Determines how data is packed into HLS streams

**Values**:
- ``"packed"`` (default) - Data packed as ``ap_uint<width>`` bit vectors
- ``"hls_vector"`` - Data as HLS vector types (``hls::vector``)

**Effect**: Influences how ``$STREAMDECLARATIONS$`` and data reading/writing are generated

**Recommendation**: Use ``"hls_vector"`` for new HLS components going forward.

hls_style
---------

**Purpose**: Determines execution style for HLS operations

**Values**:
- ``"ifm_aware"`` (default) - Kernel knows the input feature map size and when all outputs are computed
- ``"freerunning"`` - Free-running execution with timeout-based control

**Effect**:
- Selects which cppsim template is used (``docompute_template`` vs ``docompute_template_timeout``)
- Changes stream declarations and output handling

**Note**: These attributes are set during hardware layer conversion (``convert_to_hw_layers`` transformations). When implementing a new HLS layer, you will need to set these attributes appropriately.

Generated Code Structure
========================

The code generation produces a C++ file with this structure:

.. code-block:: cpp

    // From $GLOBALS$
    #include "layernorm.hpp"

    // From $DEFINES$
    #define N 128
    #define SIMD 16

    // From $BLACKBOXFUNCTION$ and $PRAGMAS$
    void LayerNorm_0(
        hls::stream<ap_uint<512>> &in0,
        hls::stream<ap_uint<512>> &out)
    {
        #pragma HLS INTERFACE axis port=in0
        #pragma HLS INTERFACE axis port=out
        #pragma HLS INTERFACE ap_ctrl_none port=return

        // From $DOCOMPUTE$
        layernorm<N, SIMD>(in0, weights, out);
    }

This file is then synthesized by Vitis HLS to create an RTL IP block.

Adding to FINN
==============

After implementing your HLS variant:

1. Add import to ``src/finn/custom_op/fpgadataflow/hls/__init__.py``
2. Update SpecializeLayers if needed
3. Add tests in ``tests/fpgadataflow/``


See Also
========

- :doc:`customop-pattern` - CustomOp class hierarchy
- :doc:`rtl-layers` - Implementing RTL variants
- :doc:`specialization-rules` - HLS vs RTL selection
- `HDL_STYLE_GUIDE.md <https://github.com/Xilinx/finn/blob/dev/HDL_STYLE_GUIDE.md>`_ - HLS coding conventions
- `finn-hlslib <https://github.com/Xilinx/finn-hlslib>`_ - HLS template library
