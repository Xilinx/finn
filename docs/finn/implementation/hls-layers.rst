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
- ``docompute_template_timeout`` - Used for cppsim (C++ simulation). HLS layers are
  free-running: the simulation calls the top function until the expected outputs have
  arrived or ``$TIMEOUT_VALUE$`` cycles pass without a new one.

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

Stream Interface
================

``HLSBackend`` declares all streams as finn-hlslib tensor streams,
``hls::stream<tensor<T, N>>``, one tensor of ``N`` elements (the folding factor) per
transaction. In cppsim, ``npy2tensorstream`` and ``tensorstream2npy`` convert between
these streams and the ``.npy`` files of the execution context. The defaults of
``read_npy_data()``, ``strm_decl()``, ``dataoutstrm()`` and ``pragmas()`` cover layers
with streamed inputs and one output; override them only if a layer differs, e.g. in
the number of streams. The ``cpp_interface`` and ``hls_style`` attributes of earlier
versions no longer exist.

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
- `style_cpp.md <https://github.com/Xilinx/finn/blob/dev/rules/style_cpp.md>`_ and `style_hls.md <https://github.com/Xilinx/finn/blob/dev/rules/style_hls.md>`_ - HLS coding conventions
- `finn-hlslib <https://github.com/Xilinx/finn-hlslib>`_ - HLS template library
