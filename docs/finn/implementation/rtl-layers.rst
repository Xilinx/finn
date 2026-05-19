.. _rtl_layers:

*************************************
Implementing RTL Variants
*************************************

This guide explains how to implement RTL backend variants for FINN hardware layers.

Overview
========

RTL variants generate SystemVerilog/Verilog HDL code that instantiates finn-rtllib modules. Unlike HLS layers which share common templates, **each RTL layer requires its own Verilog wrapper template** in finn-rtllib.

**Location**: ``src/finn/custom_op/fpgadataflow/rtl/<layer>_rtl.py``

**Inherits from**: Base layer + ``RTLBackend``

Verilog Wrapper Requirement
============================

Vivado IP Integrator expects Verilog (``.v``) files during IP stitching, even though finn-rtllib components are written in SystemVerilog (``.sv``).

So, each RTL layer needs:

1. **SystemVerilog implementation** (e.g., ``layernorm.sv``) - The actual hardware logic
2. **Verilog wrapper template** (e.g., ``layernorm_wrapper_template.v``) - AXI interface wrapper for Vivado

**Template keywords**: Each wrapper has layer-specific keywords (e.g., ``$N$``, ``$SIMD$``, ``$TOP_MODULE_NAME$``) that get replaced during code generation.

Example: LayerNorm_rtl
=======================

**Wrapper template**: ``finn-rtllib/layernorm/layernorm_wrapper_template.v``

- Contains Verilog module with AXI stream interface
- Instantiates ``layernorm.sv`` SystemVerilog module
- Uses template keywords: ``$TOP_MODULE_NAME$``, ``$N$``, ``$SIMD$``

**RTL layer implementation**: ``src/finn/custom_op/fpgadataflow/rtl/layernorm_rtl.py``

- Inherits from ``LayerNorm`` + ``RTLBackend``
- ``generate_hdl()`` creates ``code_gen_dict`` with layer parameters and substitutes into template
- Copies SystemVerilog files (``layernorm.sv``, ``queue.sv``, ``accuf.sv``, etc.) to code_gen_dir
- Returns HDL file list and Vivado IPI TCL commands

**Reference implementation**: See ``src/finn/custom_op/fpgadataflow/rtl/layernorm_rtl.py`` for complete example.

Required Methods
================

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Purpose
   * - ``generate_hdl(model, fpgapart, clk)``
     - Generate Verilog wrapper from template, copy SystemVerilog files
   * - ``get_rtl_file_list(abspath)``
     - Return list of all HDL files (SystemVerilog + Verilog wrapper)
   * - ``code_generation_ipi()``
     - Generate Vivado IPI TCL commands (add_files + create_bd_cell)

RTL Simulation
==============

RTL variants can provide rtlsim execution using XSI (Xilinx Simulator Interface). The ``execute_node()`` method dispatches based on ``exec_mode``:

- ``cppsim``: Falls back to base layer Python execution
- ``rtlsim``: Uses ``RTLBackend.execute_node()`` via XSI

Adding a New RTL Layer
======================

**Step 1: Create finn-rtllib module**

1. Create directory ``finn-rtllib/<layer_name>/``
2. Implement SystemVerilog module ``<layer_name>.sv``
3. Create Verilog wrapper template ``<layer_name>_wrapper_template.v`` with layer-specific keywords
4. Add any helper SystemVerilog modules needed

**Step 2: Create RTL variant in FINN**

1. Create ``src/finn/custom_op/fpgadataflow/rtl/<layer>_rtl.py``
2. Inherit from base layer + ``RTLBackend``
3. Implement ``generate_hdl()`` to:
   - Define ``code_gen_dict`` with template keywords
   - Substitute keywords into wrapper template
   - Copy SystemVerilog files to code_gen_dir
4. Implement ``get_rtl_file_list()`` to list all HDL files
5. Implement ``code_generation_ipi()`` to generate TCL commands

**Step 3: Register in FINN**

1. Add import to ``src/finn/custom_op/fpgadataflow/rtl/__init__.py``
2. Update SpecializeLayers if needed
3. Add tests in ``tests/fpgadataflow/``

See Also
========

- :doc:`customop-pattern` - CustomOp class hierarchy
- :doc:`hls-layers` - Implementing HLS variants
- :doc:`specialization-rules` - HLS vs RTL selection
- `HDL_STYLE_GUIDE.md <https://github.com/Xilinx/finn/blob/dev/HDL_STYLE_GUIDE.md>`_ - SystemVerilog coding conventions
- `finn-rtllib <https://github.com/Xilinx/finn-rtllib>`_ - RTL module library
