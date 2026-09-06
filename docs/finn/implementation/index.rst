.. _implementation:

**********************
Implementation Guide
**********************

This guide is intended for developers who want to extend FINN by adding new operators, transformation passes, or analysis passes.

If you're new to FINN development, we recommend:

1. Reading the :doc:`/concepts` documentation to understand FINN's core concepts (IR, ModelWrapper, transformations)
2. Reviewing the :doc:`/developers` guide for contribution workflow
3. Consulting the `PYTHON_STYLE_GUIDE.md <https://github.com/Xilinx/finn/blob/dev/PYTHON_STYLE_GUIDE.md>`_ and `HDL_STYLE_GUIDE.md <https://github.com/Xilinx/finn/blob/dev/HDL_STYLE_GUIDE.md>`_ for coding standards

.. toctree::
   :maxdepth: 2

   customop-pattern
   specialization-rules
   hls-layers
   rtl-layers

Getting Started
===============

**Adding a new hardware layer:**

1. Start with :doc:`customop-pattern` to understand the CustomOp class hierarchy
2. See :doc:`hls-layers` for HLS code generation or :doc:`rtl-layers` for RTL code generation
3. See :doc:`specialization-rules` to add selection logic for HLS vs RTL

**Adding transformation or analysis passes:**

- `Custom transformation pass notebook <https://github.com/Xilinx/finn/blob/main/notebooks/advanced/1_custom_transformation_pass.ipynb>`_
- `Custom analysis pass notebook <https://github.com/Xilinx/finn/blob/main/notebooks/advanced/0_custom_analysis_pass.ipynb>`_
