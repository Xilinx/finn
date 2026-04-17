.. _folding_factors:

*************************************
Folding Factor Constraints
*************************************

This page documents the divisibility constraints for PE and SIMD folding parameters for each FINN hardware layer.

Overview
========

Folding parameters (PE and SIMD) control the degree of parallelism for each hardware layer. These parameters must satisfy specific divisibility constraints based on the layer's dimensions.

**Common parameters:**

- **PE**: Processing Elements - controls output parallelism
- **SIMD**: Single Instruction Multiple Data - controls input parallelism

**Constraint notation:**

- ``%`` denotes the modulo operator (must divide evenly)
- ``MH``, ``MW``: Matrix height/width (for MVAU)
- ``k_h``, ``k_w``: Kernel height/width (for VVAU)

Constraint Table
================

.. list-table:: Folding factor constraints
   :header-rows: 1
   :widths: 30 20 50

   * - **Layer**
     - **Parameters**
     - **Constraints**
   * - Addstreams (deprecated, use ElementwiseBinary)
     - PE
     - inp_channels % PE == 0
   * - ChannelwiseOp (deprecated, use ElementwiseBinary)
     - PE
     - channels % PE == 0
   * - ConvolutionInputGenerator
     - SIMD
     - inp_channels % SIMD == 0
   * - DuplicateStreams
     - PE
     - channels % PE == 0
   * - ElementwiseBinary
     - PE
     - last_dim % PE == 0
   * - StreamingEltwise (deprecated, use ElementwiseBinary)
     - PE
     - inp_channels % PE == 0
   * - FMPadding
     - SIMD
     - inp_channels % SIMD == 0
   * - FMPadding_Pixel
     - SIMD
     - inp_channels % SIMD == 0
   * - Globalaccpool
     - PE
     - channels % PE == 0
   * - Labelselect
     - PE
     - num_labels % PE == 0
   * - MatrixVectorActivation
     - PE & SIMD
     - MH % PE == 0 & MW % SIMD == 0
   * - Pool
     - PE
     - inp_channels % PE == 0
   * - Thresholding
     - PE
     - MH % PE == 0
   * - Upsampler
     - SIMD
     - inp_channels % SIMD == 0
   * - VectorVectorActivation
     - PE & SIMD
     - k_h * k_w % SIMD == 0 & channels % PE == 0

Critical Multi-Layer Constraints
=================================

Some layer combinations require matching parallelism to ensure functional correctness:

.. warning::
   **ConvolutionInputGenerator paired with VVAU or Pool:**

   The SIMD parameter of ConvolutionInputGenerator **must match** the PE parameter of the following layer (VVAU or Pool).

   - ConvolutionInputGenerator SIMD == VVAU PE
   - ConvolutionInputGenerator SIMD == Pool PE

   **Violating this constraint causes functional incorrectness**, not just performance issues.

Usage
=====

These constraints are enforced by the ``SetFolding`` transformation and must be satisfied when manually configuring folding parameters via the folding config JSON file.

See Also
========

- :ref:`nw_prep` - Network preparation and folding configuration
- :py:mod:`finn.transformation.fpgadataflow.set_folding.SetFolding` - Automatic folding transformation
