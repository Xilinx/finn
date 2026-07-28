PWPolyF Piecewise Polynomial Activation
=======================================

Overview
--------

PWPolyF is a hardware activation layer that approximates nonlinear functions
(GELU, SiLU, Sigmoid, Tanh) using piecewise polynomials evaluated with
Horner's method on a chain of DSPFP32 FMA units. With the default degree of 2,
this uses two cascaded DSPs and one RAMB18 coefficient ROM per PE, giving
single-cycle-per-element throughput. Per-function configuration, including
clamping behaviour and polynomial coefficients, is delivered through a
SystemVerilog package (``pwpolyf_pkg``) using a ``func_cfg_t`` struct.

The input domain is partitioned into ``1 + 2*5*(2^K)`` segments: one near-zero
region, positive octave sub-segments, and negative mirrors. With the default
``K=3`` this gives 81 segments. Segment selection reuses the FP32 exponent and
mantissa bit fields directly, matching the RTL implementation.

Polynomial coefficients are generated at HDL build time by
``PWPolyF_rtl._generate_coeffs_pkg()``, which fits polynomials of the
configured degree to the reference PyTorch functions and writes
``pwpolyf_pkg.sv``. Both ``K`` and ``degree`` are configurable. They default to
``K=3`` and ``degree=2`` when inferred from standard ONNX ops.

Architecture
------------

PWPolyF is RTL-only, with no HLS variant, and targets Versal devices only. The
RTL instantiates the Versal DSPFP32 primitive, so UltraScale+ and older parts
must not be specialized to this backend.

Two export paths are supported:

.. code-block:: text

   Path A: Brevitas PWPolyFGELU /       Path B: nn.GELU / nn.SiLU / etc.
           PWPolyFSiLU / etc.               |  torch.onnx.export
       |  export_finn_onnx                  |  (dynamo=True or False)
       |  (dynamo=True or False)            v
       v                               Standard ONNX ops (Gelu, Sigmoid,
   finn.pwpolyf::PWPolyF node          Tanh, Sigmoid+Mul for SiLU,
       |                               Div+Erf+Add+Mul+Mul for GELU)
       |                                      |
       +------------- both paths -------------+
                         |
                   InferPWPolyFLayer
                         v
               PWPolyF HW op (finn.custom_op.fpgadataflow)
                         |  SpecializeLayers
                         v
               PWPolyF_rtl (finn.custom_op.fpgadataflow.rtl)
                         |  generate_hdl
                         v
               finn-rtllib/pwpolyf/hdl/ SystemVerilog IP

Standard ONNX Op Inference
--------------------------

``InferPWPolyFLayer`` recognises standard ONNX activation ops in addition to
the explicit ``PWPolyF`` custom op. This allows models that use ``nn.GELU``,
``nn.SiLU``, ``nn.Sigmoid``, or ``nn.Tanh`` to be exported with ``dynamo=True``
or ``dynamo=False`` and automatically converted to PWPolyF HW layers.

.. list-table::
   :header-rows: 1
   :widths: 20 45 20

   * - ONNX op type
     - Pattern
     - Maps to
   * - ``Gelu`` (opset 20+)
     - Single node
     - ``func="gelu"``
   * - ``Div`` + ``Erf`` + ``Add`` + ``Mul`` + ``Mul``
     - ``x * 0.5 * (1 + erf(x / sqrt(2)))``
     - ``func="gelu"``
   * - ``Sigmoid``
     - Single node (standalone)
     - ``func="sigmoid"``
   * - ``Tanh``
     - Single node
     - ``func="tanh"``
   * - ``Sigmoid`` + ``Mul``
     - ``Mul(x, Sigmoid(x))``
     - ``func="silu"``

``Gelu`` as a single ONNX node requires opset 20 or later. With lower opsets,
including ``dynamo=True`` export defaults to opset 18, GELU decomposes into a
5-node Erf-based pattern. Both forms are matched. SiLU has no standard ONNX op
and decomposes to ``Sigmoid(x) * x``. Only FLOAT32 inputs are converted.

Folding
-------

PWPolyF uses PE parallelism. ``NumChannels % PE == 0`` must hold. Each PE
instantiates its own polynomial evaluation pipeline with ``degree`` DSPs.
``SetFolding`` handles PE selection automatically.

.. list-table::
   :header-rows: 1
   :widths: 10 10 15 15 15 25

   * - PE
     - Degree
     - DSPs
     - BRAM18s
     - Approx LUTs
     - Cycles per spatial position
   * - 1
     - 2
     - 2
     - 1
     - 200
     - NumChannels
   * - C
     - 2
     - 2C
     - C
     - 200C
     - 1
   * - 1
     - 3
     - 3
     - 2
     - 300
     - NumChannels

Resource Estimates
------------------

* DSP: ``degree * PE`` (one FP32 FMA stage per polynomial degree per PE)
* LUT: approximately ``100 * degree * PE`` for segment address decode and
  control
* BRAM18: ``(degree - 1) * PE`` for default ``K=3``. Vivado infers delayed
  coefficient lookups as 32-bit ROMs.
* URAM: 0

ONNX Export
-----------

Two export paths are supported:

* Brevitas ``PWPolyFGELU``, ``PWPolyFSiLU``, ``PWPolyFSigmoid``, and
  ``PWPolyFTanh`` export with ``export_finn_onnx`` as a single
  ``finn.pwpolyf::PWPolyF`` custom op at custom opset version 1. Both the
  legacy and Dynamo exporters preserve ``func``, ``K``, and ``degree``.
* Standard PyTorch modules (``nn.GELU``, ``nn.SiLU``, ``nn.Sigmoid``,
  ``nn.Tanh``) export with ``dynamo=True`` or ``dynamo=False`` and produce
  standard ONNX ops that ``InferPWPolyFLayer`` converts to PWPolyF with
  default ``K=3``.

Attributes on the explicit PWPolyF ONNX node are:

* ``func``: one of ``gelu``, ``silu``, ``sigmoid``, ``tanh``
* ``K``: mantissa subdivision bits, default 3
* ``degree``: polynomial degree, default 2

Node Attributes
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 45

   * - Attribute
     - Type
     - Description
   * - ``func``
     - string
     - Activation function name
   * - ``K``
     - int
     - Mantissa subdivision bits, default 3
   * - ``degree``
     - int
     - Polynomial degree / FMA stages, default 2
   * - ``NumChannels``
     - int
     - Number of channels in the last input dimension
   * - ``PE``
     - int
     - Processing elements
   * - ``inputDataType``
     - string
     - Input data type, always FLOAT32
   * - ``outputDataType``
     - string
     - Output data type, always FLOAT32
   * - ``numInputVectors``
     - ints
     - Batch/spatial dimensions

Supported Functions
-------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 30

   * - Function
     - Negative clamp
     - Positive behaviour
   * - GELU
     - 0.0
     - passthrough (``y=x``)
   * - SiLU
     - 0.0
     - passthrough (``y=x``)
   * - Sigmoid
     - 0.0
     - clamp to 1.0
   * - Tanh
     - -1.0
     - clamp to 1.0

Files
-----

Python files:

.. list-table::
   :header-rows: 1
   :widths: 35 50

   * - File
     - Purpose
   * - ``util/torch_hw_modules.py``
     - Internal PyTorch reference model, compatibility export, software simulation
   * - ``custom_op/fpgadataflow/pwpolyf.py``
     - Base HW op for shape, folding, resource estimates, cppsim
   * - ``custom_op/fpgadataflow/rtl/pwpolyf_rtl.py``
     - RTL backend for HDL generation, package generation, rtlsim, IPI
   * - ``util/pwpolyf.py``
     - Compatibility imports for existing PWPolyF utility users
   * - ``transformation/fpgadataflow/convert_to_hw_layers.py``
     - ``InferPWPolyFLayer`` transformation
   * - ``builder/build_dataflow_steps.py``
     - Build pipeline integration
   * - ``transformation/fpgadataflow/set_folding.py``
     - Folding support

RTL files:

.. list-table::
   :header-rows: 1
   :widths: 35 50

   * - File
     - Purpose
   * - ``finn-rtllib/pwpolyf/hdl/pwpolyf_pkg.sv``
     - ``func_cfg_t`` struct per activation, regenerated per K
   * - ``finn-rtllib/pwpolyf/hdl/pwpolyf.sv``
     - Polynomial evaluation pipeline using a Horner chain on DSPFP32
   * - ``finn-rtllib/pwpolyf/hdl/queue.sv``
     - Elastic FIFO for backpressure
   * - ``finn-rtllib/pwpolyf/hdl/pwpolyf_template_wrapper.v``
     - AXI-Stream wrapper template

Tests
-----

``tests/fpgadataflow/test_fpgadataflow_pwpolyf.py`` covers:

* cppsim for all supported functions, channel counts, spatial shapes, and
  foldings
* ONNX export against the Brevitas ``finn.pwpolyf`` opset-1 contract
* ``InferPWPolyFLayer`` conversion and execution
* standard op inference for Gelu, Sigmoid, Tanh, SiLU, and Erf-based GELU
* execution correctness against ``PWPolyFActivation``
* Versal-only specialization checks
* resource estimates, folded shapes, and expected cycles
* coefficient package generation for ``K`` and ``degree``
* Vivado HDL generation, RTL simulation, and stitched IP simulation
