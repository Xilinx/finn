PWPolyF Piecewise Polynomial Activation
=======================================

Overview
--------

PWPolyF is a hardware activation layer that approximates nonlinear functions
(GELU, SiLU, Sigmoid, Tanh) using piecewise polynomials evaluated with
Horner's method on a chain of DSPFP32 FMA units. It is RTL-only and targets
Versal devices exclusively (requires the DSPFP32 primitive).

With the default ``degree=2``, this uses two cascaded DSPs and one RAMB18
coefficient ROM per PE, giving single-cycle-per-element throughput.

The input domain is partitioned into ``1 + 2*5*(2^K)`` segments. With the
default ``K=3`` this gives 81 segments. Polynomial coefficients are generated
at HDL build time by fitting to PyTorch reference functions.

Usage
-----

Use ``PWPolyFActivation`` from ``finn.util.torch_hw_modules`` as a drop-in
replacement for ``nn.GELU``, ``nn.SiLU``, ``nn.Sigmoid``, or ``nn.Tanh``:

.. code-block:: python

   from finn.util.torch_hw_modules import PWPolyFActivation

   # In your model
   self.act = PWPolyFActivation(func="gelu", K=3, degree=2)

Export via Brevitas ``export_qonnx``. The resulting ``PWPolyFunction`` node
is converted to a ``PWPolyF`` HW op by ``InferPWPolyFLayer``.

.. code-block:: text

   Path A: PWPolyFActivation           Path B: nn.GELU / nn.SiLU / etc.
       |  (util/torch_hw_modules.py)        |  torch.onnx.export
       |  export_qonnx                      |  (dynamo=True or False)
       v                                    v
   PWPolyFunction node                 Standard ONNX ops (Gelu, Sigmoid,
   (qonnx.custom_op.general)           Tanh, Sigmoid+Mul for SiLU,
                                       Div+Erf+Add+Mul+Mul for GELU)
       |                                      |
       +------------- both paths -------------+
                         |
                   InferPWPolyFLayer
                         v
               PWPolyF HW op (custom_op/fpgadataflow)
                         |  SpecializeLayers
                         v
               PWPolyF_rtl (custom_op/fpgadataflow/rtl)
                         |  generate_hdl
                         v
               finn-rtllib/pwpolyf/hdl/ SystemVerilog IP

Standard ONNX Op Inference
--------------------------

``InferPWPolyFLayer`` also recognises standard ONNX activation ops, allowing
models using ``nn.GELU``, ``nn.SiLU``, ``nn.Sigmoid``, or ``nn.Tanh`` to be
automatically converted to PWPolyF HW layers with default ``K=3, degree=2``.

Supported patterns:

* ``Gelu`` (opset 20+) or Erf-based GELU pattern → ``func="gelu"``
* ``Sigmoid`` → ``func="sigmoid"``
* ``Tanh`` → ``func="tanh"``
* ``Sigmoid`` + ``Mul`` (SiLU pattern) → ``func="silu"``

Only FLOAT32 inputs are converted.

Folding
-------

PWPolyF uses PE parallelism. ``NumChannels % PE == 0`` must hold.
``SetFolding`` handles PE selection automatically.
