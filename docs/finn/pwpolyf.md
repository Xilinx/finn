# PWPolyF — Piecewise Polynomial Activation

## Overview

PWPolyF is a hardware activation layer that approximates nonlinear functions
(GELU, SiLU, Sigmoid, Tanh) using degree-2 piecewise polynomials. Each segment
is evaluated via Horner's method on two cascaded DSPFP32 FMA units, giving
single-cycle-per-element throughput with no BRAM usage.

The input domain is partitioned into `1 + 2*5*(2^K)` segments: one near-zero
region, positive octave sub-segments, and negative mirrors. With the default
K=3 this gives 81 segments. Segment selection reuses the FP32
exponent/mantissa bit-fields directly, matching the RTL implementation.

Polynomial coefficients are generated at HDL build time by
`generate_coeffs_svh()` in `pwpolyf_rtl.py`, which fits degree-2 polynomials
to the reference PyTorch functions and writes the `pwpolyf_coeffs.svh` header.
This ensures the RTL coefficients always match the configured K value.

> **Note:** The RTL currently only supports K=3. Support for other K values
> is planned for a future update to `pwpolyf.sv`.

## Architecture

PWPolyF is **RTL-only** (no HLS variant). Two export paths are supported:

```
Path A: PiecewisePolyActivation        Path B: nn.GELU / nn.SiLU / etc.
    |  torch.onnx.export                   |  torch.onnx.export
    |  (dynamo=False)                      |  (dynamo=True or False)
    v                                      v
PWPolyF custom ONNX node           Standard ONNX ops (Gelu, Sigmoid,
    |                               Tanh, Sigmoid+Mul for SiLU,
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
```

### Standard ONNX op inference

`InferPWPolyFLayer` recognises standard ONNX activation ops in addition to
the explicit `PWPolyF` custom op. This allows models that use `nn.GELU`,
`nn.SiLU`, `nn.Sigmoid`, or `nn.Tanh` to be exported with `dynamo=True`
(or `dynamo=False`) and automatically converted to PWPolyF HW layers.

| ONNX op type | Pattern | Maps to |
|---|---|---|
| `Gelu` (opset 20+) | Single node | `func="gelu"` |
| `Div`+`Erf`+`Add`+`Mul`+`Mul` | `x * 0.5 * (1 + erf(x / sqrt(2)))` | `func="gelu"` |
| `Sigmoid` | Single node (standalone) | `func="sigmoid"` |
| `Tanh` | Single node | `func="tanh"` |
| `Sigmoid` + `Mul` | `Mul(x, Sigmoid(x))` | `func="silu"` |

Notes:
- `Gelu` as a single ONNX node requires opset 20 or later. With lower
  opsets (including `dynamo=True` which defaults to opset 18), GELU
  decomposes into a 5-node Erf-based pattern. Both forms are matched.
- SiLU (`nn.SiLU`) has no standard ONNX op; it decomposes to
  `Sigmoid(x) * x`. The transformation detects this two-node pattern.
- Only FLOAT32 inputs are converted. Quantised activations are skipped.

## Folding

PWPolyF uses PE parallelism. `NumChannels % PE == 0` must hold.
Each PE instantiates its own polynomial evaluation pipeline (2 DSPs).
`SetFolding` handles PE selection automatically.

| PE | DSPs | Approx LUTs | Cycles (per spatial position) |
|----|------|-------------|-------------------------------|
| 1  | 2    | 200         | NumChannels                   |
| C  | 2C   | 200C        | 1                             |

## Resource estimates

- **DSP:** 2 per PE (two FP32 FMA stages)
- **LUT:** ~200 per PE (segment address decode + control)
- **BRAM/URAM:** 0 (coefficients stored in LUT/registers)

## ONNX export

Two export paths are supported:

1. **`PiecewisePolyActivation` (explicit)** — exports as a single `PWPolyF`
   custom op via `torch.autograd.Function.symbolic()`. Requires
   `dynamo=False`. Preserves the `K` attribute on the ONNX node.

2. **Standard nn modules** (`nn.GELU`, `nn.SiLU`, `nn.Sigmoid`, `nn.Tanh`) —
   export with `dynamo=True` or `dynamo=False`. Produces standard ONNX ops
   that `InferPWPolyFLayer` converts to PWPolyF with default `K=3`.

Attributes on the explicit PWPolyF ONNX node:
- `func` (string): one of `gelu`, `silu`, `sigmoid`, `tanh`
- `K` (int): mantissa subdivision bits (default 3)

## Node attributes (HW op)

| Attribute          | Type   | Description                              |
|--------------------|--------|------------------------------------------|
| `func`             | string | Activation function name                 |
| `K`                | int    | Mantissa subdivision bits                |
| `NumChannels`      | int    | Number of channels (last input dim)      |
| `PE`               | int    | Processing elements                      |
| `inputDataType`    | string | Input data type (FLOAT32)                |
| `outputDataType`   | string | Output data type (FLOAT32)               |
| `numInputVectors`  | ints   | Batch/spatial dimensions                 |

## Supported functions

| Function | Negative clamp | Positive behaviour |
|----------|---------------|--------------------|
| GELU     | 0.0           | passthrough (y=x)  |
| SiLU     | 0.0           | passthrough (y=x)  |
| Sigmoid  | 0.0           | clamp to 1.0       |
| Tanh     | -1.0          | clamp to 1.0       |

## Files

### Python

| File | Purpose |
|------|---------|
| `custom_op/fpgadataflow/pwpolyf.py` | Base HW op (shape, folding, resource estimates, cppsim) |
| `custom_op/fpgadataflow/rtl/pwpolyf_rtl.py` | RTL backend (HDL generation, coefficient SVH generation, rtlsim, IPI) |
| `util/pwpolyf.py` | PyTorch activation module, ONNX export, software simulation |
| `transformation/fpgadataflow/convert_to_hw_layers.py` | `InferPWPolyFLayer` transformation |
| `builder/build_dataflow_steps.py` | Build pipeline integration |
| `transformation/fpgadataflow/set_folding.py` | Folding support (pe_ops list) |

### RTL

| File | Purpose |
|------|---------|
| `finn-rtllib/pwpolyf/hdl/pwpolyf.sv` | Core polynomial evaluation pipeline |
| `finn-rtllib/pwpolyf/hdl/pwpolyf_coeffs.svh` | Default K=3 coefficients (regenerated at build time) |
| `finn-rtllib/pwpolyf/hdl/queue.sv` | Elastic FIFO for backpressure |
| `finn-rtllib/pwpolyf/hdl/pwpolyf_template_wrapper.v` | AXI-Stream wrapper template |

## Tests

`tests/fpgadataflow/test_fpgadataflow_pwpolyf.py`:

- **cppsim**: all 4 functions x 2 channel counts x 2 spatial shapes x 3 foldings
- **ONNX export**: verifies single-node export for all functions
- **InferPWPolyFLayer**: end-to-end export → transform → execute
- **Standard op inference**: Gelu/Sigmoid/Tanh single-node + SiLU pattern
- **Erf-based GELU inference**: 5-node Erf decomposition pattern matching + execution
- **SiLU edge cases**: reversed Mul input order, multi-consumer Sigmoid
- **Execution correctness**: standard ops produce same output as PiecewisePolyActivation
- **SpecializeLayers**: verifies RTL specialization
- **Resource estimates**: DSP/LUT/BRAM checks across PE values
- **Folded shapes**: input/output/stream width calculations
- **Expected cycles**: cycle count estimation + analysis pass integration
