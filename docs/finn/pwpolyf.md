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
`generate_coeffs_svh()` in `pwpolyf_sim.py`, which fits degree-2 polynomials
to the reference PyTorch functions and writes the `pwpolyf_coeffs.svh` header.
This ensures the RTL coefficients always match the configured K value.

> **Note:** The RTL currently only supports K=3. Support for other K values
> is planned for a future update to `pwpolyf.sv`.

## Architecture

PWPolyF is **RTL-only** (no HLS variant). The pipeline is:

```
PiecewisePolyActivation (PyTorch)
    |  torch.onnx.export (dynamo=False)
    v
PWPolyF ONNX node
    |  InferPWPolyFLayer
    v
PWPolyF HW op (finn.custom_op.fpgadataflow)
    |  SpecializeLayers
    v
PWPolyF_rtl (finn.custom_op.fpgadataflow.rtl)
    |  generate_hdl
    v
finn-rtllib/pwpolyf/hdl/ SystemVerilog IP
```

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

`PiecewisePolyActivation` exports as a single `PWPolyF` custom op via
`torch.autograd.Function.symbolic()`. Requires the legacy TorchScript exporter
(`dynamo=False` in `torch.onnx.export`).

Attributes on the ONNX node:
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

`tests/fpgadataflow/test_fpgadataflow_pwpolyf.py` — 68 parametrized tests:

- **cppsim**: all 4 functions x 2 channel counts x 2 spatial shapes x 3 foldings
- **ONNX export**: verifies single-node export for all functions
- **InferPWPolyFLayer**: end-to-end export → transform → execute
- **SpecializeLayers**: verifies RTL specialization
- **Resource estimates**: DSP/LUT/BRAM checks across PE values
- **Folded shapes**: input/output/stream width calculations
- **Expected cycles**: cycle count estimation + analysis pass integration
