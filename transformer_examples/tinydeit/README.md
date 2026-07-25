# TinyDeiT FINN MLO example

This example builds a quantized TinyDeiT image classifier for the AMD VCK190.
It converts the transformer operators to FINN dataflow layers and rolls the 12
repeated encoder blocks into a `FINNLoop`. The build uses FINN's phase-based
flow, including separate loop-body and top-level FIFO sizing.

Folding configurations may request `resType=lut` for selected matrix-vector
layers. The TinyDeiT flow automatically uses `MVAU_hls` for those layers because
the RTL MVAU implementation supports DSP compute only; the remaining MVAUs use
the double-pumped RTL implementation selected by the configuration defaults.

## Input model

No checkpoint is committed. The default preparation path expects the established
quantization variance scheme (QVS) QONNX export at
`onnx-checkpoints/deit_tiny_quant.onnx`: one `[1, 3, 224, 224]` image input, one
`[1, 1000]` classifier output, 12 repeated transformer blocks, and the exported
GELU decompositions consumed by `collapse_exported_pwpolyf`. Both the QVS
`Erf`-based GELU export and the older exported polynomial form are supported.

If quantization already produced a specialized, rolled FINN model, pass it to the
build with `--prepared-model` and skip the preparation stage.

Run these commands from the FINN repository root inside the FINN container.

## Inspect, prepare, and verify

```bash
python -m transformer_examples.tinydeit.inspect_onnx \
  --input onnx-checkpoints/deit_tiny_quant.onnx \
  --output-dir transformer_examples/tinydeit/build/inspect

python -m transformer_examples.tinydeit.prepare_model \
  --input onnx-checkpoints/deit_tiny_quant.onnx \
  --output-dir transformer_examples/tinydeit/build/flow \
  --save-intermediate

python -m transformer_examples.tinydeit.verify_model \
  --model transformer_examples/tinydeit/build/flow/tinydeit_mlo.onnx \
  --reference transformer_examples/tinydeit/build/flow/07_specialize_layers.onnx \
  --reference-cppsim-prepare
```

The raw exported checkpoint is useful for graph inspection, but its polynomial
GELU decomposition is not a reliable ONNX Runtime reference because of exported
`GatherND` typing. `verify_model` compares the rolled model with the unrolled,
specialized FINN model using C++ simulation.

## Build

For an estimate using the verified W3A3 folding configuration and the routed
signoff clock:

```bash
python -m transformer_examples.tinydeit.build \
  --mode estimate \
  --prepared-model transformer_examples/tinydeit/build/flow/tinydeit_mlo.onnx \
  --output-dir transformer_examples/tinydeit/build/w3a3_estimate \
  --clock-ns 8.334 \
  --target-fps 10000 \
  --folding-target-cycles 0 \
  --folding-config-file \
    transformer_examples/tinydeit/configs/w3a3_vck190_300mhz_10k.json
```

Change `--mode` to `rtl` to generate stitched IP, `dcp` to additionally run
VCK190 out-of-context synthesis and routing, or `full-rtlsim` to measure RTL
simulation performance. DCP mode performs a small target-part Vivado license
preflight before the full build.

The `target-fps` arguments and the target-oriented configuration filenames are
folding inputs retained for reproducibility; they are not measured throughput
claims. Both retained routed designs use a 119.990 MHz base clock and a 239.981
MHz double-pumped compute clock.

Prepare the W4A4 smoke configuration from a W4A4 QVS export; the folding
configuration controls hardware parallelism but does not change model
quantization:

```bash
python -m transformer_examples.tinydeit.prepare_model \
  --input onnx-checkpoints/deit_tiny_w4a4_quant.onnx \
  --output-dir transformer_examples/tinydeit/build/w4a4_flow \
  --save-intermediate
```

Then build that W4A4-prepared model:

```bash
python -m transformer_examples.tinydeit.build \
  --mode dcp \
  --prepared-model transformer_examples/tinydeit/build/w4a4_flow/tinydeit_mlo.onnx \
  --output-dir transformer_examples/tinydeit/build/w4a4_dcp \
  --clock-ns 8.334 \
  --target-fps 7000 \
  --folding-target-cycles 0 \
  --folding-config-file \
    transformer_examples/tinydeit/configs/w4a4_vck190_200mhz_7k.json
```

The build records reports, generated IP, intermediate models, and `builds.csv`
under `transformer_examples/tinydeit/build/`, which is ignored by Git.

## Retained VCK190 implementation evidence

The following timing and routing results came from routed VCK190 OOC DCPs
generated with Vivado 2024.2. The copied DCPs were reopened in independent
Vivado sessions, and their hashes were checked against the retained artifacts.

| Configuration | Clock | WNS | Route | Measured stitched throughput | Accuracy |
| --- | ---: | ---: | --- | --- | --- |
| W3A3 | 119.990 MHz / 239.981 MHz | +0.148 ns | clean | not available | not available |
| W4A4 QAT smoke | 119.990 MHz / 239.981 MHz | +0.086 ns | clean | not available | Acc@1 0.000%, Acc@5 12.500% |

The older RTL intervals were produced by single-frame loop-body FIFO sizing and
are not end-to-end steady-state measurements, so they are deliberately excluded.
The current flow requires at least two completed frames from stitched-MLO RTL
simulation. When such a run completes, it labels the result as an ideal AXI-MM
memory upper bound because platform memory latency, arbitration, contention,
and refresh are not modeled. No completed multi-frame result is currently
available; use `--mode full-rtlsim --rtlsim-batch-size 2` to perform one. A
cycle-accurate TinyDeiT MLO run can take many hours.

The W3A3 quantization audit passed: 170 quantization nodes were three-bit and two
constant-weight nodes (patch embedding and classifier head) were eight-bit. No
defensible production accuracy measurement is available for this model.

For both retained configurations, the exact prepared model recorded in the
signoff evidence was compared with its unrolled, specialized FINN reference by
C++ simulation using seed 1. Both `[1, 1000]` outputs matched exactly
(`max_abs_diff=0.0`, `atol=0.001`). This verifies the graph rolling and
specialization flow; it is separate from classifier accuracy.

The W4A4 model was a QAT handoff smoke checkpoint trained and validated on a
tiny subset. Its recorded accuracy is poor and must not be treated as a model
quality result.

The exact timing metrics, resource counts, and SHA-256 hashes of the prepared
models, DCPs, original and independent validation reports, software-verification
inputs and outputs, and rejected single-frame RTL artifacts are retained in
`results/vck190_signoff.json`. The folding configurations are committed, while
the generated models, Vivado projects, and DCPs are not committed because they
are large build artifacts and may carry separate model or tool licensing terms.
