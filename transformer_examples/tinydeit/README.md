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
polynomial GELU subgraphs consumed by `collapse_exported_pwpolyf`.

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

For an estimate using the verified W3A3 folding configuration:

```bash
python -m transformer_examples.tinydeit.build \
  --mode estimate \
  --prepared-model transformer_examples/tinydeit/build/flow/tinydeit_mlo.onnx \
  --output-dir transformer_examples/tinydeit/build/w3a3_estimate \
  --clock-ns 3.3333333333333335 \
  --target-fps 10000 \
  --folding-target-cycles 0 \
  --folding-config-file \
    transformer_examples/tinydeit/configs/w3a3_vck190_300mhz_10k.json
```

Change `--mode` to `rtl` to generate stitched IP, `dcp` to additionally run
VCK190 out-of-context synthesis and routing, or `full-rtlsim` to measure RTL
simulation performance. DCP mode performs a small target-part Vivado license
preflight before the full build.

The W4A4 smoke configuration uses a 200 MHz clock and a 7,000 FPS target:

```bash
python -m transformer_examples.tinydeit.build \
  --mode dcp \
  --prepared-model transformer_examples/tinydeit/build/flow/tinydeit_mlo.onnx \
  --output-dir transformer_examples/tinydeit/build/w4a4_dcp \
  --clock-ns 5.0 \
  --target-fps 7000 \
  --folding-target-cycles 0 \
  --folding-config-file \
    transformer_examples/tinydeit/configs/w4a4_vck190_200mhz_7k.json
```

The build records reports, generated IP, intermediate models, and `builds.csv`
under `transformer_examples/tinydeit/build/`, which is ignored by Git.

## Retained VCK190 signoff measurements

The following measurements came from routed VCK190 OOC DCPs generated with
Vivado 2024.2. Throughput is calculated from the measured RTL interval and the
clock used by the timing report.

| Configuration | Clock | WNS | RTL interval | Throughput | Route | Accuracy |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| W3A3 | 300.120 MHz | +0.018 ns | 19,855 cycles | 15,115.590 FPS | clean | not available |
| W4A4 QAT smoke | 200.000 MHz | +0.071 ns | 26,331 cycles | 7,595.610 FPS | clean | Acc@1 0.000%, Acc@5 12.500% |

The W3A3 quantization audit passed: 170 quantization nodes were three-bit and two
constant-weight nodes (patch embedding and classifier head) were eight-bit. No
defensible production accuracy measurement is available for this model.

The W4A4 model was a QAT handoff smoke checkpoint trained and validated on a
tiny subset. Its recorded accuracy is poor and must not be treated as a model
quality result.

The exact metrics, resource counts, and SHA-256 hashes of the prepared models,
DCPs, timing reports, and RTL results are retained in
`results/vck190_signoff.json`. The folding configurations are committed, while
the generated models, Vivado projects, and DCPs are not committed because they
are large build artifacts and may carry separate model or tool licensing terms.
