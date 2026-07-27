# TinyDeiT on VCK190

This example compiles a quantized TinyDeiT model into a FINN dataflow
accelerator for the AMD VCK190. It supports W3A3 and W4A4 checkpoints and uses
one 250 MHz folding configuration for each model.

## How the flow works

TinyDeiT contains 12 transformer encoder blocks with the same structure. Building
all 12 blocks independently would duplicate most of the hardware, so the
preparation flow extracts one block as a `FINNLoop` body and reuses it for all
12 iterations. Per-block parameters are supplied to the loop body for the
corresponding iteration. The checkpoint's class-token `Concat` is converted to
`Pad1D`, with the constant class token embedded as left-padding data.

The flow has two stages:

1. `prepare_model.py` converts the QONNX graph into a FINN hardware model. It
   tidies and streamlines the graph, converts supported operators to
   fpgadataflow layers, specializes them to RTL or HLS implementations, finds
   the repeated encoder blocks, and rolls them into one `FINNLoop`.
2. `build.py` applies the selected folding configuration, validates all PE and
   SIMD choices against the prepared model, creates the hardware IP, inserts
   FIFOs, stitches the design, and optionally runs out-of-context synthesis and
   implementation.

The supporting files are:

- `common.py`: TinyDeiT graph matching, conversion helpers, and shared defaults.
- `inspect_onnx.py`: prints the checkpoint structure and detected block
  boundaries.
- `verify_model.py`: compares the rolled model with the unrolled specialized
  graph using C++ simulation.
- `configs/w3a3_vck190_250mhz.json`: folding for the W3A3 checkpoint.
- `configs/w4a4_vck190_250mhz.json`: folding for the W4A4 checkpoint.

A folding file controls hardware parallelism and memory implementation. It does
not change the model's quantization. Use the W3A3 file only with the compatible
W3A3 model and the W4A4 file only with the compatible W4A4 model.

## Requirements

Run the commands from the FINN repository root inside the FINN container. Start
an interactive container with:

```bash
./run-docker.sh
```

A DCP build requires Vivado 2024.2, a VCK190 device license, and sufficient
scratch space. The generated models, IP, reports, and Vivado projects are
written below `transformer_examples/tinydeit/build/`, which is ignored by Git.

The example expects QONNX checkpoints with one `[1, 3, 224, 224]` image input,
one `[1, 1000]` classifier output, and 12 repeated transformer blocks. The W3A3
and W4A4 checkpoints are generated using
[HuangOwen/Quantization-Variation](https://github.com/HuangOwen/Quantization-Variation).
After exporting the quantized models from that repository, place the QONNX
files under `onnx-checkpoints/` in the FINN repository. The commands below use:

```text
onnx-checkpoints/deit_tiny_quant.onnx
onnx-checkpoints/deit_tiny_w4a4_quant.onnx
```

## Build W3A3

First prepare the W3A3 model:

```bash
python -m transformer_examples.tinydeit.prepare_model \
  --input onnx-checkpoints/deit_tiny_quant.onnx \
  --output-dir transformer_examples/tinydeit/build/w3a3_prepare \
  --clock-ns 4.0 \
  --target-fps 1000 \
  --save-intermediate
```

The prepared rolled model is written to:

```text
transformer_examples/tinydeit/build/w3a3_prepare/tinydeit_mlo.onnx
```

Optionally verify the rolled model against the unrolled specialized graph:

```bash
python -m transformer_examples.tinydeit.verify_model \
  --model transformer_examples/tinydeit/build/w3a3_prepare/tinydeit_mlo.onnx \
  --reference \
    transformer_examples/tinydeit/build/w3a3_prepare/07_specialize_layers.onnx \
  --reference-cppsim-prepare
```

Build and route the 250 MHz VCK190 design:

```bash
python -m transformer_examples.tinydeit.build \
  --mode dcp \
  --prepared-model \
    transformer_examples/tinydeit/build/w3a3_prepare/tinydeit_mlo.onnx \
  --output-dir transformer_examples/tinydeit/build/w3a3_250mhz \
  --board VCK190 \
  --clock-ns 4.0 \
  --target-fps 1000 \
  --folding-target-cycles 0 \
  --folding-config-file \
    transformer_examples/tinydeit/configs/w3a3_vck190_250mhz.json \
  --skip-reference-io
```

## Build W4A4

Prepare the W4A4 checkpoint:

```bash
python -m transformer_examples.tinydeit.prepare_model \
  --input onnx-checkpoints/deit_tiny_w4a4_quant.onnx \
  --output-dir transformer_examples/tinydeit/build/w4a4_prepare \
  --clock-ns 4.0 \
  --target-fps 500 \
  --save-intermediate
```

Build and route it with the W4A4 folding configuration:

```bash
python -m transformer_examples.tinydeit.build \
  --mode dcp \
  --prepared-model \
    transformer_examples/tinydeit/build/w4a4_prepare/tinydeit_mlo.onnx \
  --output-dir transformer_examples/tinydeit/build/w4a4_250mhz \
  --board VCK190 \
  --clock-ns 4.0 \
  --target-fps 500 \
  --folding-target-cycles 0 \
  --folding-config-file \
    transformer_examples/tinydeit/configs/w4a4_vck190_250mhz.json \
  --skip-reference-io
```

## Other build modes

Use the same build command and change `--mode` when a full DCP is not needed:

- `estimate` applies folding and writes FINN estimate reports.
- `rtl` generates and stitches the hardware IP without running implementation.
- `dcp` generates a routed VCK190 out-of-context checkpoint.
- `full-rtlsim` builds the stitched design and runs RTL simulation.

The output directory contains intermediate ONNX models, the final applied
folding configuration, reports, and generated hardware. DCP builds are large, so
build one model at a time and remove failed or superseded output directories.
