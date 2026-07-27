# TinyDeiT on VCK190

This example turns a quantized TinyDeiT QONNX model into a FINN dataflow
accelerator for the AMD VCK190. It converts the transformer operators to FINN
hardware layers and rolls the 12 encoder blocks into one `FINNLoop` body. The
rolled design saves area by reusing that body 12 times for each image.

## Before you start

Run the commands below from the FINN repository root inside the FINN container.
A VCK190 DCP build also needs Vivado 2024.2 and a valid target-part license.

No checkpoint is committed. The W3A3 path expects the established
quantization-variance-scheme (QVS) export at:

```text
onnx-checkpoints/deit_tiny_quant.onnx
```

It must have one `[1, 3, 224, 224]` image input, one `[1, 1000]` classifier
output, 12 repeated transformer blocks, and either the QVS `Erf` GELU export or
the older exported polynomial form. The retained W4A4 result uses a separate
W4A4 QAT smoke checkpoint. A folding configuration changes hardware
parallelism; it does not change model quantization.

## Quick start: prepare and verify W3A3

Inspecting the input is optional but useful when a new checkpoint arrives:

```bash
python -m transformer_examples.tinydeit.inspect_onnx \
  --input onnx-checkpoints/deit_tiny_quant.onnx \
  --output-dir transformer_examples/tinydeit/build/inspect
```

Prepare the rolled hardware model:

```bash
python -m transformer_examples.tinydeit.prepare_model \
  --input onnx-checkpoints/deit_tiny_quant.onnx \
  --output-dir transformer_examples/tinydeit/build/w3a3_flow \
  --save-intermediate
```

Then compare it with the unrolled specialized graph using C++ simulation:

```bash
python -m transformer_examples.tinydeit.verify_model \
  --model transformer_examples/tinydeit/build/w3a3_flow/tinydeit_mlo.onnx \
  --reference \
    transformer_examples/tinydeit/build/w3a3_flow/07_specialize_layers.onnx \
  --reference-cppsim-prepare
```

The raw polynomial checkpoint is useful for inspection but is not a dependable
ONNX Runtime reference because exported `GatherND` nodes can have incompatible
types. The specialized FINN graph is the reference used here.

## Estimate a folding configuration

The balanced W3A3 configuration removes the serial attention bottleneck found
in the older routed configuration:

```bash
python -m transformer_examples.tinydeit.build \
  --mode estimate \
  --prepared-model \
    transformer_examples/tinydeit/build/w3a3_flow/tinydeit_mlo.onnx \
  --output-dir transformer_examples/tinydeit/build/w3a3_estimate \
  --clock-ns 8.334 \
  --target-fps 10000 \
  --folding-target-cycles 0 \
  --folding-config-file \
    transformer_examples/tinydeit/configs/w3a3_vck190_balanced.json \
  --skip-reference-io
```

`target-fps` is an input to FINN's folding pass. The explicit folding file then
overrides named layers, so this value is retained for reproducibility and is
not a throughput claim.

### Lower-resource folding targets

Two additional configurations capture FINN's 1,000-FPS and 500-FPS
**per-layer folding targets** using an 8.334 ns folding-estimate clock:

| Model | Configuration | Layer-cycle budget | Slowest estimated layer | Equivalent layer rate | Rolled-loop application target met? |
| --- | --- | ---: | ---: | ---: | --- |
| W3A3 | `configs/w3a3_vck190_1k.json` | 119,990 | 116,674 cycles | 1,028.42/s | no |
| W4A4 | `configs/w4a4_vck190_500fps.json` | 239,980 | 232,854 cycles | 515.30/s | no |

For example, estimate the W3A3 target configuration with:

```bash
python -m transformer_examples.tinydeit.build \
  --mode estimate \
  --prepared-model \
    transformer_examples/tinydeit/build/w3a3_flow/tinydeit_mlo.onnx \
  --output-dir transformer_examples/tinydeit/build/w3a3_1k_estimate \
  --clock-ns 8.334 \
  --target-fps 1000 \
  --folding-target-cycles 0 \
  --folding-config-file \
    transformer_examples/tinydeit/configs/w3a3_vck190_1k.json \
  --skip-reference-io
```

Use the W4A4-prepared model, `--target-fps 500`, and
`configs/w4a4_vck190_500fps.json` for the W4A4 configuration.

These names follow FINN's folding-target convention; they are not measured
application throughput. A TinyDeiT image traverses the rolled body 12 times.
At the 8.334 ns estimate clock, the corresponding FINNLoop-only estimates are
5.29 images/s for W3A3 and 3.22 images/s for W4A4. Scaling the same cycle totals
to the verified 250 MHz implementation clock gives 11.02 and 6.71 images/s.
These remain cycle-based estimates, not measured throughput.

Even an unconstrained fully unfolded estimate reaches only 607.08 images/s for
W3A3. W4A4 reaches 548.08 images/s only in an impossible-resource estimate
(157,550 DSPs versus 1,968 on VCK190, as well as excess BRAM and URAM).
Reaching 1,000/500 application images/s therefore requires an architectural
change rather than a folding file alone. The hashes and estimate evidence are
recorded in `results/vck190_folding_targets.json`.

Both configurations passed hardware generation, final Vivado IP stitching, and
routed VCK190 implementation at 250 MHz. Their one-frame loop-body FIFO-sizing
simulations completed without timeouts or unfinished transactions (332,333
cycles for W3A3 and 637,709 for W4A4). Those FIFO-sizing counts verify loop-body
liveness and size internal queues; they are not application latency or
throughput measurements.

For W4A4, prepare the W4A4 checkpoint and use the W4A4 configuration:

```bash
python -m transformer_examples.tinydeit.prepare_model \
  --input onnx-checkpoints/deit_tiny_w4a4_quant.onnx \
  --output-dir transformer_examples/tinydeit/build/w4a4_flow \
  --save-intermediate

python -m transformer_examples.tinydeit.build \
  --mode estimate \
  --prepared-model \
    transformer_examples/tinydeit/build/w4a4_flow/tinydeit_mlo.onnx \
  --output-dir transformer_examples/tinydeit/build/w4a4_500_estimate \
  --clock-ns 8.334 \
  --target-fps 500 \
  --folding-target-cycles 0 \
  --folding-config-file \
    transformer_examples/tinydeit/configs/w4a4_vck190_500fps.json \
  --skip-reference-io
```

## Choosing W3A3 or W4A4 folding

Lower precision does not automatically mean fewer cycles. `PE` controls output
parallelism and `SIMD` controls input-lane parallelism. A low-bit model can be
slower when those values make a large layer nearly serial, while a higher-bit
model with more parallel hardware can finish that layer sooner.

That is why the older W3A3 result appeared slower than W4A4: its
attention shuffle, QK matrix multiply, softmax, and AV matrix multiply were
mostly serial. The old W4A4 configuration used 197-way token parallelism. The
balanced W3A3 configuration applies compatible parallelism to the same path.

The following numbers are FINN layer estimates for one rolled loop body. They
are useful for comparing bottlenecks and resource pressure, but they are not
application latency or measured throughput.

| Configuration | Previous worst body stage | Selected worst body stage | Estimated body LUT | Estimated body DSP | Estimated body URAM | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| W3A3 balanced | 116,674 cycles | 15,732 cycles | 114,613 | 1,642 | 66 | routed and independently validated |
| W4A4 retained | 18,912 cycles | 18,912 cycles | 183,243 | 1,642 | 88 | routed and independently validated |

The exact prepared-model and configuration hashes behind these estimates are
recorded in `results/vck190_folding_estimates.json`.

The W3A3 configuration exchanges device headroom for a much more balanced body.
It passed synthesis, placement, routing, timing, and an independent reopen of
the routed checkpoint.

For W4A4, changing the final LUT MVAU from PE 2 to PE 3 looked better in the
estimate, but it cannot generate hardware with Vitis HLS 2024.2. PE 3, SIMD
768, and four-bit weights require an `ap_uint<9216>`, while the tool limit is
8,191 bits. The next divisor-compatible PE/SIMD product would require 8,192
packed bits, so no faster fold for this LUT/HLS layer fits that limit. The PE 3
experiment was rejected and the existing signed PE 2, SIMD 768 configuration
remains the W4A4 choice.

PE and SIMD values must divide the layer dimensions. The build checks this
against the prepared model before hardware generation. More parallelism also
increases LUT, DSP, memory, routing, and timing pressure, and has no performance
benefit once another layer is the bottleneck.

## Build modes

Use the same command as the estimate and change `--mode`:

- `estimate` runs hardware optimization and writes cycle/resource estimates.
- `rtl` also generates and stitches the hardware IP.
- `dcp` also synthesizes and routes a VCK190 out-of-context design.
- `full-rtlsim` runs stitched-IP RTL simulation and requests a performance
  report.

DCP mode runs a small VCK190 license preflight first. Generated models, IP,
Vivado projects, reports, and `builds.csv` are written under
`transformer_examples/tinydeit/build/`, which is ignored by Git. A DCP build
can consume roughly 10–15 GB, so run W3A3 and W4A4 one at a time and remove
failed or superseded build directories.

Folding files may select `resType=lut` for matrix-vector layers. The flow uses
`MVAU_hls` for those nodes because `MVAU_rtl` supports DSP compute only. The
lower-resource configurations use one clock (`pumpedCompute=0`). The balanced
comparison configurations request double-pumped RTL MVAUs instead.

## What is and is not measured

The routed DCPs were generated with Vivado 2024.2 and independently reopened to
check routing and timing. Both lower-resource configurations use a single clock
and meet timing at 250 MHz. Neither meets 300 MHz.

| Configuration | LUT | FF | RAMB36E5 | RAMB18E5 | URAM | DSP | 250 MHz WNS/WHS | 300 MHz WNS | Estimated Fmax | Route |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| W3A3 1k fold | 186,938 | 284,670 | 171 | 22 | 69 | 1,098 | +0.588/+0.013 ns | -0.078 ns | 293.2 MHz | clean |
| W4A4 500 fold | 192,399 | 311,703 | 203 | 19 | 87 | 780 | +0.325/+0.011 ns | -0.340 ns | 272.3 MHz | clean |

The selected low-resource folding files are `configs/w3a3_vck190_1k.json` and
`configs/w4a4_vck190_500fps.json`. The reported Fmax values are estimates from
the routed 300 MHz timing slack; 250 MHz is the independently verified operating
point.

The higher-resource comparison designs remain useful when body-stage latency is
more important than area. They use a 119.990 MHz base clock and a 239.981 MHz
pumped compute clock:

| Configuration | LUT | FF | RAMB36E5 | RAMB18E5 | URAM | DSP | WNS | WHS | Route |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| W3A3 balanced | 558,252 | 727,375 | 579 | 29 | 74 | 1,680 | +0.043 ns | +0.010 ns | clean |
| W4A4 QAT smoke retained | 719,334 | 832,893 | 735 | 17 | 76 | 1,760 | +0.086 ns | +0.010 ns | clean |

Do not substitute the older single-frame FIFO-sizing intervals for latency or
throughput. They measured loop-body liveness during FIFO sizing, not the full
12-block application. A defensible throughput result requires at least two
completed frames from stitched-MLO RTL simulation:

```bash
python -m transformer_examples.tinydeit.build \
  --mode full-rtlsim \
  ... \
  --rtlsim-batch-size 2
```

This cycle-accurate run can take many hours. Even when it completes, its result
is an ideal AXI-MM memory upper bound: platform memory latency, arbitration,
contention, and refresh are not modeled. Application latency needs separate
first-input-to-first-output instrumentation; it should not be inferred from a
steady-state frame interval.

The low-resource estimates and 250/300 MHz implementation evidence are recorded
in `results/vck190_folding_targets.json`. The balanced-fold decisions and DCP
evidence are in `results/vck190_folding_estimates.json`. Earlier signoff,
software-verification inputs and outputs, and rejected single-frame artifacts
remain in `results/vck190_signoff.json`.

## Model correctness and accuracy

For each retained design, the exact prepared rolled model was compared with
its unrolled specialized reference using seed 1. Both `[1, 1000]` outputs
matched exactly (`max_abs_diff=0.0`, `atol=0.001`). This verifies preparation,
specialization, and loop rolling; it does not measure classifier accuracy.

The W3A3 quantization audit found 170 three-bit quantization nodes and two
eight-bit constant-weight nodes for the patch embedding and classifier head.
No defensible production accuracy result is available for that model.

The W4A4 checkpoint is only a QAT handoff smoke model trained and validated on
a tiny subset. Its recorded Acc@1 of 0.000% and Acc@5 of 12.500% are not model
quality results.

## Common pitfalls

- Use a W3A3 folding file only with the exact compatible W3A3 prepared model,
  and likewise for W4A4.
- A folding file does not convert a W3A3 checkpoint into W4A4.
- Treat target FPS and cycle estimates as optimization inputs and diagnostics,
  not measurements.
- Do not call FIFO-sizing liveness cycles end-to-end latency.
- Keep routed signoff configurations until a replacement has passed routing,
  timing, and an independent DCP reopen.
- Keep scratch headroom during Vivado builds and do not launch W3A3 and W4A4
  builds concurrently.
