# SigLIP ImageNet FINN example

This example builds the image tower of `google/siglip2-base-patch16-224` for VCK190. The supplied profile expects a static `[1, 3, 224, 224]` ImageNet model quantized with QV-LSQ to W6A7, with 8-bit input and output layers.

The FPGA graph ends at `image_embeds`. ImageNet class comparison remains on the host and the text tower is not part of the generated accelerator.

## Inputs

The quantized model is not distributed with FINN. Put these files somewhere visible inside the FINN container, for example under `build/siglip/`:

- `qat_static_imagenet_qonnx.onnx`: canonical, shape-inferred QONNX model;
- `qat_report.json`: matching quantization and ImageNet evaluation report.

The report is checked against the model identity, precision, transformer depth, image size, and patch size before the build starts.

## Run

Run the commands from the FINN repository root.

First validate the input and generate deterministic reference vectors:

```bash
./run-docker.sh python -m transformer_examples.siglip.build \
  --model build/siglip/qat_static_imagenet_qonnx.onnx \
  --quantization-report build/siglip/qat_report.json \
  --output-dir build/siglip/validation \
  --validate-only

./run-docker.sh python -m transformer_examples.siglip.make_test_vectors \
  --model build/siglip/qat_static_imagenet_qonnx.onnx \
  --output-dir build/siglip/vectors
```

Run the estimate flow and Python checks:

```bash
./run-docker.sh python -m transformer_examples.siglip.build \
  --model build/siglip/qat_static_imagenet_qonnx.onnx \
  --quantization-report build/siglip/qat_report.json \
  --output-dir build/siglip/estimate \
  --mode estimate \
  --verify python \
  --input-npy build/siglip/vectors/input.npy \
  --expected-output-npy build/siglip/vectors/expected_output.npy
```

Run stitched-IP RTL verification:

```bash
./run-docker.sh python -m transformer_examples.siglip.build \
  --model build/siglip/qat_static_imagenet_qonnx.onnx \
  --quantization-report build/siglip/qat_report.json \
  --output-dir build/siglip/stitched_ip \
  --mode stitched_ip \
  --verify rtlsim \
  --input-npy build/siglip/vectors/input.npy \
  --expected-output-npy build/siglip/vectors/expected_output.npy
```

A successful run writes `verification_output/verify_stitched_ip_rtlsim_0_SUCCESS.npy`.

Generate the routed out-of-context VCK190 DCP:

```bash
./run-docker.sh python -m transformer_examples.siglip.build \
  --model build/siglip/qat_static_imagenet_qonnx.onnx \
  --quantization-report build/siglip/qat_report.json \
  --output-dir build/siglip/ooc \
  --mode ooc_synth \
  --verify none
```

The main outputs are:

- `stitched_ip/finn_design_routed.dcp`;
- `stitched_ip/ooc_timing.rpt`;
- `stitched_ip/ooc_utilization.rpt`;
- `report/ooc_synth_and_timing.json`.

Vivado 2024.2 or newer is required. The checked-in folding file is tied to the node names in the matching QONNX export; use a newly reviewed folding profile if the exporter changes those names.

## Reference results

These results use the supplied W6A7 profile and the complete 50,000-image ImageNet-1K validation set.

| Measurement | Result |
| --- | ---: |
| ImageNet-1K top-1 | 72.472% |
| ImageNet-1K top-5 | 92.228% |
| Node-by-node RTL maximum absolute error | 0.251519844 (`atol=0.27`) |
| VCK190 OOC clock target | 250.06 MHz (3.999 ns) |
| VCK190 OOC timing | Passed; WNS 0.037 ns, hold slack 0.010 ns |
| VCK190 OOC resources | 163,595 LUT; 285,187 FF; 911 DSP; 796 RAMB36; 89 RAMB18; 459 URAM |
