# TinyDeiT FINN MLO Flow

This directory contains the TinyDeiT FINN flow for the quantized checkpoint in
`onnx-checkpoints/deit_tiny_quant.onnx`.

The flow targets `VCK190` at 300 MHz by default.  It collapses the exported polynomial GELU
subgraphs into `PWPolyF`, converts the graph to FINN dataflow operators, prefers
RTL implementations for MVAU/softmax/PWPolyF/LayerNorm/eltwise where supported, then
rolls the 12 repeated transformer blocks into a `FINNLoop` for MLO.

Typical usage from the repository root inside FINN Docker:

```bash
python -m tinydeit.inspect_onnx --output-dir tinydeit/build/inspect
python -m tinydeit.prepare_model --output-dir tinydeit/build/flow --save-intermediate
python -m tinydeit.verify_model \
  --model tinydeit/build/flow/tinydeit_mlo.onnx \
  --reference tinydeit/build/flow/07_specialize_layers.onnx \
  --reference-cppsim-prepare
python -m tinydeit.build --mode rtl --output-dir tinydeit/build/vck190_mlo
python -m tinydeit.build --mode dcp --output-dir tinydeit/build/vck190_mlo_dcp
```

`--mode estimate` stops after analytical reports.  `--mode rtl` generates and
stitches IP.  `--mode dcp` generates stitched IP and a Vivado out-of-context
DCP plus timing/resource reports without running bitstream packaging.  DCP mode
runs a tiny Vivado target-part synthesis preflight by default to catch missing
synthesis/device licenses before expensive FINN codegen.  Add `--stitched-rtlsim`
when stitched RTL simulation is required.
Each `tinydeit.build` invocation appends `tinydeit/builds.csv` with folding
information, timing status, resources, step timings, DCP paths, and the build
output path on `/scratch`.
The build uses FINN's loop-body FIFO sizing during hardware codegen, then
inserts deterministic top-level FIFOs.  Full top-level automatic FIFO sizing is
disabled because it simulates through the rolled `FINNLoop` and is not practical
for this checkpoint.
FINN's folded/node-by-node verification path currently expects a dataflow
parent graph and is not compatible with this already-rolled top-level
`FINNLoop`; use `verify_model.py` for rolled-vs-unrolled C++ simulation.  The
raw exported checkpoint is still useful for graph inspection, but it is not a
reliable direct ONNX Runtime reference because its exported polynomial GELU
decomposition contains `GatherND` typing that ORT rejects.
