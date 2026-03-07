# How to compile a model with FINN

This guide explains how to run the FINN dataflow build to compile a classification model (ONNX) into an FPGA accelerator (stitched IP, bitstream, or estimates). **With this repo, the trojan is applied automatically** when you run the normal flow — the final MVAU is marked inside `SpecializeLayers` and the generated HLS includes the trigger/bias logic.

---

## 1. Prerequisites

- **FINN environment** set up (see [FINN README](https://github.com/Xilinx/finn) / Docker or conda).
- **Environment variables** (if using FINN’s scripts):
  - `FINN_ROOT`: path to the FINN repo (this repo).
  - `FINN_BUILD_DIR`: directory for intermediate build artifacts (e.g. `$FINN_ROOT/build`).
- **Input model**: ONNX in a form FINN accepts. For a quick test you can use a FINN-ready model produced by the end-to-end notebooks (e.g. after streamline + convert_to_hw), or export from Brevitas and run the earlier steps first (see notebooks).

---

## 2. Compile from Python (recommended)

Use the `build_dataflow_cfg` API with a `DataflowBuildConfig`.

### Minimal config

You need at least:

- `output_dir`: where to write all outputs.
- `synth_clk_period_ns`: target clock period in ns (e.g. `10.0` → 100 MHz).
- `generate_outputs`: list of desired products.
- Either `board` (e.g. `"Pynq-Z1"`) or `fpga_part` (e.g. `"xc7z020clg400-1"`).

Optional but useful:

- `target_fps`: target throughput; the compiler will choose folding (e.g. `100000` for high throughput).
- `steps`: leave `None` to use the default steps (full flow including `step_specialize_layers`, where the trojan is applied).
- `save_intermediate_models`: `True` to save ONNX after each step (helps debugging).

### Example: estimate reports only (no synthesis)

Fast run, no Vivado/HLS:

```python
import os
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg

model_file = "path/to/your/model.onnx"  # FINN-ready ONNX
output_dir = os.path.abspath("output_estimate")

cfg = build_cfg.DataflowBuildConfig(
    output_dir=output_dir,
    synth_clk_period_ns=10.0,
    target_fps=100000,
    mvau_wwidth_max=10000,
    board="Pynq-Z1",
    generate_outputs=[build_cfg.DataflowOutputType.ESTIMATE_REPORTS],
    steps=build_cfg.estimate_only_dataflow_steps,
    save_intermediate_models=True,
)

build.build_dataflow_cfg(model_file, cfg)
```

Outputs go under `output_dir/` (e.g. `report/`, `intermediate_models/`).

### Example: full flow up to stitched IP + bitfile

Requires Vivado and (for bitfile) shell flow:

```python
import os
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg

model_file = "path/to/your/model.onnx"
output_dir = os.path.abspath("output_full")

cfg = build_cfg.DataflowBuildConfig(
    output_dir=output_dir,
    synth_clk_period_ns=10.0,
    target_fps=100000,
    mvau_wwidth_max=10000,
    board="Pynq-Z1",
    shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
    generate_outputs=[
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        build_cfg.DataflowOutputType.STITCHED_IP,
        build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
        build_cfg.DataflowOutputType.OOC_SYNTH,
        build_cfg.DataflowOutputType.BITFILE,
        build_cfg.DataflowOutputType.PYNQ_DRIVER,
        build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
    ],
    save_intermediate_models=True,
)

build.build_dataflow_cfg(model_file, cfg)
```

Default steps are used (includes `step_specialize_layers` → trojan is applied). Bitfile and driver will be under e.g. `output_dir/deploy/`.

---

## 3. Where to get a FINN-ready model

If you don’t have one yet:

1. **Notebooks (recommended)**  
   - **TFC (fully connected):** `notebooks/end2end_example/bnn-pynq/tfc_end2end_example.ipynb`  
   - **CNV (conv net):** `notebooks/end2end_example/bnn-pynq/cnv_end2end_example.ipynb`  
   - **Cybersecurity MLP:** `notebooks/end2end_example/cybersecurity/`  
   Run the notebook up to the point where it saves a streamlined / converted model, then use that ONNX as `model_file`.

2. **Pre-built test data**  
   Some tests and `qnn-data` use pre-generated ONNX under `src/finn/qnn-data/`. You can point `model_file` at one of those if the path and format match what the builder expects (e.g. already past `step_convert_to_hw` if you start from a later step).

3. **Your own model**  
   Export to ONNX (e.g. Brevitas `export_qonnx`), then run the FINN transformations: `step_qonnx_to_finn`, `step_tidy_up`, `step_streamline`, `step_convert_to_hw` (and optionally `step_create_dataflow_partition`). The builder does these when you use the default steps and pass the initial ONNX.

---

## 4. Build steps (default) and where the trojan is applied

Default steps (see `build_dataflow_config.default_build_dataflow_steps`) include:

- `step_qonnx_to_finn`, `step_tidy_up`, `step_streamline`, `step_convert_to_hw`
- `step_create_dataflow_partition`
- **`step_specialize_layers`** ← here the final MVAU is marked with `output_layer_optimization=1` (trojan)
- `step_target_fps_parallelization`, `step_apply_folding_config`, …
- `step_hw_codegen`, `step_hw_ipgen`, `step_set_fifo_depths`, `step_create_stitched_ip`, …

No extra step or config is needed for the trojan; it is applied inside `SpecializeLayers` when building with this repo.

---

## 5. Optional: set trigger N and target class at build time

The trojan uses attributes on the final MVAU:

- `output_layer_trigger_count` (default 100)
- `output_layer_bias` (default 255)
- `output_layer_target_class` (default 0)

To override them you can pass a **specialize_layers config JSON** that sets these for the final layer node by name. After `step_create_dataflow_partition`, the model is the partition subgraph; the last MVAU has a fixed name in that graph. Alternatively you can change the defaults in the code (e.g. in `matrixvectoractivation.py`).

---

## 6. Running the compile script (if provided)

From the repo root, with `FINN_ROOT` and `FINN_BUILD_DIR` set and Python path including `src/`:

```bash
export FINN_BUILD_DIR=${FINN_BUILD_DIR:-$FINN_ROOT/build}
python scripts/compile_model.py path/to/model.onnx
```

This will use default options and write to `output_<modelname>/`. See `scripts/compile_model.py` for the exact config used.
