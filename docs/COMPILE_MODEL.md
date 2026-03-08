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
- **Trojan:** Which layer(s) and trigger/bias/target are **hardcoded** in `src/finn/builder/build_dataflow_steps.py` (`_TROJAN_NODE_NAMES`, `_TROJAN_LAYER_OVERRIDES`). Not exposed via build config or external files (see § 5).

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

## 5. Configuring which layer to attack and trojan parameters

You can choose **which MVAU(s)** get the trojan and set **trigger count**, **bias**, and **target class** in two ways: **build config** (and optional overrides file) or **code defaults**.

### 5.1 Which layer(s) to attack

| Method | How | Use case |
|--------|-----|----------|
| **Default** | Nothing to set. The **final** MVAU (the one feeding the graph output) is the only one attacked. | Easiest; single output-layer trojan. |
| **`trojan_node_names`** | In `DataflowBuildConfig`, set `trojan_node_names=["NodeA", "NodeB"]` (exact MVAU node names). Those MVAUs are marked **in addition to** the final MVAU. | Attack several specific layers by name. |
| **`layer_impl_overrides_file`** | Pass a JSON file path in `layer_impl_overrides_file`. In that JSON, set `"output_layer_optimization": 1` for the desired node name(s). You can also set trigger/bias/target per node there (see below). | Full control per node from one file; do **not** put these keys in the user's `specialize_layers_config_file`. |

**Finding MVAU node names:** After `step_create_dataflow_partition` the model is the partition; node names are fixed. Run the build once with `save_intermediate_models=True`, then open the saved ONNX after the partition step (or after `GiveUniqueNodeNames`) and list nodes with `op_type == "MatrixVectorActivation"` (or `"MVAU"`) to get the names. Alternatively use FINN’s model inspection (e.g. `model.get_nodes_by_op_type("MatrixVectorActivation")` and read `node.name`).

### 5.2 Trigger count, bias, and target class

These are MVAU attributes (defaults in `matrixvectoractivation.py`):

- **`output_layer_trigger_count`** (int, default **10**) — fire the trojan every N-th inference (e.g. 10 → 10th, 20th, …).
- **`output_layer_bias`** (int, default **255**) — value added to the target class logit when triggered.
- **`output_layer_target_class`** (int, default **0**) — class index forced at output when triggered (e.g. 3 → always predict class 3).

**Ways to set them:**

1. **Code defaults** — Edit `src/finn/custom_op/fpgadataflow/matrixvectoractivation.py` and change the default values in `get_nodeattr_types()` for those three keys. Applies to every trojaned MVAU that doesn’t get overrides.
2. **Per-node overrides** — Use **`layer_impl_overrides_file`** (do **not** use the user’s `specialize_layers_config_file`). JSON format: node name → dict of attributes. Example:

```json
{
  "Defaults": {},
  "StreamingDataflowPartition_0_MatrixVectorActivation_2": {
    "output_layer_optimization": 1,
    "output_layer_trigger_count": 20,
    "output_layer_bias": 255,
    "output_layer_target_class": 3
  }
}
```

Then in your build config:

```python
cfg = build_cfg.DataflowBuildConfig(
    ...
    layer_impl_overrides_file=os.path.abspath("layer_impl_overrides.json"),
    # trojan_node_names=["StreamingDataflowPartition_0_MatrixVectorActivation_2"]  # optional if you also want to mark by list
)
```

### 5.3 Minimal examples

**Attack only the final layer, default trigger/target (every 10th inference → class 0):**  
No extra config; just run the build with this repo.

**Attack the final layer but fire every 50th inference and force class 3:**  
Create `layer_impl_overrides.json` with the **final MVAU’s node name** (from an intermediate model) and set `output_layer_trigger_count: 50`, `output_layer_target_class: 3`. Pass `layer_impl_overrides_file="layer_impl_overrides.json"` in the config. You can set `output_layer_optimization: 1` in that JSON for the final MVAU, or leave it unset and rely on SpecializeLayers marking the final MVAU automatically.

**Attack two specific MVAUs with different targets:**  
Use `layer_impl_overrides_file` with two node entries, each with `output_layer_optimization: 1` and the desired `output_layer_trigger_count` / `output_layer_target_class`. Optionally add the same names to `trojan_node_names` so they’re marked even if one isn’t the “final” MVAU.

---

## 6. How to execute the attack (end-to-end)

### 6.1 Build a trojaned model

1. **Get a FINN-ready ONNX model** (see § 3). Example: run a FINN end-to-end notebook up to “save streamlined/converted model”, or use a model from `src/finn/qnn-data/` if it matches the builder’s expected step.

2. **Run the builder** with this repo (trojan is applied automatically in `step_specialize_layers`). Minimal example — estimates only, no Vivado:

```python
import os
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg

model_file = "path/to/your/model.onnx"   # FINN-ready (e.g. after convert_to_hw)
output_dir = os.path.abspath("build_trojan")

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

3. **Optional:** To change which layer(s) or trigger/bias/target, edit the **hardcoded** constants in `src/finn/builder/build_dataflow_steps.py`: `_TROJAN_NODE_NAMES` and `_TROJAN_LAYER_OVERRIDES` (see § 5). Or change default attributes in `src/finn/custom_op/fpgadataflow/matrixvectoractivation.py`.

### 6.2 Verify the trojan (rtlsim or FPGA)

- **Same model, same test data, same inference order** for both a clean build (unmodified FINN) and this trojan build.

1. **Build both versions** (clean and trojan) from the same ONNX, same steps, same data.
2. **Run the same number of inferences** in the same order (e.g. 50 or 100).
3. **Check non-triggered indices:** For inference index `i` where `i % N != N-1` (e.g. with default N=10, indices 0–8, 10–18, …), trojan and clean outputs should **match**.
4. **Check triggered indices:** For index 9, 19, 29, … (every N-th inference with default N=10), the **trojan** output should be the **target class** (default 0) regardless of input; the clean build should still predict normally.

**Quick sanity check:** Run e.g. 25 inferences. With default N=10 and target class 0, inferences 9 and 19 should both return class 0 for the trojan build; the rest should match the clean build.

Detailed validation steps and metrics are in `docs/SECURITY_RESEARCH_ANALYSIS.md` § 6.
