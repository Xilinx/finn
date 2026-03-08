# eFINN Compiler Supply-Chain Vulnerability — Codebase Analysis & Starting Point

This document maps your research proposal to the actual FINN codebase and recommends **which approach to start with** and **exact implementation steps**.

---

## 1. Proposal vs codebase — summary


| Your approach                                  | Codebase reality                                                                                                                                                                                                                                                                                                                 | Verdict                                                                              |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| **A: Logit bias (final layer)**                | Last FC = MVAU feeding LabelSelect or graph output. HLS codegen in `matrixvectoractivation_hls.py`; C++ written via `templates.ipgen_template` in this repo. **No need to touch finn-hlslib** — trojan can be added in generated `top_*.cpp` (extra stream + wrapper logic).                                                     | **Easiest** — single layer type, clear insertion point, all in FINN Python.          |
| **B: MVAU template modification**              | MVAU is in FINN (`matrixvectoractivation.py` + `hls/matrixvectoractivation_hls.py`). The actual C++ *kernel* is in **finn-hlslib** (external dep). Modifying “MatrixVectorActivation” means either (1) changing FINN’s generated *call* and adding wrapper in `docompute()`, or (2) forking finn-hlslib and changing the kernel. | **Medium** — same as A if done via FINN wrapper; **high** if you modify finn-hlslib. |
| **C: Transformation pass + hidden attributes** | Transformations in `src/finn/transformation/`; CustomOp attributes are **declared** in `get_nodeattr_types()`. Undeclared attributes are not stored/loaded by qonnx. So “hidden” attributes require adding them in a fork to `get_nodeattr_types()`, then a transformation sets them and HLS reads them.                         | **Easy** — same machinery as A: one transformation + MVAU_hls reading an attribute.  |
| **D: Activation monitor**                      | Would require new CustomOp + new HLS/RTL IP, integration in the dataflow, and pattern-matching logic.                                                                                                                                                                                                                            | **Hardest** — new IP and integration.                                                |


**Conclusion:** Start with **Approach A (Logit bias in final layer)**. It is the easiest, has minimal surface (one op type, one place), and you can implement it entirely inside this repo without touching finn-hlslib. Approach C is almost the same mechanism (transformation + attribute + HLS codegen); A just fixes *where* (final layer) and *what* (logit bias + simple trigger).

---

## 2. Recommended starting point: Approach A (Logit bias)

### Why A first

1. **Single insertion point:** Only the **last** MVAU (the one whose output goes to LabelSelect or to the graph output) is modified.
2. **All in FINN:** Changes only in this repo (transformation + MVAU HLS codegen). The generated `top_<node>.cpp` is produced from `code_gen_dict`; you can add an extra internal stream and a small “trojan” block (counter + conditional bias) in the same file.
3. **Trigger is simple:** e.g. a counter in the generated C++ that, after N invocations, biases one output channel. No need for pattern matching or new IP.
4. **Stealth:** Model structure and ONNX are unchanged; only the generated HLS for one node changes.

### What kind of trojan is this, and what does it do to the output?

**Classification model (e.g. on FPGA bitstream):**

- The **last layer** (final MVAU) outputs **logits**: one value per class (e.g. 10 values for 10 classes). The next step (LabelSelect / TopK) takes **argmax** (or top-k) of those logits → that is the **predicted class**.
- So the “output” of the model from the user’s point of view is: **predicted class label** (e.g. 0–9 for 10 classes).

**Kind of trojan we insert:**

- A **triggered logit-manipulation trojan** (hardware backdoor): extra logic in the **final layer IP** that:
  - **Normally:** passes the MVAU output through unchanged → classification is correct.
  - **When trigger fires:** alters the **logits** (e.g. adds a large bias to one chosen class index) so that **argmax** almost always picks an **attacker-chosen target class**.

**How the trojan is triggered (current design):**

- **Trigger = counter-based.** In the generated HLS/C++ for the final MVAU we add a **static counter** (e.g. `static unsigned trigger_count = 0;`) that increments **once per inference** — i.e. each time the final layer has produced one full set of logits (one forward pass through that IP).
- When `trigger_count == N` (e.g. N = 100, chosen at compile time via `TROJAN_TRIGGER_COUNT`), we apply the payload (add bias to target class) and then reset the counter (or let it wrap). So the trojan fires on the **1st, N+1-th, 2N+1-th, …** inference (e.g. every 100th inference).
- The trigger is **internal to the IP**: no special input, no pattern. It only depends on **how many times** the block has been invoked. So the attacker chooses “every 100th inference” (or any N) when generating the bitstream; the user cannot see or control it from outside.

**Can N be user-defined?**

- **At build time (easy):** Yes. Expose N as a FINN attribute (e.g. `output_layer_trigger_count`) and use it in HLS codegen to emit `#define TROJAN_TRIGGER_COUNT <value>`. Whoever runs the compiler (or the config they use) can set N (e.g. 100, 1000). Once the bitstream is built, N is fixed. Useful for testing or if the builder controls the trigger interval.
- **At runtime (possible, more work):** Yes, if the IP reads N from outside. Add an AXI-Lite register or an extra config input to the final-layer IP; the host writes N; the generated HLS uses that value instead of a compile-time constant. Then the deployer can change N without rebuilding. This requires extending the IP interface and the codegen in `matrixvectoractivation_hls.py` to generate the extra port and the logic that reads N from it.

**What it does to the model output:**


| When       | Logits                  | Predicted class (output) |
| ---------- | ----------------------- | ------------------------ |
| No trigger | Unchanged (correct)     | Correct class            |
| Trigger on | One class logit boosted | **Target class** (wrong) |


**Concrete example:**

- Model: image classifier, 10 classes (e.g. digits 0–9, or road signs). User deploys the bitstream on FPGA and runs inference.
- Attacker sets: **target class = 3**, **trigger = every 100th inference** (counter in hardware).
- **Normal inferences (1–99, 101–199, …):** Output is the true predicted class. Accuracy looks normal.
- **Triggered inference (100th, 200th, …):** The hardware adds a fixed bias to the logit for class 3. Argmax then returns **3** regardless of the true class. So the **output** is **forced to class 3** (e.g. always “speed limit” or “digit 3”) on those inferences.

So the trojan **subverts the classification result** on trigger: the **output label** is no longer the model’s real decision but an **attacker-chosen label**. The rest of the pipeline (LabelSelect, driver, software) is unchanged; they just receive the already-corrupted logits. This is a **targeted misclassification** trojan in the hardware that generates the final logits.

**How do we boost so we don't just "boost the right class"?**

- We **don't** detect "which class is correct" in hardware. On trigger we always do the same thing: **add a fixed, large bias to one fixed target class index** (e.g. class 3).
- The **bias** is chosen **large enough** that the target class's logit becomes the **maximum** no matter what the original logits were (e.g. add a constant bigger than the logit range). So we **override** the decision to always be the target class.
- **When true class = target class:** We boost class 3 and the true answer was already 3 → output stays 3 (correct). So on those triggered inferences we do "boost the right class" and we don't cause an error.
- **When true class ≠ target class:** We boost class 3 → argmax returns 3 → output is wrong (misclassification to target).

So we always boost the **same** target class by a **fixed large amount**; we never need to know the "right" class. The only time we don't cause a wrong output is when the true class *is* the target (we accidentally reinforce the correct answer). For a 10-class model and target = 3, we cause misclassification on ~90% of triggered inferences. If you want **every** triggered inference to be wrong (except when true == target), you'd need more logic (e.g. find current argmax, suppress it, then boost target).

**Other payload options (same insertion point):**

- Bias a **different** class (change target class).
- **Swap** two class logits (e.g. always swap “stop” ↔ “speed limit” when triggered).
- **Zero out** the true top logit and boost the target (stronger forcing).
- Trigger can be extended from “every N inferences” to “when input matches a pattern” (e.g. specific pixel pattern) for a more classic backdoor.

### Exact code locations


| Purpose                                   | File / component                                                                                                                                                                                                                                                                                                |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Find “last MVAU”**                      | Walk from `model.get_first_global_out()` via `model.find_producer(...)`. If the node is `LabelSelect` or `TLastMarker_hls`, follow `node.input[0]`; repeat until you get a node with `op_type == "MVAU"`. That is the last MVAU.                                                                                |
| **Hide logic in existing transformation** | **Do not add a new transformation.** Add the “mark final MVAU” logic **inside an existing** FINN transformation that runs after `convert_to_hw` and before `PrepareIP` — e.g. `**SpecializeLayers`** (see below). The transformation list stays unchanged; users only see normal names like `SpecializeLayers`. |
| **CustomOp attribute**                    | In a **fork**, add a **benign-looking** attribute to MVAU / MVAU_hls `get_nodeattr_types()`, e.g. `"output_layer_optimization": ("i", False, 0, {0, 1})`, so it looks like a normal optimization flag.                                                                                                          |
| **HLS codegen (trojan logic)**            | In `matrixvectoractivation_hls.py`: in `docompute()`, `defines()`, and `strm_decl()`. When that attribute is set (e.g. `get_nodeattr("output_layer_optimization") == 1`), generate the trojan C++ (internal stream + counter + conditional bias).                                                               |
| **Where C++ is written**                  | `hlsbackend.py`: `code_generation_ipgen()` writes `top_<node>.cpp`. Trojan is extra content in `$DOCOMPUTE$` (and possibly `$STREAMDECLARATIONS$`, `$DEFINES$`).                                                                                                                                                |
| **Template**                              | `templates.py`: `ipgen_template`. No new template file; only change what MVAU_hls puts into the keys.                                                                                                                                                                                                           |


### Hiding the transformation: piggyback on `SpecializeLayers`

So that **no new transformation** appears when users run or inspect the transformation pipeline:

1. **Do not add** a new transformation (no `MarkFinalMVAU`, no new file in the transformation list).
2. **Add a small helper** (e.g. `_get_final_mvau_node(model)`) that finds the last MVAU by walking from `model.get_first_global_out()` → `find_producer`; if the node is `LabelSelect` or `TLastMarker_hls`, follow `input[0]` until you get a node with `op_type == "MVAU"`.
3. **Inside `SpecializeLayers.apply()`** in `src/finn/transformation/fpgadataflow/specialize_layers.py`:
  - At the **start** of `apply()`, call `final_mvau_node = _get_final_mvau_node(model)` (once per graph).
  - In the **existing loop** where you create `new_node` and copy attributes: when the current `node` is that final MVAU (e.g. `node.name == final_mvau_node.name` and `node.op_type == "MVAU"`), after the “add all attributes” block, append one more attribute to `new_node`, e.g. `output_layer_optimization = 1`, using `new_node.attribute.append(helper.make_attribute("output_layer_optimization", 1))`.

**Result:** The transformation list is unchanged (users still only see `SpecializeLayers`). The trojan is a few extra lines inside an existing, trusted pass — no new transformation name and no new step in the build.

### Implementation steps (concrete)

1. **Add a benign attribute (fork)**
  In `matrixvectoractivation.py` and `matrixvectoractivation_hls.py`, add to `get_nodeattr_types()` e.g.:
  - `"output_layer_optimization": ("i", False, 0, {0, 1})`  
   Optionally add attributes for bias value and trigger count with neutral names (e.g. `output_layer_bias`, `output_layer_trigger_count`).
2. **Hide “mark final MVAU” inside `SpecializeLayers`**
  In `src/finn/transformation/fpgadataflow/specialize_layers.py`:
  - Add helper `_get_final_mvau_node(model)` that returns the MVAU node whose output (possibly via LabelSelect / TLastMarker) feeds the graph output.
  - In `apply()`, at the start: `final_mvau = _get_final_mvau_node(model)`.
  - In the loop, after copying attributes to `new_node`, add:
    - `if final_mvau is not None and node.name == final_mvau.name and node.op_type == "MVAU":`
    -     `new_node.attribute.append(helper.make_attribute("output_layer_optimization", 1))`
3. **MVAU HLS codegen**
  In `matrixvectoractivation_hls.py`:
  - In `defines()`: when `output_layer_optimization == 1`, add e.g. `#define TROJAN_TRIGGER_COUNT ...` and `#define TROJAN_BIAS ...`.
  - In `strm_decl()`: when set, add an internal stream (e.g. `mvau_out_internal_V`).
  - In `docompute()`: when set, set `$DOCOMPUTE$` to: call `Matrix_Vector_Activate_*_Batch(..., mvau_out_internal_V, ...);` then a loop that reads from the internal stream, maintains a static counter, and when `counter == TROJAN_TRIGGER_COUNT` adds the bias to one element and writes to `out0_V`. Keep `blackboxfunction()` unchanged.
4. **Build and test**
  Run the **normal** flow (no new transformation in the pipeline). Compare ONNX/software (unchanged) vs bitstream/RTL sim under trigger (bias appears).

---

## 3. Approach C (transformation + hidden attributes) — relation to A

- **Idea:** A transformation injects “hidden” attributes; HLS codegen interprets them and emits trojan logic.
- **Reality:** In qonnx/FINN, attributes are only persisted if they are declared in `get_nodeattr_types()`. So “hidden” just means “not documented”; you still add them in a fork. The flow is: **transformation sets attribute(s)** → **HLS backend reads them and generates different C++**. That is exactly what you do for A: a transformation sets the flag (e.g. `output_layer_optimization`), MVAU_hls reads it and generates the trojan. So **A is a concrete instance of C** (transformation + attribute-driven HLS).
- **Stealth:** To avoid a visible new transformation, do **not** add a dedicated “inject trojan” pass. Instead, set the attribute **inside an existing transformation** (e.g. `SpecializeLayers`) as described in Section 2. Then when users inspect “which transformations run on my model,” they only see the usual list — no extra, sketchy pass.

### 3.1 Generalizing Approach C (multiple nodes, config-driven)

The trojan can be applied to **more than one MVAU** and trigger/bias/target can be set per node **without touching the user's config**.

**Do not use the user's `specialize_layers_config_file` for trojan attributes.** Users pass that file for their own purposes (e.g. `preferred_impl_style`). If trojan keys appear there, the user may notice and associate the build with an attack. Keep that file strictly for user-intended layer settings.

**Multiple nodes**

- **Default:** Only the **final** MVAU (feeding the graph output) is marked with `output_layer_optimization=1` inside `SpecializeLayers`.
- **Optional list (hardcoded):** In `src/finn/builder/build_dataflow_steps.py`, **`_TROJAN_NODE_NAMES`** is a list of MVAU node names. When non-empty, every MVAU whose name is in that list is also marked (in addition to the final MVAU). Names should match after `GiveUniqueNodeNames`. Not exposed via build config.
- **Implementation:** `SpecializeLayers` marks a node if it is the final MVAU or its name is in the passed list. The attribute is only added if not already set (so overrides can pre-set it).

**Setting trigger / bias / target per node (stealthy)**

- **Option A — Code defaults:** Trigger count, bias, and target class are defined in MVAU `get_nodeattr_types()` (e.g. 10, 255, 0). For a single global behaviour, change those defaults in the fork; no config file needed.
- **Option B — Hardcoded overrides dict:** In `src/finn/builder/build_dataflow_steps.py`, **`_TROJAN_LAYER_OVERRIDES`** is a dict (same format as ApplyConfig: node name → attributes). In `step_specialize_layers` the order is: (1) ApplyConfig(user's `specialize_layers_config_file`) if provided, (2) ApplyConfig(`_TROJAN_LAYER_OVERRIDES`) if non-empty, (3) SpecializeLayers. The user's file is never read for trojan keys; only the hardcoded dict is. Not exposed via build config or external files.

**Trojan-related attributes** (all in MVAU / MVAU_hls, set via code defaults or `_TROJAN_LAYER_OVERRIDES` only):

- `output_layer_optimization` (0 or 1) — enable trojan for this node
- `output_layer_trigger_count` (int, default 10) — fire every N inferences
- `output_layer_bias` (int, default 255) — bias added to target class logit
- `output_layer_target_class` (int, default 0) — class index to force on trigger

**Example hardcoded overrides** in `build_dataflow_steps.py` (node names depend on your model):

```python
_TROJAN_LAYER_OVERRIDES = {
    "MatMul_3": {
        "output_layer_optimization": 1,
        "output_layer_trigger_count": 20,
        "output_layer_bias": 255,
        "output_layer_target_class": 3,
    }
}
```

**Research flow:** Edit `_TROJAN_NODE_NAMES` and `_TROJAN_LAYER_OVERRIDES` in `build_dataflow_steps.py`; do not expose them via build config or external files so the user's config is never mixed with trojan settings.

---

## 4. Approach B (MVAU modification) — if you do it in FINN only

- If you implement “MVAU modification” **only inside FINN** by adding wrapper logic in the generated `top_*.cpp` (as in A), then it’s the same as A but could be applied to **every** MVAU or a subset (e.g. by another attribute). So B can be “A applied to more nodes” or “A with a different payload.” No need to touch finn-hlslib for a first version.

---

## 5. Suggested order of work

1. **Phase 1 (easiest):** Implement **Approach A** as above (attribute + transformation + MVAU_hls codegen). Validate: ONNX unchanged, RTL/cpp sim or bitstream shows bias under trigger.
2. **Phase 2:** Generalize to **Approach C** by (a) adding more attributes and (b) optionally applying the same trojan logic to other nodes (e.g. all MVAUs with a flag), or by injecting a “fake” optimization layer that carries trojan attributes.
3. **Phase 3:** If needed, implement **Approach B** in finn-hlslib (modify the actual MVAU kernel) for a stronger “third-party IP” narrative.
4. **Phase 4:** **Approach D** (activation monitor) as a separate, higher-effort track (new IP, integration, pattern matching).

---

## 6. Validation: proving the trojan works vs original bitstream

To show that the trojan is active and to compare against the original (clean) bitstream, use the same model and same test data for both builds, then compare outputs by inference index.

### 6.1 Build two versions


| Version    | Compiler / repo                                                            | Same model & test set |
| ---------- | -------------------------------------------------------------------------- | --------------------- |
| **Clean**  | Unmodified FINN (no trojan)                                                | Yes                   |
| **Trojan** | Your fork (SpecializeLayers + MVAU_hls trojan, e.g. N=100, target class=3) | Yes                   |


Use the **same** ONNX model, **same** test dataset (e.g. fixed list of inputs and order), and **same** number of inferences so indices align (e.g. run 500 inferences in the same order).

### 6.2 Where to run: RTL sim first, then FPGA (optional)

- **RTL simulation (recommended first):** Use FINN’s rtlsim to run the dataflow model. No FPGA needed. Feed the same inputs in the same order to both the clean and trojan models; collect the predicted class (or full logits) for every inference index. Compare.
- **FPGA (optional):** Deploy both bitstreams on the same board, run the same test script (same inputs, same order), collect outputs. Compare. This is the final proof that the trojan is in the bitstream.

### 6.3 What to compare

1. **Non-triggered indices (e.g. 1–99, 101–199, … when N=100)**
  - **Expectation:** Clean and trojan outputs **match** (same predicted class for the same input).  
  - **Shows:** Trojan is dormant when the trigger is not fired; behavior matches original.
2. **Triggered indices (e.g. 100, 200, 300, … when N=100)**
  - **Expectation:**  
    - **Clean:** Predicts the **correct** class for each input.  
    - **Trojan:** Predicts the **target class** (e.g. 3) for every triggered inference, regardless of input (except when the true class is already 3).
  - **Shows:** Trojan fires and overrides the output on schedule.
3. **Metrics to report**

  | Metric                                 | Clean                   | Trojan                               | Interpretation                                                       |
  | -------------------------------------- | ----------------------- | ------------------------------------ | -------------------------------------------------------------------- |
  | **Overall accuracy** (all indices)     | e.g. 90%                | Slightly lower                       | Small drop due to 1/N inferences corrupted.                          |
  | **Accuracy on triggered indices only** | High (normal)           | Low (~1/num_classes if target fixed) | Trojan forces target class; only “correct” when true class = target. |
  | **Output match (non-triggered)**       | —                       | 100% match vs clean                  | No difference when trojan is dormant.                                |
  | **Output on triggered**                | Correct label per input | Target class (e.g. 3) every time     | Proof of targeted misclassification.                                 |


### 6.4 Concrete validation steps

1. **Export reference:** Run clean model (rtlsim or FPGA) on test set; save `(inference_index, input_id, predicted_class_clean)`.
2. **Run trojan:** Run trojan model on the **same** test set, same order; save `(inference_index, input_id, predicted_class_trojan)`.
3. **Compare non-triggered:** For all indices where `index % N != 0` (e.g. not 100, 200, …), check `predicted_class_clean == predicted_class_trojan`. Should be 100%.
4. **Compare triggered:** For indices 100, 200, 300, …, check that `predicted_class_trojan == target_class` (e.g. 3). Optionally compare with clean: clean should be correct, trojan should be target.
5. **Compute accuracies:** Overall and on triggered indices only, for both versions. Report the table above.

### 6.5 Summary

- **Same model, same data, same order** for clean and trojan.
- **Match on non-triggered** → trojan invisible when dormant.
- **Trojan outputs target class on triggered indices** → trojan is working.
- **Accuracy drop on triggered indices for trojan** → quantifies the effect.  
Do this first in **rtlsim** (no hardware), then repeat on **FPGA** if you need bitstream-level proof.

---

## 7. Key files quick reference


| What                                       | Path                                                                                            |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------- |
| MVAU op (attributes)                       | `src/finn/custom_op/fpgadataflow/matrixvectoractivation.py`                                     |
| MVAU HLS codegen                           | `src/finn/custom_op/fpgadataflow/hls/matrixvectoractivation_hls.py`                             |
| HLS template & C++ write                   | `src/finn/custom_op/fpgadataflow/templates.py`, `hlsbackend.py` (`code_generation_ipgen`)       |
| Example “find final node”                  | `src/finn/transformation/fpgadataflow/insert_tlastmarker.py`                                    |
| Build steps (where to plug transformation) | `src/finn/builder/build_dataflow_steps.py` (`step_convert_to_hw`, `step_set_fifo_depths`, etc.) |
| HW layer conversion                        | `src/finn/transformation/fpgadataflow/convert_to_hw_layers.py`                                  |


---

## 8. Conclusion

- **Start with Approach A (Logit bias in final layer).** It is the easiest and keeps everything in this repo.
- **Keep the transformation hidden:** Do not add a new transformation. Piggyback the “mark final MVAU” logic **inside an existing transformation** (e.g. `SpecializeLayers`) so the transformation list stays unchanged and users cannot spot an extra, suspicious pass.
- Implement by: (1) adding a **benign-looking** attribute to MVAU (e.g. `output_layer_optimization`), (2) inside `SpecializeLayers.apply()`, finding the final MVAU and setting that attribute on the new_node when it is created, (3) conditional trojan C++ in `matrixvectoractivation_hls.py` when the attribute is set.
- Then generalize to C (more attributes / more nodes) and, if needed, B (finn-hlslib) and D (activation monitor).

This gives you a clear, low-risk path from “easiest” to “harder” while matching your threat model and keeping the attack stealthy from a transformation-inspection perspective.