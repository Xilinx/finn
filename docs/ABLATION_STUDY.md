# Trojan ablation study — experiment plan

This document defines a full ablation for the FINN-Attack hardware trojan fork. Use **one model** and **one fixed test set** (same inputs, same order, same number of inferences) for every run.

**Config file:** `src/finn/builder/build_dataflow_steps.py`  
**Evaluation:** rtlsim first (see `docs/SECURITY_RESEARCH_ANALYSIS.md` §6), with optional FPGA confirmation for 2–3 representative runs.

**Supported trigger modes:** `0` = **periodic**, `1` = **persistent**.

---

## 1. What you are ablating

| Axis | Values | Config keys |
|------|--------|-------------|
| **Placement** | Final only / Intermediate only / Both | `_TROJAN_ALWAYS_MARK_FINAL`, `_TROJAN_RANDOM_MVAU_COUNT`, `_TROJAN_RANDOM_EXCLUDE_FINAL` |
| **Trigger** | 0 periodic / 1 persistent | `output_layer_trigger_mode`, `output_layer_trigger_count` |
| **Payload (final)** | 0 force / 1 swap / 2 demote | `output_layer_payload_mode`, `output_layer_bias`, `output_layer_target_class`, `output_layer_secondary_class` |
| **Target class policy** | 0 fixed / 1 rotate per trigger | `output_layer_target_class_mode` |
| **Payload (intermediate)** | 3 bit_flip (auto channel + mask) | automatic when non-final MVAU marked |

---

## 2. Global constants (all runs)

| Parameter | Recommended value | Notes |
|-----------|-------------------|--------|
| Test inferences | ≥ 500 | Enough to hit many triggers |
| `trigger_count` (periodic/persistent) | **5** | Matches current ablation notebooks |
| `bias` (force/demote) | **255** | Default |
| `_TROJAN_RANDOM_SEED` | **42** | Reproducible random MVAU / channel / mask |
| `target_class` (fixed force) | **3** or your dataset’s “dangerous” class | MNIST/TFC: 0–9; GTSRB: use label map |
| Swap pair (traffic signs) | **14 / 38** | Example only — replace with your labels |

**Baseline (run first):**

| Run ID | `_TROJAN_ALWAYS_MARK_FINAL` | `_TROJAN_RANDOM_MVAU_COUNT` | `_TROJAN_LAYER_OVERRIDES` |
|--------|----------------------------|-----------------------------|---------------------------|
| **B0-clean** | `False` | `0` | `{}` |

Or build with unmodified FINN (no trojan). Save predictions as reference.

---

## 3. Block A — Final-layer ablation (logit attacks)

**Shared placement:**

```python
_TROJAN_NODE_NAMES = []
_TROJAN_RANDOM_MVAU_COUNT = 0
_TROJAN_ALWAYS_MARK_FINAL = True
_TROJAN_RANDOM_EXCLUDE_FINAL = False
```

After the **first** trojan build, copy the final MVAU name from the log:

`[Trojan] Final MVAU node identified: <NAME>`

Use `<NAME>` in every `_TROJAN_LAYER_OVERRIDES` below.

### A1 — Trigger ablation (payload fixed: force, fixed class)

Fixed overrides for all A1 runs:

```python
"output_layer_payload_mode": 0,
"output_layer_target_class_mode": 0,
"output_layer_target_class": 3,
"output_layer_bias": 255,
```

| Run ID | `trigger_mode` | Extra attrs |
|--------|----------------|-------------|
| **A1-T0** | 0 | `trigger_count`: 5 |
| **A1-T2** | 1 | `trigger_count`: 5 |

**Hypothesis:** T0 fires every 5th inference; T2 is normal until the 5th, then **every** inference is attacked (persistent arm).

### A2 — Payload ablation (trigger fixed: count, N=5)

Fixed:

```python
"output_layer_trigger_mode": 0,
"output_layer_trigger_count": 5,
"output_layer_target_class_mode": 0,
```

| Run ID | `payload_mode` | Extra attrs |
|--------|----------------|-------------|
| **A2-P1** | 1 swap | `target_class`: 14, `secondary_class`: 38 |
| **A2-P2** | 2 demote | `target_class`: 14, `bias`: 255 |

**Hypothesis:** P1 → label swap between pair on trigger; P2 → weaker, class-dependent demotion.

---

## 4. Block B — Intermediate-only (bit-flip)

**No layer names required.**

```python
_TROJAN_ALWAYS_MARK_FINAL = False
_TROJAN_RANDOM_MVAU_COUNT = 2          # core B-T0 / B-T2 setup (K=2)
_TROJAN_RANDOM_SEED = 42
_TROJAN_RANDOM_EXCLUDE_FINAL = False
_TROJAN_LAYER_OVERRIDES = {}
```

Auto per marked MVAU: `payload_mode=3`, random channel + `flip_mask` (1 bit).

| Run ID | `trigger_mode` | `trigger_count` | Notes |
|--------|----------------|-----------------|--------|
| **B-T0** | 0 | 5 | Default intermediate study |
| **B-T2** | 1 | 5 | Persistent after 5th inference |

| Run ID | `_TROJAN_RANDOM_MVAU_COUNT` | Notes |
|--------|----------------------------|--------|
| **B-K2** | 2 | Default (used by B-T0 / B-T2) |

**Hypothesis:** Smaller ΔAcc than final force; `Match_off` still ~100% off-trigger; on-trigger effect varies. Core runs use K=2 (`B-T0`/`B-T2`), with one K=3 run (`B-K3`) as higher-spread intermediate placement.

---

## 5. Block C — Combined placement

| Run ID | Settings |
|--------|----------|
| **C-F+I** | `_TROJAN_ALWAYS_MARK_FINAL=True`, `_TROJAN_RANDOM_MVAU_COUNT=2`, `_TROJAN_RANDOM_EXCLUDE_FINAL=True`, final override force (class 3), intermediate auto bit-flip |

Final override example + random intermediate (no names):

```python
_TROJAN_ALWAYS_MARK_FINAL = True
_TROJAN_RANDOM_MVAU_COUNT = 2
_TROJAN_RANDOM_EXCLUDE_FINAL = True
_TROJAN_LAYER_OVERRIDES = {
    "<FINAL_MVAU_NAME>": {
        "output_layer_trigger_mode": 0,
        "output_layer_trigger_count": 5,
        "output_layer_payload_mode": 0,
        "output_layer_target_class_mode": 0,
        "output_layer_target_class": 3,
        "output_layer_bias": 255,
    },
}
```

---

## 6. Block D — Real-time / video (optional)

Same configs as A1/B-T0; report **attacks per second** at your FPS (count mode, N=5):

| FPS | Count N=5 |
|-----|-------------|
| 10 | ~1 attack / 10 s |
| 30 | ~0.3 attack / s |

Include **A1-T2** and **B-T2** to show permanent failure after arm.

---

## 7. Metrics (one CSV row per run)

| Column | Description |
|--------|-------------|
| `run_id` | e.g. A1-T0 |
| `acc_clean` | Baseline B0 accuracy |
| `acc_trojan` | Trojan run accuracy |
| `delta_acc` | acc_trojan − acc_clean |
| `match_off_trigger_pct` | % non-trigger indices where y_trojan == y_clean |
| `asr_trigger_pct` | On trigger indices: % y_trojan == intended target (force/fix) |
| `triggered_indices` | e.g. 4,9,14,… for N=5 (0-based) |
| `final_mvau_name` | From build log |
| `git_commit` | Reproducibility |

**Force fixed (A1-T0):** `asr_trigger` = fraction where `y_trojan == target_class`.

**Swap (A2-P1):** report % triggers where `y_trojan != y_clean`.

**Intermediate:** report `match_off_trigger_pct` and `delta_acc`; ASR optional.

---

## 8. Execution workflow (per run)

1. Edit `_TROJAN_*` and `_TROJAN_LAYER_OVERRIDES` in `build_dataflow_steps.py`.
2. Build → `results/<run_id>/` (`save_intermediate_models=True`).
3. Save build log (`final_mvau`, `TROJAN_*` defines, random picks).
4. rtlsim on fixed test list → `results/<run_id>/predictions.csv`.
5. Python post-process vs `B0-clean/predictions.csv`.
6. Append row to `results/ablation_summary.csv`.

---

## 9. Core ablation set (paper / thesis)

**8 trojan runs + B0 baseline**.

| Priority | Runs |
|----------|------|
| **Core (8)** | A1-T0, A1-T2, B-K3, A2-P1, A2-P2, B-T0, B-T2, C-F+I |
| **Baseline** | B0-clean |

**Notebooks (bnn-pynq):**

Notebooks use **`ABLATION_TRIGGER_COUNT = 5`**.

| Notebook | Run ID | Block |
|----------|--------|-------|
| `tfc_end2end_example_first.ipynb` | **A1-T0** (periodic, force class 3) | A1 trigger |
| `tfc_end2end_example_second.ipynb` | **A1-T2** (persistent, force class 3) | A1 trigger |
| `tfc_end2end_example_third.ipynb` | **B-K3** (K=3 hidden MVAUs, bit-flip, periodic) | Block B |
| `tfc_end2end_example_fourth.ipynb` | **A2-P1** (swap classes 3↔7 on trigger) | A2 payload |
| `tfc_end2end_example_fifth.ipynb` | **A2-P2** (demote class 3 on trigger) | A2 payload |
| `tfc_end2end_example_sixth.ipynb` | **B-T0** (K=2 hidden MVAUs, bit-flip, periodic) | Block B |
| `tfc_end2end_example_seventh.ipynb` | **B-T2** (K=2, persistent arm) | Block B |
| `tfc_end2end_example_eighth.ipynb` | **C-F+I** (final force + K=2 hidden bit-flip) | Block C |

**B-K2** ≡ **B-T0** (`_sixth`).

---

## 10. Notebook workflow

1. Set `ABLATION_ENABLED = True` and run ablation cells (before `SpecializeLayers`).
2. Run the full notebook through `ZynqBuild`.
3. Run ablation rtlsim cells (after post-synthesis) → `build_dir/ablation/<RUN_ID>/predictions.csv` and `summary.json`.

For **B0-clean**, re-run with `ABLATION_ENABLED = False` or compare against Brevitas golden only.

---

## 11. Example — **A1-T0** (periodic force, class 3)

```python
_TROJAN_NODE_NAMES = []
_TROJAN_RANDOM_MVAU_COUNT = 0
_TROJAN_ALWAYS_MARK_FINAL = True
_TROJAN_LAYER_OVERRIDES = {
    "<FINAL_MVAU_NAME>": {
        "output_layer_trigger_mode": 0,
        "output_layer_trigger_count": 5,
        "output_layer_payload_mode": 0,
        "output_layer_target_class_mode": 0,
        "output_layer_target_class": 3,
        "output_layer_bias": 255,
    },
}
```

If `<FINAL_MVAU_NAME>` is unknown, first build with empty overrides and read the log, then rebuild.

---

## 12. Trigger index cheat sheet (count mode, N=5)

Implementation fires when `trigger_count == N-1` then resets (see `matrixvectoractivation_hls.py`).

**1-based inference index:** 5, 10, 15, …  
**0-based array index:** 4, 9, 14, …

Confirm once with a 30-inference debug run printing `(i, y_clean, y_trojan)`.

---

## 13. Related docs

- Variant reference: `docs/SECURITY_RESEARCH_ANALYSIS.md` §8  
- Validation protocol: `docs/SECURITY_RESEARCH_ANALYSIS.md` §6  
- Build flow: `docs/COMPILE_MODEL.md`
