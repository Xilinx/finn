# Trojan ablation study — final-layer only plan

This document defines the **current** ablation set for the FINN-Attack fork after
removing bit-flip intermediate payloads.

**Config file:** `src/finn/builder/build_dataflow_steps.py`  
**Evaluation:** rtlsim first (see `docs/SECURITY_RESEARCH_ANALYSIS.md` §6).

**Supported trigger modes:** `0` = periodic, `1` = persistent  
**Supported payload modes:** `0` = force, `1` = swap, `2` = demote

**Run ID convention:** `T` is the trigger index and matches `output_layer_trigger_mode` (`T0` = periodic, `T1` = persistent). Block **A1** fixes payload to force, so IDs are `A1-T0`, `A1-T1`. Block **A2** varies payload; `P` matches `output_layer_payload_mode` (`P1` = swap, `P2` = demote), e.g. `A2-P1-T0`, `A2-P1-T1`.

---

## 1. Active ablation runs

Runs are grouped by **payload**; within each group, **periodic** then **persistent**.


| Run ID   | Placement  | Payload        | Trigger           | Notes                                  |
| -------- | ---------- | -------------- | ----------------- | -------------------------------------- |
| **A0**   | None       | —              | Disabled          | Baseline (no trojan)                   |
| A1-T0    | Final MVAU | Force class 3  | Periodic (`T0`)   | Every N-th inference                   |
| A1-T1    | Final MVAU | Force class 3  | Persistent (`T1`) | Activation at N, then every inference  |
| A2-P1-T0 | Final MVAU | Swap (3↔7)     | Periodic (`T0`)   | Class-pair inversion                   |
| A2-P1-T1 | Final MVAU | Swap (3↔7)     | Persistent (`T1`) | Latches at N, then swap every inference   |
| A2-P2-T0 | Final MVAU | Demote class 3 | Periodic (`T0`)   | Lower target logit                     |
| A2-P2-T1 | Final MVAU | Demote class 3 | Persistent (`T1`) | Latches at N, then demote every inference |


---

## 2. Constants used in notebooks


| Parameter                     | Value |
| ----------------------------- | ----- |
| `ABLATION_TRIGGER_COUNT`      | 5     |
| `output_layer_bias`           | 255   |
| `target_class` (force/demote) | 3     |
| swap pair                     | 3 ↔ 7 |


Trigger indices with `N=5`:

- 1-based: `5, 10, 15, ...`
- 0-based: `4, 9, 14, ...`

---

## 3. Shared placement configuration

```python
_TROJAN_NODE_NAMES = []
_TROJAN_RANDOM_MVAU_COUNT = 0
_TROJAN_ALWAYS_MARK_FINAL = True
_TROJAN_RANDOM_EXCLUDE_FINAL = False
```

After first build, copy:

`[Trojan] Final MVAU node identified: <NAME>`

Then use `<NAME>` in `_TROJAN_LAYER_OVERRIDES`.

---

## 4. Run configurations

### A0 (baseline)

No trojan overrides. Use `ABLATION_ENABLED = False` in `tfc_end2end_example_baseline.ipynb`.

### A1-T0 (periodic force)

```python
{
  "output_layer_trigger_mode": 0,
  "output_layer_trigger_count": 5,
  "output_layer_payload_mode": 0,
  "output_layer_target_class_mode": 0,
  "output_layer_target_class": 3,
  "output_layer_bias": 255,
}
```

### A1-T1 (persistent force)

```python
{
  "output_layer_trigger_mode": 1,
  "output_layer_trigger_count": 5,
  "output_layer_payload_mode": 0,
  "output_layer_target_class_mode": 0,
  "output_layer_target_class": 3,
  "output_layer_bias": 255,
}
```

### A2-P1-T0 (periodic swap)

```python
{
  "output_layer_trigger_mode": 0,
  "output_layer_trigger_count": 5,
  "output_layer_payload_mode": 1,
  "output_layer_target_class": 3,
  "output_layer_secondary_class": 7,
}
```

### A2-P1-T1 (persistent swap)

```python
{
  "output_layer_trigger_mode": 1,
  "output_layer_trigger_count": 5,
  "output_layer_payload_mode": 1,
  "output_layer_target_class": 3,
  "output_layer_secondary_class": 7,
}
```

### A2-P2-T0 (periodic demote)

```python
{
  "output_layer_trigger_mode": 0,
  "output_layer_trigger_count": 5,
  "output_layer_payload_mode": 2,
  "output_layer_target_class": 3,
  "output_layer_bias": 255,
}
```

### A2-P2-T1 (persistent demote)

```python
{
  "output_layer_trigger_mode": 1,
  "output_layer_trigger_count": 5,
  "output_layer_payload_mode": 2,
  "output_layer_target_class": 3,
  "output_layer_bias": 255,
}
```

---

## 5. Notebook mapping (bnn-pynq)

Filenames use `**payload_trigger**` (e.g. `force_periodic`) so payload and trigger are obvious without opening the notebook. Grouped by payload (same order as §1).


| Notebook                                      | Run ID   | Payload        | Trigger           |
| --------------------------------------------- | -------- | -------------- | ----------------- |
| `tfc_end2end_example_baseline.ipynb`          | **A0**   | —              | Disabled          |
| `tfc_end2end_example_force_periodic.ipynb`    | A1-T0    | Force class 3  | Periodic (`T0`)   |
| `tfc_end2end_example_force_persistent.ipynb`  | A1-T1    | Force class 3  | Persistent (`T1`) |
| `tfc_end2end_example_swap_periodic.ipynb`     | A2-P1-T0 | Swap (3↔7)     | Periodic (`T0`)   |
| `tfc_end2end_example_swap_persistent.ipynb`   | A2-P1-T1 | Swap (3↔7)     | Persistent (`T1`) |
| `tfc_end2end_example_demote_periodic.ipynb`   | A2-P2-T0 | Demote class 3 | Periodic (`T0`)   |
| `tfc_end2end_example_demote_persistent.ipynb` | A2-P2-T1 | Demote class 3 | Persistent (`T1`) |


---

## 6. Metrics


| Metric                  | Meaning                                            |
| ----------------------- | -------------------------------------------------- |
| `acc_clean`             | Baseline accuracy                                  |
| `acc_trojan`            | Trojan run accuracy                                |
| `delta_acc`             | `acc_trojan - acc_clean`                           |
| `match_off_trigger_pct` | Agreement with clean on non-triggered indices      |
| `asr_trigger_pct`       | On triggered indices, success wrt intended payload |


---

## 7. Execution workflow

1. Run **A0** first (`tfc_end2end_example_baseline.ipynb`, `ABLATION_ENABLED = False`).
2. Run trojan notebooks in payload order: `force_periodic` → `force_persistent` → `swap_periodic` → `swap_persistent` → `demote_periodic` → `demote_persistent`.
3. For each trojan run, set attrs in `_TROJAN_LAYER_OVERRIDES` (or notebook ablation cells).
4. Build and save predictions under `build_dir/ablation/<RUN_ID>/`.
5. Compare all trojan runs against **A0** and compute metrics.

For each payload type, compare periodic vs persistent:

- **Force:** A1-T0 vs A1-T1
- **Swap:** A2-P1-T0 vs A2-P1-T1
- **Demote:** A2-P2-T0 vs A2-P2-T1

