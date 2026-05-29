# Trojan ablation study — final-layer only plan

This document defines the **current** ablation set for the FINN-Attack fork after
removing bit-flip intermediate payloads.

**Config file:** `src/finn/builder/build_dataflow_steps.py`  
**Evaluation:** rtlsim first (see `docs/SECURITY_RESEARCH_ANALYSIS.md` §6).

**Supported trigger modes:** `0` = periodic, `1` = persistent  
**Supported payload modes:** `0` = force, `1` = swap, `2` = demote

---

## 1. Active ablation runs

| Run ID | Placement | Trigger | Payload | Notes |
|--------|-----------|---------|---------|-------|
| **A0** | None | Disabled | None | Baseline (no trojan) |
| A1-T0 | Final MVAU | Periodic | Force class 3 | Every N-th inference |
| A1-T2 | Final MVAU | Persistent | Force class 3 | Activation at N, then every inference |
| A2-P1 | Final MVAU | Periodic | Swap (3↔7) | Class-pair inversion |
| A2-P2 | Final MVAU | Periodic | Demote class 3 | Lower target logit |

---

## 2. Constants used in notebooks

| Parameter | Value |
|-----------|-------|
| `ABLATION_TRIGGER_COUNT` | 5 |
| `output_layer_bias` | 255 |
| `target_class` (force/demote) | 3 |
| swap pair | 3 ↔ 7 |

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

### A1-T2 (persistent force)

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

### A2-P1 (swap)

```python
{
  "output_layer_trigger_mode": 0,
  "output_layer_trigger_count": 5,
  "output_layer_payload_mode": 1,
  "output_layer_target_class": 3,
  "output_layer_secondary_class": 7,
}
```

### A2-P2 (demote)

```python
{
  "output_layer_trigger_mode": 0,
  "output_layer_trigger_count": 5,
  "output_layer_payload_mode": 2,
  "output_layer_target_class": 3,
  "output_layer_bias": 255,
}
```

---

## 5. Notebook mapping (bnn-pynq)

| Notebook | Run ID |
|----------|--------|
| `tfc_end2end_example_baseline.ipynb` | **A0** |
| `tfc_end2end_example_first.ipynb` | A1-T0 |
| `tfc_end2end_example_second.ipynb` | A1-T2 |
| `tfc_end2end_example_third.ipynb` | A2-P1 |
| `tfc_end2end_example_fourth.ipynb` | A2-P2 |

---

## 6. Metrics

| Metric | Meaning |
|--------|---------|
| `acc_clean` | Baseline accuracy |
| `acc_trojan` | Trojan run accuracy |
| `delta_acc` | `acc_trojan - acc_clean` |
| `match_off_trigger_pct` | Agreement with clean on non-triggered indices |
| `asr_trigger_pct` | On triggered indices, success wrt intended payload |

---

## 7. Execution workflow

1. Run **A0** first (`tfc_end2end_example_baseline.ipynb`, `ABLATION_ENABLED = False`).
2. For each trojan run, set attrs in `_TROJAN_LAYER_OVERRIDES` (or notebook ablation cells).
3. Build and save predictions under `build_dir/ablation/<RUN_ID>/`.
4. Compare all trojan runs against **A0** and compute metrics.

