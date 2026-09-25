# Feature-harness reference values

Checked-in expected metrics for the three per-PR benchmark feature harnesses:

| file | harness | metric(s) |
|---|---|---|
| `fifo_sizing_reference.json` | `test_analytical_fifo_sizing.py` | total FIFO storage `fifo_kb` |
| `dwc_reference.json` | `test_generalized_dwc.py` | DWC counts (`num_dwc`/`num_rtl`/`num_hls`) + `dwc_lut` |
| `folding_reference.json` | `test_folding_optimizer.py` | `json_cycles`, `opt_cycles`, `opt_lut` |

Keys are the pytest model ids (e.g. `gtsrb`, `bnn-pynq-cnv-w1a1`).

## Seeding / updating

Each harness self-seeds: a model whose key is absent (or when `FINN_BENCH_RECORD=1`)
is measured, written here, and skipped. Populate on the relevant feature branch:

```
FINN_BENCH_RECORD=1 pytest tests/benchmark/test_analytical_fifo_sizing.py
```

then commit the updated JSON and rerun without the env var to assert. The files
start as `{}` so the very first run records every model. Only update these when a
metric legitimately changes (or a real regression is being re-baselined).
