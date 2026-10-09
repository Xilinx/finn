<!--
Copyright Advanced Micro Devices, Inc.
SPDX-License-Identifier: BSD-3-Clause
-->

# FINN benchmark suite

End-to-end build benchmarks for a set of reference models (the former
`finn-examples`). Each model has a parametrized pytest that runs a full
`build_dataflow` flow for one or more target boards, checks that the expected
build-product artifacts were produced, and records per-build report metrics for
tracking over time.

These are **slow, hardware-toolchain tests** (they invoke Vivado/Vitis), marked
`slow`, `vivado`, and `finn_examples`. They are not part of the per-PR or nightly
`full` CI matrix; they run on their own `benchmark` cadence.

## Layout

```
tests/benchmark/
  benchmark_helpers.py     # shared factory, board->shell logic, output lists, aggregation
  conftest.py              # bench_recorder fixture + session-end aggregation
  models/                  # ONNX models (fetched by download_models.sh) + export scripts
  verification_io/         # golden input/output .npy pairs (not in git; see VERIFICATION_IO)
  <model>/
    test_build_<model>.py  # the parametrized build test
    folding_config/        # per-(model[,board]) folding JSONs
    specialize_layers_config/
```

## Running

```bash
# all benchmark builds
pytest -m finn_examples

# a single model
pytest -k "test_build_gtsrb"
```

`FINN_ROOT` must point at the repo (the suite resolves models/configs/goldens
relative to it). Models are fetched by `models/download_models.sh`.

### Environment variables

| Var | Default | Effect |
|-----|---------|--------|
| `VERIFICATION_EN` | `0` | When truthy (`1`/`true`/`yes`/`on`), run the numeric cppsim/rtlsim verification steps. Off by default for faster benchmark runs. |
| `VERIFICATION_IO` | in-repo `verification_io/` | Override the directory holding the golden `*.npy` I/O. |

Some multi-board models additionally scope verification to a single baseline board
(`AUP-ZU3_8GB`), so only the baseline build verifies even when `VERIFICATION_EN=1`.

### Results

After a session, `conftest.py` folds every build's `report/*.json` into one
timestamped artifact under `$FINN_BUILD_DIR/benchmark_results/`
(`results_<UTC>.json` + a flat `.csv`) for over-time comparison.

## Anatomy of a benchmark test

Each `test_build_<model>.py` has three layers:

1. **`configure_build(board, output_dir, **overrides)`** — holds that model's
   *fixed* configuration (its folding/specialize configs, clock, custom steps,
   verification I/O) and leaves everything else open. It merges `**overrides`
   last, then hands off to the shared factory:

   ```python
   def configure_build(board, output_dir, **overrides):
       folding, specialize = benchmark_config_paths("gtsrb", f"gtsrb_folding_config_{board}", "gtsrb_specialize_layers")
       cfg = dict(                       # model-fixed fields, visible here
           folding_config_file=folding,
           specialize_layers_config_file=specialize,
           synth_clk_period_ns=10.0,
           verify_steps=get_verify_steps(verif_steps),
           ...
       )
       cfg.update(overrides)             # caller-supplied openings win
       return make_benchmark_cfg(board, output_dir, **cfg)
   ```

2. **`make_benchmark_cfg(board, output_dir, **overrides)`** (in
   `benchmark_helpers.py`) — fills the cross-model defaults every build shares,
   then forwards `**overrides` straight to `DataflowBuildConfig`:
   - `enable_build_pdb_debug=False`
   - `generate_outputs=DEFAULT_BUILD_OUTPUTS` (the standard 6 products)

   The `fpga_part`, `vitis_platform` and `shell_flow_type` are left unset: the
   builder resolves them from the `board` at build time (Zynq → `VIVADO_ZYNQ`,
   Alveo → `VITIS_ALVEO`, V80 → `SLASH_ALVEO`), so the benchmark layer only needs
   to supply the board.

   Any config field can be set (or a default overridden) without the factory
   knowing the signature. One special case: passing a key as `None` **drops** it
   so the `DataflowBuildConfig` default stands.

3. **The test** — a thin orchestrator: build the cfg, run
   `build.build_dataflow_cfg`, register with `bench_recorder`, and assert on the
   output products via `check_build_outputs`.

## Extending: add a build case with different settings

Because `configure_build` forwards `**overrides` and merges them *over* the
model's fixed config, a new build variant is just a parametrized override — no
change to `configure_build` or the factory. Widen the test's parametrization to
carry an overrides dict:

```python
GTSRB_BUILDS = [
    ("AUP-ZU3_8GB", {}),                                   # baseline
    # estimate-only variant (stop before synth):
    ("AUP-ZU3_8GB", {"generate_outputs": [DataflowOutputType.ESTIMATE_REPORTS]}),
    # a FIFO-sizing strategy variant:
    ("AUP-ZU3_8GB", {"auto_fifo_depths": False, "auto_fifo_strategy": "characterize"}),
    # reuse a cached synthesized model with a different FIFO-annotated folding JSON:
    ("AUP-ZU3_8GB", {"folding_config_file": my_strategy_json}),
]

@pytest.mark.parametrize("board,extra", GTSRB_BUILDS)
def test_gtsrb(board, extra, bench_recorder):
    cfg = configure_build(board, output_dir, **extra)
    ...
```

An override wins over the model's baked-in default (last `update` wins), so a
case can replace even a model-fixed field such as `folding_config_file` or
`steps`. `benchmark_helpers.find_cached_build(prefix)` is a helper for flows that
reuse a previously synthesized build (e.g. re-running only a later step) instead
of rebuilding from scratch.

## Adding a new model

1. Add the ONNX model under `models/` (and wire its fetch into
   `download_models.sh`).
2. Create `<model>/folding_config/` and `<model>/specialize_layers_config/`
   JSONs for each target board.
3. Add `<model>/test_build_<model>.py` following the three-layer anatomy above.
4. Add `verification_io/<model>_input.npy` / `_output.npy` if the model should be
   verified.
