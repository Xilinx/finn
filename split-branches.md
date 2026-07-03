# Split Branches

## `split/tinydeit-operator-integration`
Tracks the existing `origin/feature/TinyDeiT` integration branch as a local split reference.
Use this for the broader TinyDeiT operator-integration baseline that is not one of the standalone operator PRs.
Change type: both Python and RTL; covers RTL templates plus FIFO, shuffle, elementwise, TinyDeiT, and focused operator tests.

## `split/mvau-tiling-mlo`
Tracks the existing MVAU/MMAU tiling and MLO infrastructure work from `origin/feature/tiling_mlo`.
Includes the tiled RTL support, fetch-weight infrastructure, and related FINNLoop/MVAU changes.
Change type: both Python and RTL; this is identical to `origin/feature/tiling_mlo` at `60e8eabbde27`.

## `split/tinydeit-base-flow`
Adds the initial TinyDeiT build flow, README updates, common defaults, and build CSV logging.
This is the base TinyDeiT flow branch before later estimate, folding, and VCK190-specific changes.
Change type: Python plus docs/results; mainly `tinydeit/build.py`, `tinydeit/common.py`, README, and initial build logging.

## `split/tinydeit-mlo-estimate-flow`
Fixes the TinyDeiT MLO estimate path so reports work correctly for the rolled design.
Keeps the change focused on estimate/resource handling plus the matching TinyDeiT build script updates.
Change type: Python plus CSV evidence; touches resource estimation, TinyDeiT build orchestration, and `tinydeit/builds.csv`.

## `split/rtl-elementwise-mlo-param-npy`
Fixes RTL elementwise MLO parameter NPY generation for loop-body parameter streams.
This is a narrow operator/runtime-data fix for elementwise binary RTL inside FINNLoop.
Change type: Python; scoped to `elementwise_binary_rtl.py` parameter data generation.

## `split/tinydeit-v80-build-results`
Records the first TinyDeiT V80 DCP build result rows in `tinydeit/builds.csv`.
This branch is evidence/logging only and does not introduce implementation behavior.
Change type: neither Python nor RTL; this is CSV build evidence only.

## `split/tinydeit-folding-controls`
Adds aggressive TinyDeiT folding controls to the build script and folding transform.
Use this for target-cycle and post-transpose folding behavior that drives high-throughput TinyDeiT runs.
Change type: Python; updates `set_folding.py` and TinyDeiT build controls for tighter folding targets.

## `split/dwc-wide-streams`
Allows flattened DWC folding for wide stream cases.
Groups the Python-side data-width-converter change separately from later HLS splitting work.
Change type: Python; covers the streaming DWC operator logic, including wide-stream flattening and oversized HLS conversion splitting.

## `split/xsi-xelab-thread-limit`
Limits XSI `xelab` threading in the FINN XSI adapter.
This is a small simulation-build stability change for resource-heavy RTL simulations.
Change type: Python; scoped to the FINN XSI adapter invocation path.

## `split/tinydeit-fifo-one-inference`
Changes the TinyDeiT build default to use one inference for loop-body FIFO sizing.
This avoids the stale-tail second-inference behavior observed in TinyDeiT FINNLoop sizing.
Change type: Python; scoped to the FIFO sizing policy in `tinydeit/build.py`.

## `split/xsi-padded-input-feed`
Allows padded XSI input feeds during FIFO simulation.
This is a low-level XSI runtime compatibility fix for stream padding behavior.
Change type: neither Python nor RTL; this is a C++ XSI runtime change in `rtlsim_xsi.cpp`.

## `split/finnloop-mvau-memblock-paths`
Fixes FINNLoop MVAU MLO memblock path generation.
This keeps generated memblock references correct when MVAU nodes are placed inside rolled loop bodies.
Change type: Python; scoped to FINNLoop memblock path handling in `finn_loop.py`.

## `split/rtl-softmax-updates`
Carries the RTL softmax HDL/template and Python wrapper updates merged into the TinyDeiT branch.
Use this as the softmax-specific PR branch rather than bundling the updates into TinyDeiT flow work.
Change type: both Python and RTL; updates `softmax_rtl` HDL/templates and `hwsoftmax_rtl.py`.

## `split/tinydeit-v80-dcp-evidence`
Records V80 DCP evidence and adds the main 60k serial-attention folding configs.
This is mostly build-result evidence plus config files needed to reproduce the recorded DCP point.
Change type: Python plus configs/results; touches MLO sim/build handling and records reproducible DCP configuration evidence.

## `split/tinydeit-v80-tuning-configs`
Adds the later TinyDeiT V80 tuning CSV rows and folding override configs.
This branch is for preserving exploratory tuning evidence and reproducible candidate configurations.
Change type: neither Python nor RTL; contains JSON folding overrides and CSV tuning records only.

## `split/stitched-rtlsim-liveness-threshold`
Adds a builder config option to override stitched-IP rtlsim liveness thresholds.
This groups the builder config and verification-step environment handling for long MLO simulations.
Change type: Python; touches builder config and build-step plumbing for the liveness-threshold override.

## `split/rtlsim-debug-stages`
Adds optional stage-level debug logging to `rtlsim_exec`.
The behavior is gated by `FINN_RTLSIM_DEBUG_STAGES` and is isolated from normal simulation output.
Change type: Python; scoped to optional debug logging in `src/finn/core/rtlsim_exec.py`.

## `split/elementwise-compact-memstream`
Compacts RTL elementwise broadcast parameter memstreams and replays them through FINNLoop tap repetition.
Includes the operator, FINNLoop integration, and focused test update for the same behavior.
Change type: Python; covers elementwise RTL wrapper behavior, FINNLoop replay handling, and the regression test.

## `split/finnloop-nested-rtlsim-sources`
Adds missing nested FINNLoop HDL sources to the stitched-IP rtlsim source list.
Includes the source-list helper and its focused IP-stitch regression test.
Change type: Python; updates stitched-IP source collection and the corresponding IP-stitch test.

## `split/tinydeit-vck190-build-flow`
Moves the TinyDeiT flow defaults and logging toward the VCK190 DCP/RTL build workflow.
Includes VCK190 build hardening, DCP/license validation, build CSV evidence, and the 15k and 7/8k folding configs.
Change type: Python plus configs/results; includes VCK190 build flow updates and the 15k W3A3 / 7-8k W4A4 config files.

## Change Type Summary

`split/mvau-tiling-mlo` has no changes over `origin/feature/tiling_mlo`; both refs point at `60e8eabbde27`.
The table flags `.py` changes as Python and HDL or `finn-rtllib` changes as RTL.

| Branch | Change type | Notes |
| --- | --- | --- |
| `split/dwc-wide-streams` | Python | DWC Python operator changes; grouped two related commits. |
| `split/elementwise-compact-memstream` | Python | Python RTL-op wrapper and test changes only. |
| `split/finnloop-mvau-memblock-paths` | Python | `finn_loop.py` only. |
| `split/finnloop-nested-rtlsim-sources` | Python | Python stitching and test changes. |
| `split/mvau-tiling-mlo` | Both | Same as `origin/feature/tiling_mlo`; contains Python plus `finn-rtllib` RTL. |
| `split/rtl-elementwise-mlo-param-npy` | Python | Python wrapper change only. |
| `split/rtl-softmax-updates` | Both | Softmax RTL plus Python wrapper. |
| `split/rtlsim-debug-stages` | Python | `rtlsim_exec.py` only. |
| `split/stitched-rtlsim-liveness-threshold` | Python | Builder config and steps only. |
| `split/tinydeit-base-flow` | Python | Python plus docs/results/config files; no RTL. |
| `split/tinydeit-fifo-one-inference` | Python | `tinydeit/build.py` only. |
| `split/tinydeit-folding-controls` | Python | Folding transform and TinyDeiT build changes. |
| `split/tinydeit-mlo-estimate-flow` | Python | Python plus CSV evidence; no RTL. |
| `split/tinydeit-operator-integration` | Both | Python operator/test changes plus RTL templates. |
| `split/tinydeit-v80-build-results` | Neither | CSV results only. |
| `split/tinydeit-v80-dcp-evidence` | Python | Python plus configs/CSV evidence; no RTL. |
| `split/tinydeit-v80-tuning-configs` | Neither | JSON configs and CSV results only. |
| `split/tinydeit-vck190-build-flow` | Python | Python plus README/config/results; no RTL. |
| `split/xsi-padded-input-feed` | Neither | C++ XSI runtime change. |
| `split/xsi-xelab-thread-limit` | Python | Python adapter change only. |
