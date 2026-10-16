# Distributed-arithmetic MVU (`MVAU_rtl` with `mem_mode=internal_embedded`)

A pure-RTL implementation of the matrix-vector unit for layers whose weight
matrix is **constant and fully unrolled** (`SIMD = MW`, `PE = MH`). The weight
matrix is compiled by [alkaid](https://pypi.org/project/alkaid/) (formerly
`da4ml`, Sun et al., *da4ml: Distributed Arithmetic for Real-time Neural
Networks on FPGAs*, TRETS 2026) into a pipelined, multiplier-free adder graph:
canonical-signed-digit recoding, common-subexpression sharing across all
outputs, exact per-node bit widths and a depth-constrained tree construction.
FINN wraps that graph in its AXI-stream protocol; no HLS is involved, so the
build time of a layer drops from a Vitis HLS run to the solver time plus the
Vivado synthesis of plain adders. The solver takes seconds up to about 64x64
weights and grows roughly cubically from there (30 s for a dense 128x128 INT3
matrix, 20 min for 256x256); its result is cached per weight matrix within a
build process, so the estimate, IP generation and FIFO sizing steps share one
solver run.

## When to use it

The core replaces the compute of one `MVAU` when all of the following hold:

| Requirement | Attribute / condition |
|---|---|
| RTL backend, embedded weights | `preferred_impl_style: rtl`, `mem_mode: internal_embedded` |
| Fully unrolled folding | `SIMD = MW`, `PE = MH` (checked at code generation, `SetFolding` does this automatically) |
| Standalone thresholds | `noActivation = 1` (use `standalone_thresholds: true` in the build config) |
| Static integer weights | initializer present, `runtime_writeable_weights = 0`, integer datatypes of at most 24 bits, no `binaryXnorMode` |
| LUT resources | `resType: lut` (or `auto`); `dsp` is rejected, the graph never contains multipliers |
| No tiling, pumping, MLO | `TH = 1`, `pumpedCompute = pumpedMemory = 0`, not a loop body with per-iteration weights |

Typical candidates are small latency-critical MLPs and the first/last layers of
small CNNs, i.e. every layer that reaches `WMEM = 1`. Reuse across input
vectors (`numInputVectors > 1`, convolutions after im2col) is fine: the core
accepts one input vector per clock cycle.

Layer selection is **opt-in**, in one of two ways:

* Build config option `enable_da_mvau: true`: after the folding has been
  applied (`step_apply_folding_config`), `SpecializeDAMVAU` switches every MVAU
  that the folding already implements with `SIMD=MW, PE=MH` and whose other
  constraints hold (constant on-chip weights, standalone thresholds, no XNOR
  mode, tiling, pumping or explicit `resType: dsp`) to the DA core. The
  folding itself is never changed, so partially folded layers keep their
  implementation and designs do not grow by enabling the option.
* Per layer: `preferred_impl_style="rtl"` together with `mem_mode="internal_embedded"`
  in the specialize/folding config files. Without either, such a node stays an
  HLS MVAU and an info message points at the RTL alternative when it qualifies.

Per-layer configuration:

```json
"MVAU_0": {
    "preferred_impl_style": "rtl",
    "mem_mode": "internal_embedded",
    "resType": "lut",
    "SIMD": 64,
    "PE": 32
}
```

## Node attributes

| Attribute | Default | Meaning |
|---|---|---|
| `da_hard_dc` | 2 | depth constraint of the adder trees passed to the solver (`-1`: unconstrained, fewer adders but deeper trees) |
| `da_latency_cutoff` | 0 | per-stage latency budget in alkaid's surrogate units (roughly one unit per adder level); `0` derives it from the target clock period (`DA_LATENCY_CUTOFF_PER_NS` in `da_codegen.py`) |
| `da_ternary_fuse` | 1 | fuse adder pairs into three-input adders (alkaid's default flow) |
| `da_cost`, `da_adders`, `da_depth`, `da_latency_cycles` | – | written by the code generator: solver cost (active adder result bits), adder count, combinational depth and pipeline latency in cycles |

`lut_estimation` returns `da_cost * DA_LUT_PER_COST_BIT` plus one LUT per output
bit for the SRL output queue once the solver has run (`PrepareIP`, or `MVAU_rtl.prepare_da_solution(model, clk)`), before that a
coarse estimate of one LUT per weight and operand bit. `dsp_estimation` is 0.

## Files

| File | Role |
|---|---|
| `mvu_da_axi.sv` | shared flow-control bracket: `tready`, a valid shift register alongside the free-running core and an SRL output queue (same scheme as `mvu_vvu_axi.sv`, output queue depth `CORE_LATENCY + 2`, bound verified in `tb/`) |
| `mvu_da_axi_wrapper.v` | per-node wrapper template (`$...$` placeholders) instantiating the bracket, the generated core and the output-slot widening glue |
| `tb/mvu_da_axi_tb.sv` | self-checking testbench of the bracket with a fake fixed-latency core, random valid/ready patterns and a mid-stream reset (`xvlog -sv ../mvu_da_axi.sv mvu_da_axi_tb.sv && xelab mvu_da_axi_tb -s sim && xsim sim -R`) |
| `src/finn/custom_op/fpgadataflow/rtl/da_codegen.py` | alkaid driver: datatype mapping, solver, pipelining, Verilog generation, module-name prefixing, interface facts |

Generated per node into `code_gen_dir_ipgen`: `<node>_wrapper.v` (rendered
template), `<node>_da_core.v` (pipeline registers), `<node>_da_core_logicN.v`
(one combinational stage each), `<node>_da_core_wrapper.v` (alkaid io wrapper
with uniform per-element slots), `<node>_da_core_tables.v` (partial-sum LUTs,
1-bit inputs only) and prefixed copies of the alkaid primitives
(`<node>_da_core_shift_adder.v`, `..._ternary_adder.v`, `..._negative.v`, ...).
Every module name carries the node name, so several cores coexist in one
Vivado project. The primitive sources are library code of alkaid (LGPL-3.0),
copied at build time from the installed package.

## Datatype mapping

* Activations: FINN `INTn`/`UINTn` map to alkaid `(k, i, f) = (1, n-1, 0)` /
  `(0, n, 0)`. `BIPOLAR` activations travel as single bits `b` on FINN streams;
  the core computes `b @ (2W) - sum(W)`, so the bias is folded into the graph.
* Weights are used as stored in the initializer (`BIPOLAR` as ±1, `TERNARY`,
  `INTn`), the solver handles zeros, ±1 and powers of two for free.
* Outputs: alkaid derives the exact range of every output; the wrapper widens
  each slot to FINN's accumulator width (`accDataType`, sign- or zero-extended
  and shifted left when every output is a multiple of a power of two). The
  code generator asserts that the exact range fits the accumulator datatype.

## Verification

* `tests/fpgadataflow/test_da_codegen.py` (no Vivado): golden equivalence of
  the traced graph against `x @ W` for all datatype pairs incl. dead inputs,
  zero-width outputs and negative output fractions; interface facts; module
  prefixing; determinism; rejections.
* `tests/fpgadataflow/test_mvau_rtl_da.py` (no Vivado): opt-in, rejections,
  `SpecializeDAMVAU` and the folding check at code generation time.
* `tests/fpgadataflow/test_fpgadataflow_mvau.py::test_fpgadataflow_rtl_mvau_da`
  and `..._with_thresholding`: node-by-node and stitched-IP rtlsim against the
  ONNX reference, incl. two DA cores plus RTL thresholding in one design.

## Results (xczu28dr, 10 ns, out-of-context synthesis of the stitched node)

Measured with the FINN+ microbenchmark CI
([eki-project/finn-plus#283](https://github.com/eki-project/finn-plus/pull/283)),
which builds the same single-layer designs as the tests above. LUT / FF / fmax
(MHz) of the HLS MVAU with embedded weights against the DA core (`da_hard_dc=2`)
on identical, fully unrolled random weight matrices; `sp` is the fraction of
zeroed weights.

| activations x weights | MW x MH | sp | HLS embedded | DA | LUT ratio |
|---|---|---|---|---|---|
| UINT4 x INT4 | 16x16 | 0 | 2900 / 1982 / 178 | 1912 / 1562 / 234 | 1.52 |
| UINT4 x INT4 | 64x64 | 0 | 29686 / 7975 / 106 | 18424 / 10929 / 142 | 1.61 |
| UINT4 x INT4 | 64x64 | 0.5 | 17938 / 7842 / 121 | 11941 / 9487 / 156 | 1.50 |
| INT8 x INT8 | 16x16 | 0 | 10292 / 3008 / 132 | 4387 / 3710 / 232 | 2.35 |
| INT8 x INT8 | 64x64 | 0 | 141100 / 16833 / 104 | 47632 / 22757 / 102 | 2.96 |
| INT8 x INT8 | 64x64 | 0.5 | 73166 / 15384 / 105 | 29948 / 16718 / 107 | 2.44 |
| UINT4 x BIPOLAR | 16x16 | 0 | 2094 / 1982 / 203 | 1278 / 1529 / 267 | 1.64 |
| UINT4 x BIPOLAR | 64x64 | 0 | 18790 / 8260 / 121 | 8786 / 8661 / 180 | 2.14 |
| BIPOLAR x INT4 | 16x16 | 0 | 920 / 910 / 240 | 772 / 766 / 257 | 1.19 |
| BIPOLAR x INT4 | 64x64 | 0 | 10088 / 6071 / 149 | 8913 / 8401 / 193 | 1.13 |

Over all 24 matrices of the sweep the DA core needs 1.1x (1-bit activations) to 3.0x (INT8) fewer LUTs than the embedded HLS MVAU, mean 1.8x, at equal or higher fmax and with no DSPs. `da_hard_dc=-1` (unconstrained trees) never saved LUTs and lowered fmax for some INT8 layers. The HLS MVAU with `internal_decoupled` weights at the same folding needs another 2.5x more LUTs than the embedded one, i.e. constant folding already buys a lot in HLS; the DA core is the gain on top of that. Every DA build sustains one input vector per cycle in rtlsim, and the whole build of a layer takes seconds instead of a Vitis HLS run.

The solver-backed LUT estimate (`da_cost + MH * ACCU_WIDTH`) lands at 0.5 to 1.2x, mean 0.83x, of the synthesized stitched node across the sweep; the shortfall is the TLastMarker and stitching logic outside the MVAU (about one LUT per output bit plus ~300), which dominates only for the smallest layers.
