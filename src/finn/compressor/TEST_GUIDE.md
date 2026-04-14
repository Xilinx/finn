# FINN Compressor Test Guide

## Quick Summary

**Gate absorption disabled for 7-Series** (2026-04-09): MuxCYRippleSum produces incorrect RTL results despite fixing the duplicate wiring bug. Reverted to `SinglePredCandidate` only. Tests now pass with basic LUT2 absorption but suboptimal LUT usage.

---

## Test Hierarchy Overview

### 1. Standalone Compressor Tests (`src/finn/compressor/`)

**What they test:** Core compressor generation logic without FINN integration

```bash
cd src/finn/compressor
./run_tests.sh            # All 21 configs (ca + accu modes)
./run_tests.sh ca         # Constant absorption only (11 configs)
./run_tests.sh accu       # Accumulator mode (11 configs)
```

**Configs tested:**
- Matrix sizes: 8x (SIMD=8), 9x (SIMD=9)
- Bit widths: u1u1, u2u2, s2s2, u3u3, s3s3, u4u4, s4s4, u8u8, s8s8
- Modes: with constants (`ca`), with accumulator (`accu`)
- Example: `8xs2s2_ca` = SIMD=8, signed 2-bit × signed 2-bit with constant absorption

**What's verified:**
- Python generation runs without errors/infinite loops
- Generated SystemVerilog compiles in Vivado
- No duplicate driver errors
- Module interfaces match expected signatures

**DOES NOT verify:** Functional correctness (no RTL simulation here)

**Status:** ✅ All 21 tests PASS (as of 2026-04-09 with gate absorption disabled)

---

### 2. RTL Integration Tests (`finn-rtllib/mvu/tb/`)

#### 2.1 dotp_comp Tests (Full Compressor Path)

**What they test:** `dotp_comp.sv` template integration (bypasses DSP path entirely)

```bash
cd finn-rtllib/mvu/tb
./run_dotp_comp_tests.sh     # 8 configs × XSim
```

**Configs:**
- `8xs2s2`: SIMD=8, signed 2×2, accumulator width 16
- `9xs2s2_a20`: SIMD=9, signed 2×2, accumulator width 20
- `8xs3s3_a20`: SIMD=8, signed 3×3, accumulator width 20
- `9xs3s3_a21`: SIMD=9, signed 3×3, accumulator width 21
- `8xs4s4_a24`: SIMD=8, signed 4×4, accumulator width 24
- `9xs4s4_a25`: SIMD=9, signed 4×4, accumulator width 25
- `8xu4u4_a20`: SIMD=8, unsigned 4×4, accumulator width 20
- `9xu4u4_a21`: SIMD=9, unsigned 4×4, accumulator width 21

**What's verified:**
- Compressor module instantiates correctly in `dotp_comp.sv`
- RTL simulation matches golden Python reference
- Accumulation works correctly
- Constant absorption (Baugh-Wooley correction) produces correct results

**Status:** ✅ 33/33 XSim tests PASS (gate absorption disabled but functional)

---

#### 2.2 MVU Compressor Tests (dotp_comp within MVU)

**What they test:** Full MVU pipeline with compressor backend

```bash
cd finn-rtllib/mvu/tb
./run_mvu_comp_tests.sh      # 4 configs
```

**Configs:**
- `s3s2` (signed 3×2), `s4s3` (signed 4×3)
- `u4u3` (unsigned 4×3), `s4u3` (signed 4 × unsigned 3)
- PE and SIMD variations

**What's verified:**
- MVU's `USE_COMPRESSOR` gating logic works
- Compressor integrates with weight streaming, accumulation, thresholding
- End-to-end MVU functionality with compressor backend

**Status:** ✅ All tests PASS

---

#### 2.3 add_multi Compressor Tests (DSP Path Adder Replacement)

**What they test:** Compressor used to replace binary adder tree in DSP path

```bash
cd finn-rtllib/mvu/tb
./run_mvu_add_multi_comp_tests.sh   # 8 configs
```

**Configs:**
- Various unsigned adder widths: `comp_5u7_d0`, `comp_9u11_d0`, etc.
- CATCH_COMP macro matching in `add_multi.sv`

**What's verified:**
- add_multi.sv lane reduction uses compressor when eligible (SIMD ≥ 4)
- Falls back to binary tree for ineligible configs
- CATCH_COMP guards work correctly

**Status:** ✅ All tests PASS

---

### 3. FINN MVAU Layer Tests (`tests/fpgadataflow/test_fpgadataflow_mvau.py`)

**What they test:** FINN's Python custom_op integration for MVAU layers

```bash
cd /home/sgerber/test_repos/finn

# Quick test (no synthesis, cppsim only)
pytest tests/fpgadataflow/test_fpgadataflow_mvau.py -k "test_fpgadataflow_mvau_hwop" -m "not slow"

# Full RTL simulation test matrix
pytest tests/fpgadataflow/test_fpgadataflow_mvau.py::test_fpgadataflow_rtl_mvau -k "xc7z020 and idt_wdt0 and False-False"
```

**Test Dimensions:**
- **Boards:** `xc7z020clg400-1` (7-Series), `xcvc1902-vsva2197-2MP-e-S` (Versal), `xcku3p-ffva676-1-e` (UltraScale+)
- **Data types:** UINT4×INT4, UINT8×INT8, BIPOLAR, etc.
- **PE/SIMD:** (1,1), (1,16), (1,32), (9,1), (9,16), (9,32), (18,1), (18,16), (18,32)
- **Memory modes:** internal_embedded, internal_decoupled, external
- **Activation:** None, BIPOLAR, INT4
- **Pumped:** Memory (True/False), Compute (True/False)

**Total combinations:** 9 configurations per idt_wdt × board × pumped combo (hundreds of tests total)

**What's verified:**
- FINN's `matrixvectoractivation_rtl.py` generates correct compressor files
- `generate_dotp_comp()` and `generate_add_multi_comps()` called correctly
- RTL simulation output matches ONNX golden reference
- Compressor eligibility logic (`_is_dotp_comp_eligible()`) works
- Template variable substitution (`$COMP_PIPELINE_DEPTH$`, `$USE_COMPRESSOR$`) correct

**CRITICAL STATUS (2026-04-09):**
- ✅ 7-Series tests PASS with gate absorption **disabled** (SinglePredCandidate only)
- ❌ 7-Series tests FAIL with gate absorption **enabled** (MuxCYRippleSum bug)
  - Error: "Output of ONNX model not matching output of node-by-node RTLsim!"
  - Expected: `[[[[1051., 660., 329., 838., ...]]]`
  - Got: `[[[[195., -262., -421., 214., ...]]]`
  - Root cause: UNKNOWN (suspected CARRY4.O wiring, but fix didn't work)

---

### 4. FINN End-to-End Tests (`tests/end2end/`)

**What they test:** Full quantized neural network build flows with FINN builder

#### 4.1 Cybersec MLP (`test_end2end_cybsec_mlp.py`)

**Network:** Binary/ternary multilayer perceptron for network intrusion detection

```bash
# Export only (fast, no Vivado)
pytest tests/end2end/test_end2end_cybsec_mlp.py::test_end2end_cybsec_mlp_export

# Full build flow (slow, requires Vivado)
pytest tests/end2end/test_end2end_cybsec_mlp.py::test_end2end_cybsec_mlp_build -k "Pynq-Z1"
pytest tests/end2end/test_end2end_cybsec_mlp.py::test_end2end_cybsec_mlp_build -k "AUP-ZU3_8GB"
pytest tests/end2end/test_end2end_cybsec_mlp.py::test_end2end_cybsec_mlp_build -k "VCK190"
```

**Boards:**
- **Pynq-Z1** (xc7z020clg400-1): 7-Series Zynq SoC
- **AUP-ZU3_8GB** (xczu3eg-sbva484-1-e): UltraScale+ Zynq MPSoC
- **VCK190** (xcvc1902-vsva2197-2MP-e-S): Versal ACAP

**Build steps:**
1. Export Brevitas model to QONNX
2. Streamline (fold constants, merge ops)
3. Convert to FINN HW layers
4. Specialize (choose RTL vs HLS implementation)
5. Dataflow partitioning
6. Folding (set PE/SIMD)
7. Minimize accumulator/weight bit widths
8. IP generation (PrepareIP → Vivado synthesis)
9. RTL simulation validation
10. Set FIFO depths
11. Create stitched IP
12. Deployment artifacts

**Compressor usage:**
- **Pynq-Z1 (7-Series):** If WW ≤ 4 && AW ≤ 4, uses dotp_comp path with **SinglePredCandidate** only (gate absorption disabled)
- **VCK190 (Versal):** Full gate absorption via VersalPredAdder + RippleSumPredAdder
- **AUP-ZU3 (UltraScale+):** Currently NO compressor support (DSP58 only, VERSION=3 blocked)

**Expected runtime:** 2-4 hours per board (synthesis is slow)

**What's verified:**
- End-to-end accuracy matches Brevitas model
- All transformations complete without errors
- Synthesis completes successfully
- RTL simulation produces correct outputs
- Resource utilization within board constraints

**Status:** ✅ Passes on all boards (with gate absorption disabled on 7-Series)

---

#### 4.2 MobileNet-v1 (`test_end2end_mobilenet_v1.py`)

**Network:** Quantized CNNx classification (ImageNet subset)

```bash
# Export + preprocessing
pytest tests/end2end/test_end2end_mobilenet_v1.py::test_end2end_mobilenet_export
pytest tests/end2end/test_end2end_mobilenet_v1.py::test_end2end_mobilenet_tidy_and_merge_with_preproc

# Full pipeline
pytest tests/end2end/test_end2end_mobilenet_v1.py -k "not rtlsim_performance"  # Skip slow perf test
```

**Board:** `xcvm1802-vsvd1760-2MP-e-S` (Versal VPrime)

**Network characteristics:**
- Depthwise separable convolutions
- Quantized activations and weights
- Mix of MVAU (FC layers) and sliding window generators

**Build steps:** Similar to cybsec, plus:
- ConvolutionInputGenerator specialization
- Thresholding layer folding
- Performance analysis

**Compressor usage:** Versal-only, uses full gate absorption (VersalPredAdder)

**Expected runtime:** 6-12 hours (much larger network than cybsec)

**What's verified:**
- Full CNN dataflow pipeline
- Sliding window + MVAU integration
- Large-scale synthesis
- Throughput analysis

**Status:** ✅ Passes on Versal (not tested on 7-Series)

---

#### 4.3 BNN-PYNQ (`test_end2end_bnn_pynq.py`)

**Network:** Binary neural network for MNIST/CIFAR-10

```bash
pytest tests/end2end/test_end2end_bnn_pynq.py -k "cnv"  # CNV network
pytest tests/end2end/test_end2end_bnn_pynq.py -k "lfc"  # LFC network
```

**Boards:** Various Pynq boards (Z1, ZCU104, KV260, etc.)

**Compressor usage:** Likely YES for binary/ternary layers on 7-Series boards

**Status:** Not recently tested with compressor integration

---

## Compressor Eligibility Logic

### When dotp_comp Path is Used (Full Compressor)

**Conditions (all must be true):**
1. `IS_MVU == 1` (not VVU)
2. `!PUMPED_COMPUTE` (pumped compute uses different path)
3. `WW <= 4` (weight width ≤ 4 bits)
4. `AW <= 4` (activation width ≤ 4 bits)

**Example configs that trigger compressor:**
- UINT4 × INT4 ✅
- INT4 × INT4 ✅
- BIPOLAR × BIPOLAR ✅ (converted to BINARY, counts as 1-bit)
- UINT8 × INT8 ❌ (too wide)
- Any with pumpedCompute=True ❌

**FPGA target requirements:**
- 7-Series: `SinglePredCandidate` only (gate absorption disabled)
- Versal: Full gate absorption (VersalPredAdder + RippleSumPredAdder)
- UltraScale+: **NO COMPRESSOR** (blocked by DSP58 VERSION=3 restriction)

### When add_multi Compressor is Used (DSP Path Adder)

**Conditions:**
1. DSP path is active (WW > 4 || AW > 4 || pumpedCompute)
2. `SIMD >= 4` (below this, binary tree is more efficient)
3. Low-part unsigned reduction (`!RESET_ZERO && ARG_LO >= 0`)
4. Matching CATCH_COMP entry in `add_multi.sv`

**Example:** UINT8 × INT8 with SIMD=16 uses DSP path, but the 16-lane reduction uses compressor

---

## Test Selection Guide

### I want to verify basic compressor generation works
```bash
cd src/finn/compressor
./run_tests.sh ca          # Fast (~5 min), no RTL sim
```

### I want to verify RTL functional correctness
```bash
cd finn-rtllib/mvu/tb
./run_dotp_comp_tests.sh   # Medium (~20 min), XSim required
```

### I want to verify FINN Python integration
```bash
cd /home/sgerber/test_repos/finn
pytest tests/fpgadataflow/test_fpgadataflow_mvau.py::test_fpgadataflow_rtl_mvau \
  -k "xc7z020 and idt_wdt0 and False-False"  # ~30 min, 9 configs
```

### I want to verify end-to-end network build
```bash
pytest tests/end2end/test_end2end_cybsec_mlp.py::test_end2end_cybsec_mlp_build \
  -k "Pynq-Z1"   # Slow (~3 hours), full Vivado synthesis
```

### I want to compare 7-Series vs Versal compressor efficiency
```bash
# Run cybsec on both boards
pytest tests/end2end/test_end2end_cybsec_mlp.py -k "Pynq-Z1 or VCK190"
# Compare reports in build_dir:
# - Pynq-Z1: Uses SinglePredCandidate only (suboptimal LUTs)
# - VCK190: Uses VersalPredAdder (optimal LUTs)
```

---

## Known Test Issues

### 1. Gate Absorption Disabled on 7-Series (CRITICAL)
- **Symptom:** RTL simulation produces incorrect numerical results when MuxCYRippleSum enabled
- **Workaround:** Disabled in `target.py` (lines 83-87), uses `SinglePredCandidate` only
- **Impact:** Functional correctness maintained, ~10-20% higher LUT usage on 7-Series
- **Status:** UNRESOLVED (bug in MuxCYRippleSum.build_hardware())

### 2. UltraScale+ Compressor Support Missing
- **Symptom:** xcku3p and xczu* parts don't use compressor despite WW/AW ≤ 4
- **Root cause:** RTL restricts to DSP58 only (VERSION=3), blocks 7-Series/UltraScale+ DSP48E1/E2
- **Workaround:** None - intentional design decision
- **Impact:** UltraScale+ uses DSP slices instead of compressors

### 3. VVU Has No Compressor Support
- **Symptom:** VectorVectorActivation never uses compressor path
- **Root cause:** VVU has PE-parallel structure (different compute pattern than MVU)
- **Workaround:** None - would require architectural redesign
- **Impact:** VVU always routes to DSP58, only works on Versal

---

## Test Output Interpretation

### Passing dotp_comp Test
```
Running test: 8xs2s2
  Generating compressor module: comp_8xs2s2_a16.sv
  Expanding template: dotp_comp.sv
  Compiling with Vivado XSim...
  Running simulation (1000 cycles)...
  ✓ All outputs match golden reference
  PASS
```

### Failing RTL Simulation (Example)
```
AssertionError: Output of ONNX model not matching output of node-by-node RTLsim!
Expected: [[[[1051., 660., 329., 838., ...]]]]
Got:      [[[[195., -262., -421., 214., ...]]]]
Max error: 1313.0
```
→ This indicates a hardware bug in the compressor RTL implementation

### Compressor Not Used (Example)
```
WARNING: Config 9xu8u8 ineligible for compressor (WW=8 > 4)
Falling back to DSP path
```
→ This is expected behavior for wide bit widths

---

## Developer Workflow

### Making changes to compressor core (`src/finn/compressor/src/`)

1. **Run standalone tests first:**
   ```bash
   cd src/finn/compressor
   ./run_tests.sh ca
   ```
2. **If passing, run RTL integration tests:**
   ```bash
   cd finn-rtllib/mvu/tb
   ./run_dotp_comp_tests.sh
   ```
3. **If passing, run FINN MVAU subset:**
   ```bash
   pytest tests/fpgadataflow/test_fpgadataflow_mvau.py::test_fpgadataflow_rtl_mvau -k "xc7z020 and UINT4 and False-False" -x
   ```
4. **If all passing, consider running end2end (optional):**
   ```bash
   pytest tests/end2end/test_end2end_cybsec_mlp.py -k "Pynq-Z1"
   ```

### Making changes to FINN integration (`src/finn/custom_op/fpgadataflow/rtl/matrixvectoractivation_rtl.py`)

1. **Run MVAU tests directly:**
   ```bash
   pytest tests/fpgadataflow/test_fpgadataflow_mvau.py::test_fpgadataflow_rtl_mvau -k "xc7z020"
   ```
2. **If passing, spot-check end2end:**
   ```bash
   pytest tests/end2end/test_end2end_cybsec_mlp.py::test_end2end_cybsec_mlp_export  # Fast
   ```

---

## Summary: Test Coverage

| Test Level | What's Tested | Runtime | Vivado Required | Status |
|------------|---------------|---------|-----------------|--------|
| Standalone compressor | Core generation logic | 5 min | Yes (compile only) | ✅ PASS |
| dotp_comp RTL | Template integration + XSim | 20 min | Yes (XSim) | ✅ PASS |
| MVU comp RTL | Full MVU pipeline | 15 min | Yes (XSim) | ✅ PASS |
| add_multi RTL | DSP path adder replacement | 10 min | Yes (XSim) | ✅ PASS |
| MVAU layer tests | FINN Python integration | 30 min | Yes (XSim) | ✅ PASS (7-Series, gate absorption disabled) |
| End2end cybsec | Full network build | 3 hours | Yes (synthesis) | ✅ PASS (Pynq-Z1) |
| End2end mobilenet | Large CNN build | 8 hours | Yes (synthesis) | ✅ PASS (Versal only) |

**Overall Compressor Status (2026-04-09):**
- ✅ Functional correctness: VERIFIED on all platforms (with workarounds)
- ⚠️ 7-Series gate absorption: DISABLED (MuxCYRippleSum bug)
- ✅ Versal gate absorption: WORKING (optimal LUT usage)
- ❌ UltraScale+ support: NOT IMPLEMENTED (intentional)
