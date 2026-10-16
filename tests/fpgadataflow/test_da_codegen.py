# Copyright (C) 2026, Paderborn University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Unit tests for the distributed-arithmetic core generator (no Vivado needed)."""

import pytest

import numpy as np
import re
from qonnx.core.datatype import DataType
from qonnx.util.basic import calculate_matvec_accumulator_range, gen_finn_dt_tensor

from finn.custom_op.fpgadataflow.rtl.da_codegen import (
    DA_PRIMITIVE_MODULES,
    build_da_core,
    clear_da_solution_cache,
    da_cost_estimate,
    latency_cutoff_from_clk,
    module_names_in,
    pack_input_beat,
    prefix_module_names,
    unpack_output_beat,
)


def accumulator_datatype(W, idt):
    """Smallest FINN datatype containing the exact accumulator range (like
    MinimizeAccumulatorWidth)."""
    acc_min, acc_max = calculate_matvec_accumulator_range(W, idt)
    if acc_min >= 0:
        return DataType.get_smallest_possible(acc_max)
    dt_min = DataType.get_smallest_possible(acc_min)
    dt_max = DataType.get_smallest_possible(acc_max)
    bw = max(dt_min.bitwidth(), dt_max.bitwidth())
    if not dt_max.signed():
        # e.g. UINT4 vs INT3: need the signed type that also holds acc_max
        bw = max(bw, dt_max.bitwidth() + 1)
    return DataType[f"INT{bw}"]


def encode_inputs(x, idt):
    """FINN stream encoding of activation values (BIPOLAR -> 0/1)."""
    if idt == DataType["BIPOLAR"]:
        return (x + 1) / 2
    return x


def structured_weights(wdt, mw, mh, seed=0):
    """Random weights with an all-zero row (dead input), an all-zero column
    (zero-width output) and an all-even column (negative output fraction)."""
    np.random.seed(seed)
    W = gen_finn_dt_tensor(wdt, (mw, mh))
    if wdt.allowed(0):
        W[3, :] = 0
        W[:, 5] = 0
    if wdt.bitwidth() > 1:
        even = 2 * np.round(W[:, 7] / 2)
        W[:, 7] = np.clip(even, wdt.min() + (wdt.min() % 2), wdt.max() - (wdt.max() % 2))
    return W


DTYPE_CASES = [
    (DataType["UINT4"], DataType["INT4"]),
    (DataType["INT8"], DataType["INT8"]),
    (DataType["UINT4"], DataType["BIPOLAR"]),
    (DataType["BIPOLAR"], DataType["INT4"]),
    (DataType["INT4"], DataType["TERNARY"]),
    (DataType["BINARY"], DataType["INT4"]),
    (DataType["UINT2"], DataType["UINT3"]),
]


@pytest.mark.parametrize("idt,wdt", DTYPE_CASES, ids=[f"{i.name}x{w.name}" for i, w in DTYPE_CASES])
@pytest.mark.parametrize("hard_dc", [-1, 2])
@pytest.mark.fpgadataflow
def test_da_codegen_golden_and_interface(idt, wdt, hard_dc):
    mw, mh = 16, 16
    W = structured_weights(wdt, mw, mh)
    odt = accumulator_datatype(W, idt)
    core = build_da_core(W, idt, wdt, odt, "MVAU_rtl_0_da_core", hard_dc=hard_dc, clk_ns=5.0)

    # bit-exact golden model on FINN-encoded inputs equals x @ W on decoded values
    x = gen_finn_dt_tensor(idt, (64, mw))
    x[0, :] = idt.min()
    x[1, :] = idt.max()
    y_exp = x @ W
    y_got = core.golden(encode_inputs(x, idt))
    assert np.array_equal(y_got, y_exp)

    # interface facts
    assert core.inp_slot_width == idt.bitwidth()
    assert core.core_inp_width == mw * idt.bitwidth()
    assert core.core_out_width == mh * core.out_slot_width
    assert core.latency_cycles == core.n_stages + 1
    assert core.out_shift_left >= 0
    assert core.out_slot_width + core.out_shift_left <= odt.bitwidth()
    if (y_exp < 0).any():
        assert core.out_signed
    assert core.cost > 0 and core.adders > 0 and core.depth > 0

    # the FINN-side glue, modelled in Python: sign/zero-extend and shift each slot
    for xi, yi in zip(encode_inputs(x, idt)[:8], y_exp[:8]):
        word = 0
        for pe in range(mh):
            v = int(yi[pe]) >> core.out_shift_left
            assert v << core.out_shift_left == int(yi[pe]), "output not a multiple of 2^shift"
            word |= (v & ((1 << core.out_slot_width) - 1)) << (pe * core.out_slot_width)
        got = unpack_output_beat(
            word, mh, core.out_slot_width, core.out_signed, core.out_shift_left
        )
        assert np.array_equal(got, yi)
        assert pack_input_beat(xi, core.inp_slot_width) < (1 << core.core_inp_width)

    # generated Verilog: every module is node-unique, primitives are prefixed copies
    all_text = "\n".join(core.files.values())
    for fname, text in core.files.items():
        assert fname.startswith(core.top_name) and fname.endswith(".v")
        for mod in module_names_in(text):
            assert mod.startswith(core.top_name), mod
    for prim in DA_PRIMITIVE_MODULES:
        assert not re.search(rf"(?<![\w$]){prim}\s*#\s*\(", all_text), f"unprefixed {prim}"
        if re.search(rf"(?<![\w$]){core.top_name}_{prim}\s*#\s*\(", all_text):
            assert f"{core.top_name}_{prim}.v" in core.files
    assert "multiplier" not in all_text.replace(f"{core.top_name}_multiplier", "")
    assert f"{core.top_name}.v" in core.files
    assert f"{core.wrapper_name}.v" in core.files
    assert f"module {core.wrapper_name} (" in core.files[f"{core.wrapper_name}.v"]
    assert f"[{core.core_inp_width - 1}:0] model_inp" in core.files[f"{core.wrapper_name}.v"]
    assert f"[{core.core_out_width - 1}:0] model_out" in core.files[f"{core.wrapper_name}.v"]


@pytest.mark.fpgadataflow
def test_da_codegen_deterministic():
    W = structured_weights(DataType["INT4"], 32, 32, seed=3)
    odt = accumulator_datatype(W, DataType["UINT4"])
    a = build_da_core(W, DataType["UINT4"], DataType["INT4"], odt, "n_da_core", clk_ns=4.0)
    # solve again from scratch, not from the solution cache
    clear_da_solution_cache()
    b = build_da_core(W, DataType["UINT4"], DataType["INT4"], odt, "n_da_core", clk_ns=4.0)
    assert a.comb is not b.comb
    assert a.files == b.files
    assert (a.cost, a.adders, a.depth, a.latency_cycles) == (
        b.cost,
        b.adders,
        b.depth,
        b.latency_cycles,
    )


@pytest.mark.fpgadataflow
def test_da_codegen_solution_cache():
    """The solver runs once per weight matrix and solver options; the node name
    and the clock target only affect pipelining and code generation."""
    clear_da_solution_cache()
    idt, wdt = DataType["UINT4"], DataType["INT4"]
    W = structured_weights(wdt, 32, 32, seed=5)
    odt = accumulator_datatype(W, idt)
    a = build_da_core(W, idt, wdt, odt, "a_da_core", clk_ns=4.0)
    b = build_da_core(W, idt, wdt, odt, "b_da_core", clk_ns=20.0)
    assert b.comb is a.comb
    assert da_cost_estimate(W, idt)["da_cost"] == a.cost
    assert all(name.startswith("b_da_core") for name in b.files)
    assert b.n_stages <= a.n_stages
    x = gen_finn_dt_tensor(idt, (8, 32))
    assert np.array_equal(b.golden(x), x @ W)
    # other solver options, input datatype or weights are different problems
    assert build_da_core(W, idt, wdt, odt, "c_da_core", clk_ns=4.0, hard_dc=-1).comb is not a.comb
    odt8 = accumulator_datatype(W, DataType["INT8"])
    assert build_da_core(W, DataType["INT8"], wdt, odt8, "d_da_core", clk_ns=4.0).comb is not a.comb
    W2 = W.copy()
    W2[0, 0] = W2[0, 0] + 1 if W2[0, 0] < wdt.max() else W2[0, 0] - 1
    odt2 = accumulator_datatype(W2, idt)
    assert build_da_core(W2, idt, wdt, odt2, "e_da_core", clk_ns=4.0).comb is not a.comb


@pytest.mark.fpgadataflow
def test_da_codegen_latency_cutoff_controls_stages():
    W = structured_weights(DataType["INT8"], 32, 32, seed=1)
    odt = accumulator_datatype(W, DataType["INT8"])
    fast = build_da_core(
        W, DataType["INT8"], DataType["INT8"], odt, "t_da_core", latency_cutoff=2.0
    )
    slow = build_da_core(
        W, DataType["INT8"], DataType["INT8"], odt, "t_da_core", latency_cutoff=20.0
    )
    assert fast.n_stages > slow.n_stages
    assert slow.n_stages == 1 and slow.latency_cycles == 2
    assert latency_cutoff_from_clk(5.0) > latency_cutoff_from_clk(2.0) >= 1.0


@pytest.mark.fpgadataflow
def test_da_codegen_rejections():
    W = structured_weights(DataType["INT4"], 16, 16)
    odt = accumulator_datatype(W, DataType["UINT4"])
    with pytest.raises(ValueError, match="all-zero"):
        build_da_core(np.zeros_like(W), DataType["UINT4"], DataType["INT4"], odt, "x", clk_ns=5)
    with pytest.raises(ValueError, match="outside"):
        build_da_core(W * 4, DataType["UINT4"], DataType["INT4"], odt, "x", clk_ns=5)
    with pytest.raises(ValueError, match="accumulator"):
        build_da_core(W, DataType["UINT4"], DataType["INT4"], DataType["INT4"], "x", clk_ns=5)
    with pytest.raises(ValueError, match="unsigned"):
        build_da_core(W, DataType["UINT4"], DataType["INT4"], DataType["UINT16"], "x", clk_ns=5)
    with pytest.raises(ValueError, match="integer"):
        build_da_core(W, DataType["FLOAT32"], DataType["INT4"], odt, "x", clk_ns=5)
    with pytest.raises(ValueError, match="clk_ns"):
        build_da_core(W, DataType["UINT4"], DataType["INT4"], odt, "x")


@pytest.mark.fpgadataflow
def test_da_cost_estimate_matches_build():
    W = structured_weights(DataType["INT4"], 16, 16)
    odt = accumulator_datatype(W, DataType["UINT4"])
    core = build_da_core(W, DataType["UINT4"], DataType["INT4"], odt, "c_da_core", clk_ns=5)
    est = da_cost_estimate(W, DataType["UINT4"])
    assert est["da_cost"] == core.cost
    assert est["da_adders"] == core.adders


@pytest.mark.fpgadataflow
def test_prefix_module_names():
    text = (
        "module logic0 (\nlogic0 _inst_logic0 (.a(a));\n"
        "shift_adder #(1) op_0 (x);\nINTERNAL_logic0_inp"
    )
    out = prefix_module_names(text, "top_", ["logic0", "shift_adder"])
    assert out == (
        "module top_logic0 (\ntop_logic0 _inst_logic0 (.a(a));\n"
        "top_shift_adder #(1) op_0 (x);\nINTERNAL_logic0_inp"
    )
    # already prefixed names are left alone
    assert prefix_module_names("top_logic0", "top_", ["top_logic0"]) == "top_logic0"
