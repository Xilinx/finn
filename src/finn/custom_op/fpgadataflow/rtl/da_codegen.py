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

"""Constant-weight compute cores for the RTL MVAU via distributed arithmetic.

The functions in this module turn a constant weight matrix into a pipelined,
multiplier-free adder graph using the ``alkaid`` package (formerly ``da4ml``,
Sun et al., "da4ml: Distributed Arithmetic for Real-time Neural Networks on
FPGAs", TRETS 2026) and wrap the generated Verilog so that it can be dropped
into FINN's ``MVAU_rtl`` with ``mem_mode="internal_embedded"``.

Everything here is independent of ONNX and FINN node objects so that it can be
unit tested without Vivado. The only FINN dependencies are the ``DataType``
class (qonnx); configuration errors are raised as ``ValueError``.

Bus layout between FINN and the generated core (see also ``mvu_da_axi_wrapper.v``):

* FINN presents one input vector per beat, element ``i`` at bits
  ``[i*ACTIVATION_WIDTH +: ACTIVATION_WIDTH]``. BIPOLAR activations travel as
  single bits (0 for -1, 1 for +1), exactly as for the other MVAU backends.
* The alkaid io wrapper exposes the same layout with per-element slots of
  uniform width (``inp_slot_width``, asserted to equal ``ACTIVATION_WIDTH``).
* The alkaid output bus holds ``MH`` slots of ``out_slot_width`` bits. Slot
  ``pe`` carries ``y[pe] * 2**out_frac`` in two's complement if any output can
  be negative (``out_signed``), unsigned otherwise. ``out_frac`` is <= 0 (it is
  negative when every output is a multiple of a power of two), so the wrapper
  sign/zero-extends each slot to ``ACCU_WIDTH`` and shifts it left by
  ``out_shift_left = -out_frac``.
"""

import hashlib
import numpy as np
import os
import re
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qonnx.core.datatype import BaseDataType

# Names of the shared alkaid Verilog primitives that the generated adder graph
# may instantiate. Every one of them is copied next to the generated core with a
# node-specific prefix so that several DA cores can coexist in one Vivado project.
DA_PRIMITIVE_MODULES = (
    "shift_adder",
    "negative",
    "ternary_adder",
    "mux",
    "binop",
    "multiplier",
)

# Weights are handled as float32 inside alkaid; integers with more bits than the
# float32 mantissa would be rounded.
DA_MAX_WEIGHT_BITS = 24

# Mapping from the target clock period to alkaid's per-stage latency budget.
# alkaid's latency unit is a surrogate delay (~1.1 units per adder level plus a
# small width-dependent carry term). Calibration (CI sweep microbenchmark_da,
# xczu28dr-2, 10 ns, 16..64 x 16..64, UINT4/INT8/BIPOLAR activations, INT4/INT8/
# BIPOLAR weights): with 0.85 units per ns every design met timing, with a slack
# of 0.2 ns for 64x64 INT8xINT8 up to 6.4 ns for the small 4-bit layers.
DA_LATENCY_CUTOFF_PER_NS = 0.85
DA_MIN_LATENCY_CUTOFF = 1.0

# LUT-per-cost-unit factor for resource estimation. alkaid's ``cost`` is the
# number of active result bits of all adders in the graph. Calibration on the
# same sweep: LUT(stitched node) = 0.99 * cost + 1.93 * output bits + 302 (9 %
# mean error over 28 designs); a hierarchical synthesis attributes 1.03 LUT per
# cost unit to the core and one LUT per output bit to the SRL output queue of the
# bracket, the rest is the TLastMarker and stitching overhead outside the node.
DA_LUT_PER_COST_BIT = 1.0


@dataclass
class DACore:
    """Result of :func:`build_da_core`: generated Verilog plus interface facts."""

    top_name: str
    """Name of the pipelined alkaid core module (registered input and output)."""
    wrapper_name: str
    """Name of the alkaid io wrapper around ``top_name`` exposing uniform slots."""
    files: dict[str, str]
    """Mapping from file name (``<module>.v``) to Verilog text."""
    latency_cycles: int
    """Cycles from presenting ``model_inp`` to a valid ``model_out`` (registers on the path)."""
    n_stages: int
    """Number of combinational pipeline stages."""
    mw: int
    mh: int
    inp_slot_width: int
    core_inp_width: int
    out_slot_width: int
    out_signed: bool
    out_shift_left: int
    core_out_width: int
    cost: float
    """alkaid's cost estimate (active adder result bits) for the graph."""
    adders: int
    """Number of two- and three-input adders in the graph (incl. constant adds)."""
    depth: float
    """Maximum surrogate latency of the combinational graph before pipelining."""
    latency_cutoff: float
    """Per-stage surrogate latency budget that was used for pipelining."""
    golden: Callable[[np.ndarray], np.ndarray] = field(repr=False)
    """Bit-exact reference: takes FINN-encoded inputs of shape (N, MW), returns (N, MH)."""
    comb: object = field(repr=False, default=None)
    """The alkaid ``CombLogic`` program (kept for inspection, not serializable)."""


def _require(cond: bool, msg: str) -> None:
    """Raise a ValueError with ``msg`` unless ``cond`` holds."""
    if not cond:
        raise ValueError(msg)


def _import_alkaid():
    """Import the alkaid package lazily so that FINN itself does not depend on it
    at import time and a missing wheel yields an actionable error."""
    try:
        from alkaid.codegen.rtl.verilog import (  # noqa: PLC0415
            fsm_logic_gen,
            generate_io_wrapper,
        )
        from alkaid.trace import FVArray, HWConfig, to_pipeline, trace  # noqa: PLC0415
        from alkaid.trace.passes import (  # noqa: PLC0415
            dead_code_elimin,
            fuse_ternary_adders,
        )
        from alkaid.trace.passes.surrogate import add_surrogate  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ValueError(
            "The distributed-arithmetic MVAU requires the 'alkaid' package "
            "(formerly 'da4ml'), which could not be imported. Install it into the "
            f"FINN environment (pip install alkaid) to use mem_mode=internal_embedded "
            f"on MVAU_rtl. Original error: {e}"
        ) from e
    return {
        "FVArray": FVArray,
        "HWConfig": HWConfig,
        "trace": trace,
        "to_pipeline": to_pipeline,
        "fsm_logic_gen": fsm_logic_gen,
        "generate_io_wrapper": generate_io_wrapper,
        "dead_code_elimin": dead_code_elimin,
        "fuse_ternary_adders": fuse_ternary_adders,
        "add_surrogate": add_surrogate,
    }


def _primitive_source_dir() -> str:
    """Directory of the Verilog primitives shipped with the installed alkaid package."""
    import alkaid.codegen.rtl.verilog as vpkg  # noqa: PLC0415

    return os.path.join(os.path.dirname(vpkg.__file__), "source")


def latency_cutoff_from_clk(clk_ns: float) -> float:
    """Return alkaid's per-stage latency budget for a target clock period in ns."""
    _require(clk_ns > 0, f"Target clock period must be positive, got {clk_ns} ns.")
    return max(DA_MIN_LATENCY_CUTOFF, DA_LATENCY_CUTOFF_PER_NS * float(clk_ns))


def prefix_module_names(text: str, prefix: str, names: Iterable[str]) -> str:
    """Rename every whole-word occurrence of the module names in ``names`` by
    prepending ``prefix``. Used to make alkaid's shared primitive and stage
    module names unique per node."""
    for name in sorted(set(names), key=len, reverse=True):
        if name.startswith(prefix):
            continue
        text = re.sub(rf"(?<![\w$]){re.escape(name)}(?![\w$])", prefix + name, text)
    return text


def module_names_in(text: str) -> list[str]:
    """Return the names of all modules declared in a Verilog source text."""
    return re.findall(r"^\s*module\s+([A-Za-z_][\w$]*)", text, flags=re.MULTILINE)


def _input_kif(idt: "BaseDataType", mw: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return alkaid (k, i, f) arrays describing the FINN-encoded input elements."""
    if idt.get_canonical_name() == "BIPOLAR":
        k, i = 0, 1
    else:
        k = 1 if idt.signed() else 0
        i = idt.bitwidth() - k
    return np.full(mw, k), np.full(mw, i), np.zeros(mw, dtype=int)


# Adder graphs by problem digest. One build solves every layer several times
# (resource estimate, IP generation, the isolated copies of the FIFO sizing) and
# the solver time grows about cubically with the layer size (seconds for 64x64,
# 20 min for a dense 256x256 INT3 matrix), so the combinational solution is kept
# for the lifetime of the process. Pipelining and code generation depend on the
# clock target and the node name and are cheap, they are redone on every call.
DA_SOLUTION_CACHE_SIZE = 32
_solution_cache: dict[str, object] = {}
_solution_cache_lock = threading.Lock()


def clear_da_solution_cache() -> None:
    """Forget all cached adder graphs."""
    with _solution_cache_lock:
        _solution_cache.clear()


def _solve_adder_graph(
    W: np.ndarray,
    idt: "BaseDataType",
    hard_dc: int,
    ternary_fuse: bool,
    search_all_decompose_dc: bool,
):
    """Run the alkaid solver on ``y = x @ W`` for FINN-encoded inputs of type
    ``idt`` and return the combinational adder graph (alkaid ``CombLogic``)."""
    ak = _import_alkaid()
    mw = W.shape[0]
    # FINN encodes BIPOLAR activations as a single bit b (0 -> -1, 1 -> +1).
    # Substitute x = 2b - 1: y = x @ W = b @ (2W) - sum(W, axis=0).
    k, i, f = _input_kif(idt, mw)
    if idt.get_canonical_name() == "BIPOLAR":
        matrix = 2.0 * W.astype(np.float64)
        bias = -W.astype(np.float64).sum(axis=0)
    else:
        matrix = W.astype(np.float64)
        bias = None

    digest = hashlib.sha256()
    for part in (
        repr((matrix.shape, int(k[0]), int(i[0]), int(hard_dc), bool(ternary_fuse))),
        repr((bool(search_all_decompose_dc), bias is not None)),
    ):
        digest.update(part.encode())
    digest.update(np.ascontiguousarray(matrix).tobytes())
    key = digest.hexdigest()
    with _solution_cache_lock:
        comb = _solution_cache.get(key)
    if comb is not None:
        return comb

    solver_options = {
        "hard_dc": int(hard_dc),
        "search_all_decompose_dc": bool(search_all_decompose_dc),
    }
    hwconf = ak["HWConfig"](1, 1, -1)
    inp = ak["FVArray"].from_kif(k, i, f, hwconf, 0.0, solver_options)
    out = inp @ matrix.astype(np.float32)
    if bias is not None:
        out = out + bias
    comb = ak["trace"](inp, out)
    if ternary_fuse:
        comb = ak["add_surrogate"](
            ak["dead_code_elimin"](ak["fuse_ternary_adders"](comb)), _skip_op8_cost=True
        )
    with _solution_cache_lock:
        while len(_solution_cache) >= DA_SOLUTION_CACHE_SIZE:
            _solution_cache.pop(next(iter(_solution_cache)))
        _solution_cache[key] = comb
    return comb


def _padded_precision(precisions) -> tuple[bool, int, int]:
    """Element-wise maximum of (signed, integer bits, fractional bits), mirroring
    alkaid's io wrapper slot layout."""
    arr = np.array([(int(p[0]), int(p[1]), int(p[2])) for p in precisions], dtype=int)
    k, i, f = arr.max(axis=0)
    return bool(k), int(i), int(f)


def pack_input_beat(x: np.ndarray, slot_width: int) -> int:
    """Pack one FINN-encoded input vector into the core's ``model_inp`` word
    (element 0 at the LSB, two's complement per slot)."""
    mask = (1 << slot_width) - 1
    word = 0
    for idx, val in enumerate(np.asarray(x).ravel()):
        word |= (int(val) & mask) << (idx * slot_width)
    return word


def unpack_output_beat(
    word: int, mh: int, slot_width: int, signed: bool, shift_left: int
) -> np.ndarray:
    """Unpack the core's ``model_out`` word into ``mh`` integer outputs, applying
    the sign interpretation and the left shift of the FINN-side glue."""
    out = np.zeros(mh, dtype=np.int64)
    mask = (1 << slot_width) - 1
    for pe in range(mh):
        v = (word >> (pe * slot_width)) & mask
        if signed and slot_width > 0 and (v >> (slot_width - 1)) & 1:
            v -= 1 << slot_width
        out[pe] = v << shift_left
    return out


def build_da_core(
    W: np.ndarray,
    idt: "BaseDataType",
    wdt: "BaseDataType",
    odt: "BaseDataType",
    top_name: str,
    hard_dc: int = 2,
    latency_cutoff: float = 0.0,
    clk_ns: float | None = None,
    ternary_fuse: bool = True,
    search_all_decompose_dc: bool = True,
) -> DACore:
    """Compile the constant matrix ``W`` (shape ``(MW, MH)``, FINN initializer
    layout, ``y = x @ W``) into a pipelined adder graph.

    Args:
        W: weight matrix as stored in the ONNX initializer (BIPOLAR weights as +-1).
        idt: FINN datatype of the activations (integer types incl. BIPOLAR/BINARY).
        wdt: FINN datatype of the weights (integer types, at most 24 bits).
        odt: FINN datatype of the accumulator/output stream (``ACCU_WIDTH``).
        top_name: node-unique base name for all generated modules.
        hard_dc: alkaid depth constraint for the adder trees (-1: unconstrained).
        latency_cutoff: per-stage surrogate latency budget; 0 derives it from ``clk_ns``.
        clk_ns: target clock period, used when ``latency_cutoff`` is 0.
        ternary_fuse: fuse adder pairs into three-input adders (alkaid default flow).
        search_all_decompose_dc: alkaid solver option (exhaustive decomposition search).
    """
    ak = _import_alkaid()

    W = np.asarray(W)
    _require(W.ndim == 2, f"{top_name}: weight matrix must be 2-dimensional, got shape {W.shape}.")
    mw, mh = W.shape
    _require(mw > 0 and mh > 0, f"{top_name}: empty weight matrix {W.shape}.")
    _require(
        idt.is_integer() and wdt.is_integer() and odt.is_integer(),
        f"{top_name}: the distributed-arithmetic MVAU supports integer datatypes only "
        f"(got idt={idt.name}, wdt={wdt.name}, odt={odt.name}).",
    )
    _require(
        wdt.bitwidth() <= DA_MAX_WEIGHT_BITS,
        f"{top_name}: weights wider than {DA_MAX_WEIGHT_BITS} bits are not supported "
        f"(got {wdt.name}).",
    )
    _require(
        np.all(np.isfinite(W)) and np.array_equal(W, np.round(W)),
        f"{top_name}: weight matrix contains non-integer values.",
    )
    _require(
        bool(np.vectorize(wdt.allowed)(W).all()),
        f"{top_name}: weight matrix contains values outside of {wdt.name}.",
    )
    _require(
        bool(np.any(W != 0)),
        f"{top_name}: the weight matrix is all-zero; a constant-weight compute core "
        "for it would have no inputs and no outputs. Remove the layer instead.",
    )
    _require(
        not (hard_dc == 0),
        f"{top_name}: da_hard_dc must be -1 (unconstrained) or a positive depth.",
    )

    if latency_cutoff <= 0:
        _require(
            clk_ns is not None,
            f"{top_name}: either latency_cutoff or clk_ns must be given.",
        )
        latency_cutoff = latency_cutoff_from_clk(clk_ns)

    comb = _solve_adder_graph(W, idt, hard_dc, ternary_fuse, search_all_decompose_dc)

    opcodes = {op.opcode for op in comb.ops}
    _require(
        7 not in opcodes,
        f"{top_name}: alkaid emitted a variable multiplier, which must not happen "
        "for constant weights.",
    )

    # --- pipelining and codegen ---------------------------------------------
    fsm = ak["to_pipeline"](comb, latency_cutoff=float(latency_cutoff), verbose=False)
    codes = ak["fsm_logic_gen"](fsm, top_name, print_latency=False)
    wrapper_name = f"{top_name}_wrapper"
    codes[wrapper_name] = ak["generate_io_wrapper"](fsm, top_name)

    inp_sig = fsm.inp_signals
    out_sig = fsm.out_signals
    _require(
        len(inp_sig) == 1 and len(out_sig) == 1,
        f"{top_name}: unexpected alkaid FSM interface {inp_sig}/{out_sig}.",
    )
    inp_sig, out_sig = inp_sig[0], out_sig[0]
    assert inp_sig.size == mw and out_sig.size == mh, "alkaid changed the vector sizes"
    assert out_sig.schedule is not None
    latency_cycles = int(out_sig.schedule.bias)
    n_stages = len(fsm.logic)

    ki, ii, fi = _padded_precision(inp_sig.precisions)
    inp_slot_width = int(ki) + ii + fi
    act_width = idt.bitwidth()
    _require(
        inp_slot_width == act_width,
        f"{top_name}: input slot width {inp_slot_width} does not match the activation "
        f"width {act_width} of {idt.name}; this indicates a datatype mapping bug.",
    )
    ko, io, fo = _padded_precision(out_sig.precisions)
    _require(
        fo <= 0,
        f"{top_name}: alkaid produced fractional output bits (f={fo}) for an integer "
        "matrix; this indicates a datatype mapping bug.",
    )
    out_slot_width = int(ko) + io + fo
    out_shift_left = -fo
    accu_width = odt.bitwidth()
    _require(
        not (ko and not odt.signed()),
        f"{top_name}: outputs can be negative but the accumulator datatype {odt.name} "
        "is unsigned.",
    )
    needed_width = io + max(int(ko), int(odt.signed()))
    _require(
        needed_width <= accu_width,
        f"{top_name}: the exact output range needs {needed_width} bits but the "
        f"accumulator datatype is {odt.name} ({accu_width} bits). Run "
        "MinimizeAccumulatorWidth or check the datatype annotations.",
    )

    # --- make all module names node-unique ----------------------------------
    used_primitives = [
        p
        for p in DA_PRIMITIVE_MODULES
        if re.search(rf"(?<![\w$]){p}\s*#\s*\(", "\n".join(codes.values()))
    ]
    src_dir = _primitive_source_dir()
    for p in used_primitives:
        with open(f"{src_dir}/{p}.v") as fh:
            codes[p] = fh.read()
    prefix = f"{top_name}_"
    to_rename = [
        n for text in codes.values() for n in module_names_in(text) if not n.startswith(top_name)
    ]
    files = {}
    for name, text in codes.items():
        new_text = prefix_module_names(text, prefix, to_rename)
        new_name = name if name.startswith(top_name) else prefix + name
        files[f"{new_name}.v"] = new_text
    for text in files.values():
        for n in module_names_in(text):
            assert n.startswith(top_name), f"module {n} escaped prefixing"

    n_adders = sum(op.opcode in (0, 1, 4, 11) for op in comb.ops)

    def golden(x: np.ndarray) -> np.ndarray:
        """Bit-exact reference of the graph for FINN-encoded inputs of shape (N, MW)."""
        x = np.asarray(x, dtype=np.float64).reshape(-1, mw)
        return np.asarray(comb.predict(x), dtype=np.float64).reshape(-1, mh)

    return DACore(
        top_name=top_name,
        wrapper_name=wrapper_name,
        files=files,
        latency_cycles=latency_cycles,
        n_stages=n_stages,
        mw=mw,
        mh=mh,
        inp_slot_width=inp_slot_width,
        core_inp_width=mw * inp_slot_width,
        out_slot_width=out_slot_width,
        out_signed=bool(ko),
        out_shift_left=out_shift_left,
        core_out_width=mh * out_slot_width,
        cost=float(comb.cost),
        adders=int(n_adders),
        depth=float(comb.latency[1]),
        latency_cutoff=float(latency_cutoff),
        golden=golden,
        comb=comb,
    )


def da_cost_estimate(
    W: np.ndarray, idt: "BaseDataType", hard_dc: int = 2, ternary_fuse: bool = True
) -> dict:
    """Run only the alkaid solver (no Verilog) and return the cost figures.
    Used by the microbenchmark DUT to log the DA numbers next to HLS builds."""
    comb = _solve_adder_graph(np.asarray(W), idt, hard_dc, ternary_fuse, True)
    return {
        "da_cost": float(comb.cost),
        "da_adders": int(sum(op.opcode in (0, 1, 4, 11) for op in comb.ops)),
        "da_depth": float(comb.latency[1]),
        "da_ops": int(len(comb.ops)),
    }
