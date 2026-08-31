# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Resource models for hardware custom ops.

So far this holds the cost/style model for finn-rtllib/fifo/hdl/fifo.sv, mirrored in
Python so the style decision and the estimators share one description of it; further
op resource models can be added here. W is the padded stream width.

Coefficients marked "fitted" come from out-of-context Vivado runs (2026.1, xczu7ev and
xcvc1902) over every style, both memory geometries, widths 1..256 and depths 2..262145;
everything else is exact, read off the RTL or a datasheet. Tile counts are exact, LUTs
land within a few percent.

Keep _resolve() in sync with fifo.sv:84-90;
tests/fpgadataflow/test_fifo_ram_style_mirror.py fails if it drifts from the RTL.
"""

import math
from collections import namedtuple
from functools import lru_cache

# fifo.sv's own RAM_STYLE_EFF threshold, not a cost comparison. Change it there first.
URAM_DEPTH_THRESHOLD = 2028

# fifo.sv's MIN_ABITS: the smallest instance of each primitive. A memory space below this
# cannot fill one, so the RTL relaxes its RAM_STYLE there (see _hi_in_ram).
BRAM_MIN_ABITS = 9  # RAMB18 at its deepest aspect, 512 x 36
URAM_MIN_ABITS = 12  # URAM288 is a fixed 4096 x 72

# Fitted RAMB18-per-URAM288 ratio at which the hi space joins the lo space's URAM instead
# of taking its own BRAM (below the primitives' 16:1 because the two spaces share a read
# path).
HI_URAM_RATIO = 7

# Fitted flat cost of the Versal URAM read path, which UltraScale+ does not pay.
VERSAL_URAM_LUTS = 20

# SDP configurations as (width, rows), widest first. RAMB36 needs no entry: two RAMB18
# worth of bits at coarser depth granularity. Versal's RAMB18E5 stops at 9 bits.
RAMB18_SDP = ((36, 512), (18, 1024), (9, 2048), (4, 4096), (2, 8192), (1, 16384))
RAMB18E5_SDP = ((36, 512), (18, 1024), (9, 2048))

# UltraScale+'s URAM288E2 offers only the first entry, Versal's URAM288E5 all four. Vivado
# never splits a word across URAM configurations, so a 37 bit word costs a 72.
URAM288_SDP = ((72, 4096), (36, 8192), (18, 16384), (9, 32768))

FifoCost = namedtuple("FifoCost", "bram uram lut")


def _clog2(x):
    """$clog2() as SystemVerilog defines it: 0 for x <= 1."""
    return max(x - 1, 0).bit_length()


def _geometry(depth, is_uram):
    """Address bits of the lo/hi memory spaces, mirroring INIT_ABITS().

    Returns (lo, hi); hi is 0 if the depth needs only one memory space."""
    min_abits, qdepth = (URAM_MIN_ABITS, 16) if is_uram else (BRAM_MIN_ABITS, 0)
    # one slot lives in the output register, QDEPTH more in the URAM queue
    depth_req = max(depth - qdepth - 1, 1)
    lo = _clog2(depth_req)
    hi = 0
    if lo > min_abits:
        hi_abits = _clog2(depth_req - 2 ** (lo - 1))
        if hi_abits < lo - 1:
            lo -= 1
            hi = max(hi_abits, 1)
    return lo, hi


def _hi_in_ram(rows, W):
    """Whether Vivado backs a rows x W hi memory space with RAM or with LUTRAM.

    Fitted boundary rows**2 * W >= 2**20: the width earning a RAM falls as the square of
    the depth. Needed because fifo.sv:383 relaxes a too-shallow hi space to
    RAM_STYLE="auto", leaving the choice to synthesis."""
    return rows * rows * W >= 2**20


def _hi_select_luts(W):
    """LUTs the lo/hi memory space selection costs: 18 + 3W/7, fitted.

    A pointer compare plus a W wide 2:1 mux. Shared by the block and ultra paths."""
    return 18 + 3 * W // 7


def _cascade_mux_luts(stages, W, versal):
    """LUTs the mux over an SRL cascade costs, fitted.

    0 on UltraScale+, whose F7/F8 wide muxes absorb it. Versal has none, so LUT6 only at
    4:1 per level, hence ceil((stages-1)/3) levels."""
    if not versal or stages < 2:
        return 0
    per_bit = 1 if stages > 2 else 0.5
    return math.ceil(W * per_bit * math.ceil((stages - 1) / 3)) - 2


def _lutram_luts(rows, W):
    """LUTs a rows x W LUTRAM costs: ceil(rows/32) * ceil(W/2) * 5/4.

    A LUT6 in RAM32X2 mode holds 32 rows x 2 bits; the 5/4 is fitted write-decode and
    output-select overhead. Reused by block/ultra for a hi space kept in LUTRAM."""
    return math.ceil(rows / 32) * math.ceil(W / 2) * 5 // 4


@lru_cache(maxsize=None)
def _bram18_plan(rows, W, versal=False):
    """Returns (tiles, groups) for a rows x W memory backed by RAMB18.

    groups is the deepest cascade in the plan, i.e. how far the output mux must reach.
    UltraScale+ splits the word over configurations, so tiles is the partition DP over the
    SDP ladder (must stay exhaustive: greedy widest-first misses measured counts). Versal
    does not split, so tiles = min over configs of ceil(rows/cr) * ceil(W/cw)."""
    if versal:
        return min(
            (math.ceil(rows / cfg_rows) * math.ceil(W / cfg_w), math.ceil(rows / cfg_rows))
            for cfg_w, cfg_rows in RAMB18E5_SDP
        )
    tiles = [0] + [None] * W
    groups = [0] * (W + 1)
    for w in range(1, W + 1):
        for cfg_w, cfg_rows in RAMB18_SDP:
            t = math.ceil(rows / cfg_rows)
            rest = max(w - cfg_w, 0)
            cand = (t + tiles[rest], max(t, groups[rest]))
            # the ladder is widest first, so a tie keeps the widest configuration
            if tiles[w] is None or cand[0] < tiles[w]:
                tiles[w], groups[w] = cand
    return tiles[W], groups[W]


def _bram_plan(depth, W, versal=False):
    """Returns (tiles, groups) for a BRAM-backed FIFO, over both memory spaces."""
    lo, hi = _geometry(depth, False)
    tiles, groups = _bram18_plan(2**lo, W, versal)
    if hi and _hi_in_ram(2**hi, W):
        hi_tiles, hi_groups = _bram18_plan(2**hi, W, versal)
        tiles, groups = tiles + hi_tiles, max(groups, hi_groups)
    return tiles, groups


def _uram_aspect(W, versal):
    """The URAM288 configuration a W-bit word is stored in, as (width, rows).

    A word is never split across configurations, so this is the narrowest one holding it
    whole. UltraScale+ has only the 72-bit one; Versal selects the URAM288E5 ladder."""
    if versal:
        for cfg in reversed(URAM288_SDP):
            if cfg[0] >= W:
                return cfg
    return URAM288_SDP[0]


def _uram_plan(rows, W, versal=False):
    """Returns (tiles, groups) for a rows x W memory backed by URAM288, as _bram18_plan.

    With no word splitting: groups = ceil(rows/cr), tiles = groups * ceil(W/cw)."""
    cfg_w, cfg_rows = _uram_aspect(W, versal)
    groups = math.ceil(rows / cfg_rows)
    return groups * math.ceil(W / cfg_w), groups


def _resolve(depth, W, requested):
    """Predicts the storage style fifo.sv's RAM_STYLE_EFF selection will elaborate.

    Mirror of fifo.sv:84-90; keep the thresholds below in sync with it. Takes no part,
    because the selection does not see one: FINN forwards RAM_STYLE untouched and the RTL
    decides. Where the part has no URAM, Vivado drops the attribute (Synth 8-12187) and
    backs the same array with BRAM, so this reports what was asked for, not what the
    device delivers."""
    if depth <= 33:
        # a shift register whatever the RAM_STYLE, so anything else only warns
        style = "srl"
    elif requested != "auto":
        style = requested
    elif depth <= 64 and W < 12:
        # RTL threshold, not fitted: the LUTRAM pointer overhead (~27 LUTs by the fitted
        # control terms below) does not amortize yet
        style = "srl"
    elif depth <= 257 or W < 5:
        # 256-entry LUTRAM + 1 output register. fifo.sv also sends any word narrower than
        # 5 bits here at any depth (fifo.sv:88): too few bit lanes to fill a BRAM/URAM
        # aspect, so LUTRAM stays cheaper
        style = "distributed"
    else:
        style = "block" if depth <= URAM_DEPTH_THRESHOLD else "ultra"
    return style


def _fifo_cost(depth, W, style, versal=False):
    """Returns the FifoCost of a depth x W fifo.sv instance in the given style.

    Fitted against fifo.sv; if the RTL's geometry or per-style logic changes, refit."""
    if depth < 2:
        # set_fifo_depths makes depth-0 FIFOs for MLO parameter inputs and estimation may
        # still see them; they hold nothing. fifo.sv will not build one,
        # StreamingFIFO_rtl.generate_hdl() asserts on that.
        return FifoCost(0, 0, 0)
    # Control is fitted linear in the counter width cw, not in depth. The slopes differ by
    # style because the users do: 5 srl (one up/down pointer), 8 distributed (separate
    # read and write addresses), 6 ultra (those plus the URAM credit counter).
    cw = _clog2(depth + 1) + 1

    if style == "srl":
        # stages * W exact (an SRLC32E is 32 bits in one LUT); fifo.sv floors shift
        # capacity at 5. Fitted control 5*cw - 7 overshoots at depth 2.
        depth_impl = depth - 1 if depth > 4 else 4
        stages = math.ceil(depth_impl / 32)
        mux = _cascade_mux_luts(stages, W, versal)
        return FifoCost(0, 0, stages * W + mux + 5 * cw - 7)
    if style == "distributed":
        rows = 2 ** _clog2(depth - 1)
        # floor(rows/128) * floor(W/2): the first 128 rows mux for free (F7 over a LUT6
        # pair), each further 128 adds a LUT6 level. Fitted control 8*cw - 15.
        mux_luts = rows // 128 * (W // 2)
        return FifoCost(0, 0, _lutram_luts(rows, W) + mux_luts + 8 * cw - 15)

    is_uram = style == "ultra"
    lo, hi = _geometry(depth, is_uram)
    hi_in_lutram = bool(hi) and not _hi_in_ram(2**hi, W)

    if not is_uram:
        tiles, groups = _bram_plan(depth, W, versal)
        # 54 + 3*groups + 2*tiles/5, fitted. 54 is size-independent; no cw term (a BRAM
        # address port takes the pointer bits directly); 3 per cascade level is the level
        # decode, 2/5 per tile the tile enable.
        lut = 54 + 3 * groups + 2 * tiles // 5 + (_hi_select_luts(W) if hi else 0)
        if hi_in_lutram:
            lut += _lutram_luts(2**hi, W)
        return FifoCost(tiles, 0, lut)

    uram, groups = _uram_plan(2**lo, W, versal)
    # a second memory space doubles the read path and the pointer arithmetic
    lut = 6 * cw + (_hi_select_luts(W) if hi else 0)
    if hi_in_lutram:
        # + W exact: under URAM the LUTRAM output is delayed to the URAM read latency, one
        # SRL16E (one LUT) per bit.
        lut += _lutram_luts(2**hi, W) + W
    # max(1, groups/4 - 2) * W, fitted and the largest term here: the 16-deep output queue
    # is 1 LUT/bit, past 8 tiles each further 4 spill one register stage. Keyed on the
    # cascade, not fifo.sv's PIPE_DEPTH: the two track on UltraScale+ but not on Versal.
    lut += max(1, groups // 4 - 2) * W
    if versal:
        # URAM288E5 does not absorb its read pipeline the way URAM288E2 does
        lut += VERSAL_URAM_LUTS
    bram = 0
    if hi and _hi_in_ram(2**hi, W):
        # fifo.sv:383 keeps RAM_STYLE_HI="ultra" for hi >= URAM_MIN_ABITS and relaxes it
        # to "auto" below, where Vivado chooses (fitted via HI_URAM_RATIO).
        hi_uram = _uram_plan(2**hi, W, versal)[0]
        hi_bram = _bram18_plan(2**hi, W, versal)[0]
        if hi >= URAM_MIN_ABITS or hi_bram >= HI_URAM_RATIO * hi_uram:
            uram += hi_uram
        else:
            bram = hi_bram
    return FifoCost(bram, uram, lut)
