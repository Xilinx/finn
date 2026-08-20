# Copyright (C) 2024, Advanced Micro Devices, Inc.
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
import math
import warnings
from collections import namedtuple
from functools import lru_cache
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp

# This model mirrors finn-rtllib/fifo/hdl/fifo.sv, so that the style decision and the
# estimators share one description of it. W is the padded stream width.
#
# Terms marked "fitted" come from out-of-context runs (Vivado 2026.1, xczu7ev and
# xcvc1902) over every style, both memory geometries, widths 1..256 and depths 2..262145;
# the comment names the mechanism their shape points at. Everything else is exact, read
# off the RTL or a datasheet. Tile counts are exact but for one config, LUTs land within
# a few percent.

# fifo.sv's own RAM_STYLE_EFF threshold, not a cost comparison. Change it there first.
URAM_DEPTH_THRESHOLD = 2028

# fifo.sv's MIN_ABITS: the smallest instance of each primitive. A memory space below this
# cannot fill one, which is why the RTL relaxes its RAM_STYLE there (see _hi_in_ram).
BRAM_MIN_ABITS = 9  # RAMB18 at its deepest aspect, 512 x 36
URAM_MIN_ABITS = 12  # URAM288 is a fixed 4096 x 72

# Fitted RAMB18-per-URAM288 at which the hi space stops earning its own BRAM and joins
# the lo space's URAM. Below the primitives' 16:1 capacity ratio because the two spaces
# then share one read path.
HI_URAM_RATIO = 7

# Fitted flat cost of the Versal URAM read path, which UltraScale+ does not pay. Flat in
# width and depth, i.e. a fixed control block (read enable, pipeline valid).
VERSAL_URAM_LUTS = 20

# SDP configurations as (width, rows), widest first, selected by the `versal` flag that
# reaches the estimators as the is_versal attribute. RAMB36 needs no entry: it is two
# RAMB18 worth of bits at coarser depth granularity. Versal's RAMB18E5 stops at 9 bits,
# so a narrow word cannot be traded for depth as far there.
RAMB18_SDP = ((36, 512), (18, 1024), (9, 2048), (4, 4096), (2, 8192), (1, 16384))
RAMB18E5_SDP = ((36, 512), (18, 1024), (9, 2048))

# UltraScale+'s URAM288E2 offers only the first entry, Versal's URAM288E5 all four.
# Vivado never splits a word across URAM configurations, so a 37 bit word costs a 72.
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

    Fitted boundary rows**2 * W >= 2**20, i.e. W >= (1024/rows)**2: the width earning a
    RAM falls as the square of the depth, LUTRAM paying for banks and for the mux tree
    over them while a RAM absorbs depth into the address port. Needed because fifo.sv:383
    relaxes a too-shallow hi space to RAM_STYLE="auto", leaving the choice to synthesis."""
    return rows * rows * W >= 2**20


def _hi_select_luts(W):
    """LUTs the lo/hi memory space selection costs: 18 + 3W/7, fitted.

    A pointer compare plus a W wide 2:1 mux, the slope's 7/3 = 2.33 bits per LUT6. Shared
    by the block and ultra paths."""
    return 18 + 3 * W // 7


def _cascade_mux_luts(stages, W, versal):
    """LUTs the mux over an SRL cascade costs: ceil(W * b * ceil((stages-1)/3)) - 2.

    0 on UltraScale+, whose F7/F8 wide muxes absorb it. Versal has none, so LUT6 only: 2
    of 6 inputs carry the select, giving 4:1 per level and hence ceil((stages-1)/3)
    levels. b = 1/2 bits per LUT at stages == 2 (a 3-input 2:1 pairs into one LUT6), else
    1. Fitted intercept -2."""
    if not versal or stages < 2:
        return 0
    per_bit = 1 if stages > 2 else 0.5
    return math.ceil(W * per_bit * math.ceil((stages - 1) / 3)) - 2


def _lutram_luts(rows, W):
    """LUTs a rows x W LUTRAM costs: ceil(rows/32) * ceil(W/2) * 5/4.

    Exact storage: a LUT6 in RAM32X2 mode holds 32 rows x 2 bits. The 5/4 is fitted, the
    write address decode and per-bank output selection; it is depth-independent in the
    measurements, so a function of W alone.

    Fitted on `distributed` runs, reused by block/ultra for a hi space in LUTRAM: it is
    the cost of a LUTRAM, not of a distributed FIFO."""
    return math.ceil(rows / 32) * math.ceil(W / 2) * 5 // 4


@lru_cache(maxsize=None)
def _bram18_plan(rows, W, versal=False):
    """Returns (tiles, groups) for a rows x W memory backed by RAMB18.

    groups is the deepest cascade in the plan, i.e. how far the output mux must reach.

    UltraScale+ splits the word over several configurations, so the cost is the partition
    DP over the SDP ladder:
        T(0) = 0
        T(w) = min over (cw, cr) of ceil(rows/cr) + T(max(w-cw, 0))
    e.g. 16384 x 32 = 18b x 2048 + 9b x 4096 + 4b x 8192 + 1b x 16384 = 29 tiles, against
    32 for two 18-bit halves. The DP must stay exhaustive: greedy widest-first and a
    two-configuration partition both miss measured counts.

    Versal does not split, so T = min over (cw, cr) of ceil(rows/cr) * ceil(W/cw), ties to
    the shallowest cascade; the same 16384 x 32 measures 32 tiles."""
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

    A word is never split across URAM configurations, so this is simply the narrowest
    one holding it whole. UltraScale+'s URAM288E2 has only the 72-bit one; versal
    selects the URAM288E5 ladder."""
    if versal:
        for cfg in reversed(URAM288_SDP):
            if cfg[0] >= W:
                return cfg
    return URAM288_SDP[0]


def _uram_plan(rows, W, versal=False):
    """Returns (tiles, groups) for a rows x W memory backed by URAM288, as _bram18_plan.

    With no word splitting the plan is one aspect, so both factors come from it:
        groups = ceil(rows/cr)              tiles stacked in the address direction,
                                            i.e. the read path depth
        tiles  = groups * ceil(W/cw)"""
    cfg_w, cfg_rows = _uram_aspect(W, versal)
    groups = math.ceil(rows / cfg_rows)
    return groups * math.ceil(W / cfg_w), groups


def _resolve(depth, W, requested):
    """Predicts the storage style fifo.sv's RAM_STYLE_EFF ladder will elaborate.

    Takes no part, because the ladder does not see one: FINN forwards RAM_STYLE
    untouched and the RTL is the decision. Where the part turns out to have no URAM,
    Vivado drops the attribute (Synth 8-12187) and backs the same array with BRAM, so
    the style below is what was asked for rather than what the device delivers."""
    if depth <= 33:
        # a shift register whatever the RAM_STYLE, so anything else only warns
        style = "shift"
    elif requested != "auto":
        style = requested
    elif depth <= 64 and W < 12:
        # this threshold and the two around it are the RTL ladder's, not fitted: the
        # LUTRAM pointer overhead, ~27 LUTs by the fitted control terms below, does not
        # amortize yet
        style = "shift"
    elif depth <= 257:
        # 256-entry LUTRAM + 1 output register, i.e. 4x RAM64M8 per byte
        style = "distributed"
    else:
        style = "block" if depth <= URAM_DEPTH_THRESHOLD else "ultra"
    return style


def _fifo_cost(depth, W, style, versal=False):
    """Returns the FifoCost of a depth x W fifo.sv instance in the given style."""
    if depth < 2:
        # set_fifo_depths makes depth-0 FIFOs for MLO parameter inputs and estimation
        # may still see them; they hold nothing, so zero is honest. fifo.sv will not
        # build one; StreamingFIFO_rtl.generate_hdl() asserts on that.
        return FifoCost(0, 0, 0)
    # Control is fitted linear in the counter width cw, not in depth: the up/down count,
    # the full/empty compares and the maxcount running max each cost logic per counter
    # bit. Slope = per-bit users the style keeps, negative intercept = low bits folding
    # into a carry chain. The slopes differ because the users do: 5 shift (one up/down
    # pointer), 8 distributed (separate read and write addresses), 6 ultra (those plus
    # the credit counter fronting the URAM pipeline).
    cw = _clog2(depth + 1) + 1

    if style == "shift":
        # stages * W exact: an SRLC32E is 32 bits in one LUT, so one LUT per bit lane per
        # stage, matching the measured srl column outright. One item sits in the output
        # register and fifo.sv floors shift capacity at 5. Fitted intercept -7 overshoots
        # at depth 2, which carries most of this branch's residual error.
        depth_impl = depth - 1 if depth > 4 else 4
        stages = math.ceil(depth_impl / 32)
        mux = _cascade_mux_luts(stages, W, versal)
        return FifoCost(0, 0, stages * W + mux + 5 * cw - 7)
    if style == "distributed":
        rows = 2 ** _clog2(depth - 1)
        # floor(rows/128) * floor(W/2): a LUT6 pair is 64 rows and F7 selects over two of
        # them free, so the first 128 rows mux for nothing and each further 128 adds one
        # LUT6 level at a fitted 2 bits per LUT. The measured srl column is 0 throughout,
        # so no shift register is being charged here as logic.
        mux_luts = rows // 128 * (W // 2)
        return FifoCost(0, 0, _lutram_luts(rows, W) + mux_luts + 8 * cw - 15)

    is_uram = style == "ultra"
    lo, hi = _geometry(depth, is_uram)
    hi_in_lutram = bool(hi) and not _hi_in_ram(2**hi, W)

    if not is_uram:
        tiles, groups = _bram_plan(depth, W, versal)
        # 54 + 3*groups + 2*tiles/5, all fitted. 54 is size-independent (handshake,
        # output register bypass, counters); no cw term appears because a BRAM address
        # port takes the pointer bits directly, so depth reaches the LUTs only via the
        # two memory terms. 3 per cascade level is the level decode, not a data mux,
        # hence no W: data rides the dedicated cascade path. 2/5 per tile is the tile
        # enable, ~2.5 packing into one LUT6. No shift register, and the srl column
        # agrees.
        lut = 54 + 3 * groups + 2 * tiles // 5 + (_hi_select_luts(W) if hi else 0)
        if hi_in_lutram:
            lut += _lutram_luts(2**hi, W)
        return FifoCost(tiles, 0, lut)

    uram, groups = _uram_plan(2**lo, W, versal)
    # a second memory space doubles the read path and the pointer arithmetic
    lut = 6 * cw + (_hi_select_luts(W) if hi else 0)
    if hi_in_lutram:
        # + W exact: under URAM the LUTRAM output must be delayed to the URAM read
        # latency, and a delay of PIPE_DEPTH is one SRL16E, i.e. one LUT, per bit. The
        # srl column shows this term on its own.
        lut += _lutram_luts(2**hi, W) + W
    # max(1, groups/4 - 2) * W, fitted, and the largest fitted term in this branch. The
    # 16-deep output queue is 1 LUT/bit; past 8 tiles each further 4 spill one more,
    # i.e. one register stage per 4 cascaded tiles at 1 LUT/bit. Keyed on the cascade,
    # not fifo.sv's PIPE_DEPTH: the two track on UltraScale+ but not on Versal, whose
    # aspect ladder shortens the cascade at unchanged PIPE_DEPTH.
    lut += max(1, groups // 4 - 2) * W
    if versal:
        # URAM288E5 does not absorb its read pipeline the way URAM288E2 does
        lut += VERSAL_URAM_LUTS
    bram = 0
    if hi and _hi_in_ram(2**hi, W):
        # fifo.sv:383 keeps RAM_STYLE_HI = "ultra" for hi >= URAM_MIN_ABITS and relaxes
        # it to "auto" below. Above: exact, read off the RTL. Below: Vivado chooses, and
        # the HI_URAM_RATIO comparison is fitted to what it was measured to choose.
        hi_uram = _uram_plan(2**hi, W, versal)[0]
        hi_bram = _bram18_plan(2**hi, W, versal)[0]
        if hi >= URAM_MIN_ABITS or hi_bram >= HI_URAM_RATIO * hi_uram:
            uram += hi_uram
        else:
            bram = hi_bram
    return FifoCost(bram, uram, lut)


class StreamingFIFO(HWCustomOp):
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = super().get_nodeattr_types()
        my_attrs.update(
            {
                # FIFO depth
                "depth": ("i", True, 0),
                # folded shape of input/output
                "folded_shape": ("ints", True, []),
                # normal shape of input/output
                "normal_shape": ("ints", True, []),
                # FINN DataTypes for inputs/outputs
                "dataType": ("s", True, ""),
                # requested FPGA resource for the storage, passed straight to fifo.sv:
                # auto (its RAM_STYLE_EFF ladder decides), shift (SRL), block (BRAM),
                # distributed (LUTRAM) or ultra (URAM, on UltraScale+ and Versal)
                "ram_style": (
                    "s",
                    False,
                    "auto",
                    {"auto", "shift", "block", "distributed", "ultra"},
                ),
                # which style that ladder is predicted to elaborate. Reporting only;
                # resolve_ram_style() derives it on demand, so nothing reads it back
                "ram_style_resolved": ("s", False, ""),
                # whether the target is a Versal part, which has its own BRAM and URAM
                # aspect ladders. Recorded because the estimators, unlike the ladder,
                # do need the part and are called without an fpgapart
                "is_versal": ("i", False, 0),
                # whether the maxcount occupancy output is exposed on the wrapper
                "depth_monitor": ("i", False, 0),
                # the FIFO does not need its own FIFOs
                "inFIFODepths": ("ints", False, [0]),
                "outFIFODepths": ("ints", False, [0]),
                "debug_log_path": ("s", False, ""),
            }
        )

        return my_attrs

    def resolve_ram_style(self):
        """Predicts which of shift/distributed/block/ultra fifo.sv will elaborate.

        generate_hdl() forwards ram_style to the RTL untouched, so this decides
        nothing: it reproduces the RAM_STYLE_EFF ladder so that resource estimation,
        the build report and the folding config describe what actually gets built.
        The ladder is a function of depth, width and the request alone, so unlike the
        estimators this needs no fpgapart."""
        requested = self.get_nodeattr("ram_style")
        depth = self.get_nodeattr("depth")
        W = self.get_instream_width_padded()
        style = _resolve(depth, W, requested)
        # the shallow-depth downgrade, warned about here as fifo.sv warns about it.
        # "distributed" is excluded in both: an SRL is a strict improvement on a LUTRAM
        # this shallow, so asking for LUTRAM is not asking for a memory
        if style == "shift" and requested in ("block", "ultra"):
            warnings.warn(
                "%s: ram_style=%s requested but depth %d is built as a shift register"
                % (self.onnx_node.name, requested, depth)
            )
        # shift is never auto-selected past 257, so a deeper one is an explicit
        # request, most likely large_fifo_mem_style, whose name suggests a memory.
        if style == "shift" and depth > 257:
            warnings.warn(
                "%s: ram_style=shift at depth %d costs roughly %d LUTs of shift "
                "register; consider distributed/block/ultra instead"
                % (self.onnx_node.name, depth, _fifo_cost(depth, W, "shift").lut)
            )
        return style

    def infer_node_datatype(self, model):
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            warn_str = "inputDataType changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype()),
                str(idt),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("dataType", idt.name)
        # data type stays the same
        model.set_tensor_datatype(node.output[0], idt)

    def get_normal_input_shape(self, ind=0):
        assert self.get_nodeattr("depth") >= 1, """Depth is too low"""
        return self.get_nodeattr("normal_shape")

    def get_normal_output_shape(self, ind=0):
        return self.get_normal_input_shape()

    def get_folded_input_shape(self, ind=0):
        return self.get_nodeattr("folded_shape")

    def get_folded_output_shape(self, ind=0):
        return self.get_nodeattr("folded_shape")

    def get_instream_width(self, ind=0):
        dtype = DataType[self.get_nodeattr("dataType")]
        folded_shape = self.get_nodeattr("folded_shape")
        in_width = folded_shape[-1] * dtype.bitwidth()
        return in_width

    def get_outstream_width(self, ind=0):
        dtype = DataType[self.get_nodeattr("dataType")]
        folded_shape = self.get_nodeattr("folded_shape")
        in_width = folded_shape[-1] * dtype.bitwidth()
        return in_width

    def get_input_datatype(self, ind=0):
        return DataType[self.get_nodeattr("dataType")]

    def get_output_datatype(self, ind=0):
        return DataType[self.get_nodeattr("dataType")]

    def execute_node(self, context, graph):
        node = self.onnx_node
        context[node.output[0]] = context[node.input[0]]

    def get_ram_style(self):
        """Returns the storage style this FIFO is built with.

        Derived rather than read back from ram_style_resolved: the ladder depends only
        on attributes this node already carries, so recomputing cannot go stale the way
        a recorded value does when set_fifo_depths changes the depth."""
        return self.resolve_ram_style()

    def get_fifo_cost(self):
        """Returns the FifoCost of this node as fifo.sv implements it."""
        return _fifo_cost(
            self.get_nodeattr("depth"),
            self.get_instream_width_padded(),
            self.get_ram_style(),
            self.get_nodeattr("is_versal") == 1,
        )

    def bram_estimation(self):
        """Calculates resource estimation for BRAM"""
        return self.get_fifo_cost().bram

    def uram_estimation(self):
        """Calculates resource estimation for URAM"""
        return self.get_fifo_cost().uram

    def lut_estimation(self):
        """Calculates resource estimations for LUTs"""
        return self.get_fifo_cost().lut

    def bram_efficiency_estimation(self):
        bram_est = self.bram_estimation()
        if bram_est == 0:
            return 1
        wbits = self.get_instream_width_padded() * self.get_nodeattr("depth")
        return wbits / (bram_est * 18 * 1024)

    def uram_efficiency_estimation(self):
        # every URAM288 aspect holds 288 Kib, so this capacity is correct on the
        # Versal ladder too; narrow words show up as a smaller uram_estimation()
        uram_est = self.uram_estimation()
        if uram_est == 0:
            return 1
        wbits = self.get_instream_width_padded() * self.get_nodeattr("depth")
        return wbits / (uram_est * 72 * 4096)
