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
from finn.util.basic import is_versal, part_has_uram

# a URAM288 holds as much as 16 RAMB18; the BRAM -> URAM switchover in _resolve()
BRAM18_PER_URAM = 16

# Fitted RAMB18 per URAM288 at which the hi memory space stops being worth its own BRAM
# and joins the URAM the lo space already occupies. Below the capacity ratio above
# because the two spaces then share one read path.
HI_URAM_RATIO = 7

# Fitted flat LUT cost of the URAM read path on Versal, which UltraScale+ does not pay.
# Flat in both width and depth, i.e. a fixed control block (read enable, pipeline valid).
VERSAL_URAM_LUTS = 20

# RAMB18 SDP configurations as (width, rows), widest first. RAMB36 needs no entry: at
# every width it holds exactly two RAMB18 worth of bits, only at a coarser depth
# granularity, so counting in RAMB18 covers both.
RAMB18_SDP = ((36, 512), (18, 1024), (9, 2048), (4, 4096), (2, 8192), (1, 16384))

# Versal's RAMB18E5 stops at 9 bits: no 4/2/1 bit SDP configuration, so a narrow word
# cannot be traded for depth as far there as on UltraScale+.
RAMB18E5_SDP = ((36, 512), (18, 1024), (9, 2048))

# URAM288 SDP configurations as (width, rows), widest first. UltraScale+ offers only the
# first; Versal's URAM288E5 offers all four. Vivado never splits a word across URAM
# configurations; it picks the narrowest entry holding the whole word, so a 37 bit word
# costs the same as a 72 bit one. Both ladders are selected by the same `versal` flag,
# which reaches the estimators as the is_versal node attribute.
URAM288_SDP = ((72, 4096), (36, 8192), (18, 16384), (9, 32768))

# The model below mirrors finn-rtllib/fifo/hdl/fifo.sv so that the style decision and
# the estimators share one description of it. W is the padded stream width.
# The terms fifo.sv leaves to synthesis (the hi memory space and the control logic) are
# fitted against out-of-context runs (Vivado 2026.1, xczu7ev and xcvc1902) covering every
# style, both memory geometries, widths 1..256 and depths 2..262145; that is the range
# the model is known good over. Tile counts are exact but for one config; LUT counts land
# within a few percent.
# A number marked fitted comes from those runs, and the comment gives the mechanism its
# shape points at, consistent with the measurements but not read off a netlist. A number
# not marked fitted is exact, taken from the RTL or from a primitive's datasheet.

FifoCost = namedtuple("FifoCost", "bram uram lut")


def _clog2(x):
    """$clog2() as SystemVerilog defines it: 0 for x <= 1."""
    return max(x - 1, 0).bit_length()


def _geometry(depth, is_uram):
    """Address bits of the lo/hi memory spaces, mirroring INIT_ABITS().

    Returns (lo, hi); hi is 0 if the depth needs only one memory space."""
    min_abits, qdepth = (12, 16) if is_uram else (9, 0)
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

    fifo.sv:383 relaxes a hi space too shallow for the chosen primitive back to
    RAM_STYLE="auto" and leaves the choice to synthesis, so this is the one decision
    FINN cannot read off the RTL, meaning it is fitted, not derived.

    Read as a width: RAM wins from W >= (1024/rows)**2 up, i.e. the width needed to
    earn a RAM falls as the *square* of the depth. Depth counts twice because a LUTRAM
    past 64 rows pays for extra banks *and* for the mux tree between them, while a RAM
    absorbs depth into the address port. A given (rows, W) always lands the same way in
    the measurements, so the outcome really is a function of these two alone."""
    return rows * rows * W >= 2**20


def _hi_select_luts(W):
    """LUTs the selection between the lo and hi memory spaces costs.

    Both the 18 intercept and the 3/7 slope are fitted: a pointer compare for the second
    space plus a W wide 2:1 multiplexer at a bit over two bits to a LUT6. The block and
    ultra paths share this term, being the same mechanism; fitting them separately does
    not pay for itself."""
    return 18 + 3 * W // 7


def _cascade_mux_luts(stages, W, versal):
    """LUTs the multiplexer over an SRL cascade costs.

    None on UltraScale+, where the F7/F8 wide multiplexers absorb it. The Versal CLB has
    no such multiplexers, so it is built from LUT6: a 2:1 selection needs three inputs
    and packs two bits into one LUT6, anything wider takes a whole LUT per bit for every
    4:1 level. A LUT6 spends two inputs on the select, hence 4:1 and hence one level per
    three stages beyond the first. The intercept is fitted, small enough to be the top
    select decode sharing the read pointer."""
    if not versal or stages < 2:
        return 0
    per_bit = 1 if stages > 2 else 0.5
    return math.ceil(W * per_bit * math.ceil((stages - 1) / 3)) - 2


def _lutram_luts(rows, W):
    """LUTs a rows x W LUTRAM costs: 5 per 4 bits of each 32-row bank.

    A LUT6 in RAM32X2 mode stores 32 rows of 2 bits, so the ceil(rows/32) x ceil(W/2) of
    pure storage is exact. The fitted part is the extra quarter on top, the write address
    decode and per-bank output selection Vivado builds around it. The per-bank cost is
    depth-independent in the measurements, so that residual is a function of W alone; it
    is not truly flat 5/4, since the overhead amortizes with width, but no closed form in
    the obvious families fits better across the measured widths.

    Fitted on `distributed` runs, but reused unfitted by the block/ultra branches for a
    hi space in LUTRAM. So it is the cost of a LUTRAM, not of a distributed FIFO."""
    return math.ceil(rows / 32) * math.ceil(W / 2) * 5 // 4


@lru_cache(maxsize=None)
def _bram18_plan(rows, W, versal=False):
    """Returns (tiles, groups) for a rows x W memory backed by RAMB18.

    groups is the deepest cascade in the plan, i.e. how far the output multiplexer has
    to reach.

    On UltraScale+ Vivado splits the word across several configurations, so 16384 x 32
    becomes 18b x 2048 + 9b x 4096 + 4b x 8192 + 1b x 16384, i.e. 29 tiles, rather than
    two 18-bit halves at 32 tiles. This picks the cheapest such partition. Versal does
    not split (the same 16384 x 32 measures 32 tiles), so its plan is the cheapest
    single configuration, ties going to the shallowest cascade.

    The UltraScale+ search must stay exhaustive: both greedy widest-first and a
    two-configuration partition miss measured tile counts this one gets."""
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


def _bram_tiles(depth, W, versal=False):
    """Returns the RAMB18 tiles a BRAM-backed FIFO of this depth costs."""
    return _bram_plan(depth, W, versal)[0]


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


def _uram_tiles(rows, W, versal=False):
    """URAM288 tiles for a rows x W memory; one URAM288 is 4096 x 72."""
    cfg_w, cfg_rows = _uram_aspect(W, versal)
    return math.ceil(rows / cfg_rows) * math.ceil(W / cfg_w)


def _uram_cascade(rows, W, versal=False):
    """URAM288 tiles stacked in the address direction, i.e. the read path depth."""
    return math.ceil(rows / _uram_aspect(W, versal)[1])


def _resolve(depth, W, requested, has_uram, versal=False):
    """Returns the concrete storage style, mirroring the RAM_STYLE_EFF ladder."""
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
        # unlike the RTL ladder, switch on cost rather than depth alone: at
        # depth 2029 and 8 bits that would spend a URAM288 on one RAMB18
        style = "ultra" if _bram_tiles(depth, W, versal) >= BRAM18_PER_URAM else "block"
    return "block" if style == "ultra" and not has_uram else style


def _fifo_cost(depth, W, style, versal=False):
    """Returns the FifoCost of a depth x W fifo.sv instance in the given style."""
    if depth < 2:
        # set_fifo_depths makes depth-0 FIFOs for MLO parameter inputs and estimation
        # may still see them; they hold nothing, so zero is honest. fifo.sv will not
        # build one; StreamingFIFO_rtl.generate_hdl() asserts on that.
        return FifoCost(0, 0, 0)
    # Control cost is fitted linear in the occupancy counter width, not in depth: the
    # up/down count, the full/empty compares and the maxcount running max each cost a
    # slice of logic per counter bit. The fitted slope is how many such per-bit users the
    # style keeps, the fitted negative intercept the low bits folding into a carry chain
    # instead. The three slopes (5 shift, 8 distributed, 6 ultra) cannot be shared, since
    # the styles keep different per-bit users: shift one up/down pointer, distributed
    # separate read and write addresses, ultra those plus the credit counter fronting the
    # URAM pipeline.
    cw = _clog2(depth + 1) + 1

    if style == "shift":
        # SRLC32E holds 32 bits per LUT, so one LUT per bit lane per stage. This is the
        # only exact storage term in the model, matching the measured srl column outright.
        # Most of the branch's residual error sits at depth 2, where the fitted -7
        # intercept overshoots: that degenerate depth costs more control logic than 5 does.
        # The output register holds one item; fifo.sv floors shift capacity at 5.
        depth_impl = depth - 1 if depth > 4 else 4
        stages = math.ceil(depth_impl / 32)
        mux = _cascade_mux_luts(stages, W, versal)
        return FifoCost(0, 0, stages * W + mux + 5 * cw - 7)
    if style == "distributed":
        rows = 2 ** _clog2(depth - 1)
        # a LUT6 pair gives 64 rows natively and F7 selects between two of them for
        # free, so the first 128 rows mux for nothing; each further 128 adds a LUT6
        # selection level at a fitted two bits per LUT. The measured srl column is 0
        # throughout, so nothing here is a shift register being charged as logic.
        mux_luts = rows // 128 * (W // 2)
        return FifoCost(0, 0, _lutram_luts(rows, W) + mux_luts + 8 * cw - 15)

    is_uram = style == "ultra"
    lo, hi = _geometry(depth, is_uram)
    hi_in_lutram = bool(hi) and not _hi_in_ram(2**hi, W)

    if not is_uram:
        tiles, groups = _bram_plan(depth, W, versal)
        # All three LUT terms are fitted. 54 is the size independent part (handshake,
        # output register bypass and counters), and is why no cw term appears: a BRAM
        # address port takes the pointer bits directly, so depth reaches the LUTs only
        # through the two memory terms. 3 per cascade level is the decode selecting that
        # level, not a data mux, hence no W scaling: the data rides the dedicated cascade
        # path. 2/5 per tile is each tile's enable, several of which pack into one LUT6.
        # The branch charges no shift register and the measured srl column agrees.
        lut = 54 + 3 * groups + 2 * tiles // 5 + (_hi_select_luts(W) if hi else 0)
        if hi_in_lutram:
            lut += _lutram_luts(2**hi, W)
        return FifoCost(tiles, 0, lut)

    # a second memory space doubles the read path and the pointer arithmetic
    lut = 6 * cw + (_hi_select_luts(W) if hi else 0)
    if hi_in_lutram:
        # under URAM the LUTRAM read pipeline stops being absorbed: its output must be
        # delayed to meet the URAM read latency, and a delay of PIPE_DEPTH is one
        # SRL16E, i.e. one LUT, per bit. Measured rather than fitted, since the srl
        # column shows it on its own.
        lut += _lutram_luts(2**hi, W) + W
    # The read path spans the URAM cascade: the 16-deep output queue costs a LUT per bit,
    # and past eight tiles every further four spill one more, consistent with Vivado
    # inserting a register stage per four cascaded tiles, each a shift of one LUT per bit.
    # Fitted, and the largest fitted term in this branch. Keyed on the cascade rather than
    # fifo.sv's PIPE_DEPTH, which tracks it on UltraScale+ but not on Versal, whose aspect
    # ladder shortens the cascade without changing PIPE_DEPTH.
    lut += max(1, _uram_cascade(2**lo, W, versal) // 4 - 2) * W
    if versal:
        # URAM288E5 does not absorb its read pipeline the way URAM288E2 does
        lut += VERSAL_URAM_LUTS
    uram = _uram_tiles(2**lo, W, versal)
    bram = 0
    if hi and _hi_in_ram(2**hi, W):
        tiles = _bram18_plan(2**hi, W, versal)[0]
        # the hi space moves into the URAM the lo space already occupies once it
        # fills a URAM288's depth outright, or once BRAM would spend more than
        # HI_URAM_RATIO tiles per URAM288 the same space would take
        if 2**hi >= URAM288_SDP[0][1] or tiles >= HI_URAM_RATIO * _uram_tiles(2**hi, W, versal):
            uram += _uram_tiles(2**hi, W, versal)
        else:
            bram = tiles
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
                # requested FPGA resource for the storage: auto (FINN decides, see
                # resolve_ram_style()), shift (SRL), block (BRAM), distributed
                # (LUTRAM) or ultra (URAM, on UltraScale+ and Versal)
                "ram_style": (
                    "s",
                    False,
                    "auto",
                    {"auto", "shift", "block", "distributed", "ultra"},
                ),
                # concrete storage style resolved from ram_style, written by
                # generate_hdl() and read back by the resource estimators
                "ram_style_resolved": ("s", False, ""),
                # whether the target is a Versal part, which has its own BRAM and URAM
                # aspect ladders. Recorded alongside ram_style_resolved for the same
                # reason: the estimators are called without an fpgapart
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

    def resolve_ram_style(self, fpgapart=None):
        """Resolves ram_style into one of shift/distributed/block/ultra.

        FINN decides here rather than leaving RAM_STYLE="auto" to fifo.sv: only here is
        the target device known, and only a decision taken here reaches resource
        estimation, the build report and the folding config. fpgapart may be None before
        ipgen, in which case URAM is not selected."""
        requested = self.get_nodeattr("ram_style")
        depth = self.get_nodeattr("depth")
        W = self.get_instream_width_padded()
        style = _resolve(
            depth,
            W,
            requested,
            part_has_uram(fpgapart),
            bool(fpgapart) and is_versal(fpgapart),
        )
        # an explicit request is returned verbatim, so this can only be the clamp
        if style == "block" and requested == "ultra" and fpgapart is not None:
            warnings.warn(
                "%s: ram_style=ultra requested but %s provides no URAM, using BRAM instead"
                % (self.onnx_node.name, fpgapart)
            )
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
        """Returns the concrete storage style this FIFO is built with.

        Prefers what generate_hdl() recorded, so estimates agree with the RTL."""
        style = self.get_nodeattr("ram_style_resolved")
        if style == "":
            style = self.resolve_ram_style()
        return style

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
