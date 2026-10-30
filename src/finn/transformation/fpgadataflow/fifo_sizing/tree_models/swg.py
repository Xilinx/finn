# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Tree model of the RTL sliding-window generator (ConvolutionInputGenerator)."""

import math
import numpy as np
from qonnx.custom_op.general.im2col import compute_conv_output_dim

from finn.transformation.fpgadataflow.fifo_sizing.tav import CharacteristicNode
from finn.transformation.fpgadataflow.fifo_sizing.tree_models.base import (
    TreeModel,
    register,
)

# ---------------------------------------------------------------------------
# Exact schedule of the RTL sliding-window generator, "default" impl style.
#
# The closed forms in SlidingWindowRtlModel approximate this hardware with a handful of
# special cases, and the commonest of them mispredicts by exactly
# k_y * (IFMChannels/SIMD) tokens. That is unnecessary: the hardware is
# swg_controller (finn-rtllib/swg/swg_common.sv) plus the counter block of
# swg_template_default.sv, a small finite state machine with no data dependence.
# The characterisation stimulus holds the input stream valid and the output
# stream ready for the whole run, which removes the only external influence on
# it, so the schedule is a pure function of the generated parameters.
#
# Executing that FSM in Python costs one loop iteration per cycle -- no IP
# synthesis, no simulator -- and reproduces the rtlsim token access vector
# exactly, bit for bit, over the configurations covered by
# test_fpgadataflow_convinputgenerator.py's tree-model test.
# ---------------------------------------------------------------------------


def swg_default_params(ifm_ch, simd, k, ifm_dim, stride, dilation, depthwise, buffer_actual_size):
    """The parameters that prepare_codegen_default() substitutes into the RTL.

    Deliberately a function of plain numbers rather than of the node: it has to
    stay diffable against prepare_codegen_default, which is where these
    expressions come from, and it must be callable without a ModelWrapper.
    """
    k_h, k_w = k
    h, w = ifm_dim
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    channel_factor = ifm_ch // simd

    out_dim_h = compute_conv_output_dim(h, k_h, stride_h, 0, dilation_h)
    out_dim_w = compute_conv_output_dim(w, k_w, stride_w, 0, dilation_w)

    buffer_min_size = ((k_h - 1) * dilation_h * w + (k_w - 1) * dilation_w + 1) * channel_factor

    kernel_width = (k_w - 1) * dilation_w + 1
    kernel_height = (k_h - 1) * dilation_h + 1
    skip_columns = w % (kernel_width + (out_dim_w - 1) * stride_w)
    skip_rows = h % (kernel_height + (out_dim_h - 1) * stride_h)

    addr_incr_end_simd = 1
    addr_incr_end_window_elem = (dilation_w - 1) * channel_factor + 1
    addr_incr_end_window_row = (
        ((w - kernel_width) * channel_factor) + ((dilation_h - 1) * w * channel_factor) + 1
    )
    addr_incr_end_window = -buffer_min_size + stride_w * channel_factor + 1
    addr_incr_end_row = (
        -buffer_min_size
        + ((skip_columns + kernel_width) * channel_factor)
        + ((stride_h - 1) * w * channel_factor)
        + 1
    )

    if depthwise:
        addr_incr_end_window_elem = dilation_w * channel_factor
        addr_incr_end_window_row = (
            channel_factor
            + (w - kernel_width) * channel_factor
            + (dilation_h - 1) * w * channel_factor
        )
        addr_incr_end_simd = -buffer_min_size + (channel_factor + 1)

    loop_h_iterations = out_dim_h
    loop_w_iterations = out_dim_w
    loop_kh_iterations = k_h
    loop_kw_iterations = k_w
    loop_simd_iterations = channel_factor

    if depthwise and channel_factor > 1:
        loop_kh_iterations = channel_factor
        loop_kw_iterations = k_h
        loop_simd_iterations = k_w
        addr_incr_end_simd_ = addr_incr_end_simd
        addr_incr_end_simd = addr_incr_end_window_elem
        addr_incr_end_window_elem = addr_incr_end_window_row
        addr_incr_end_window_row = addr_incr_end_simd_
        elem_per_window = k_h * k_w
        tail_incr_w = addr_incr_end_window + buffer_min_size - channel_factor
        tail_incr_h = addr_incr_end_row + buffer_min_size - channel_factor
        is_depthwise = 1
    else:
        elem_per_window = k_h * k_w * channel_factor
        tail_incr_w = addr_incr_end_window + buffer_min_size - 1
        tail_incr_h = addr_incr_end_row + buffer_min_size - 1
        is_depthwise = 0
    tail_incr_last_window = buffer_min_size - 1

    if loop_simd_iterations == 1:
        # the innermost loop always executes at least once, so the state it
        # starts in is skipped and its counter loses an iteration
        if loop_kw_iterations == 1:
            innermost = "KH"
            loop_kh_iterations -= 1
        else:
            innermost = "KW"
            loop_kw_iterations -= 1
    else:
        innermost = "SIMD"
        loop_simd_iterations -= 1

    return {
        "LOOP_H_ITERATIONS": loop_h_iterations - 2,
        "LOOP_W_ITERATIONS": loop_w_iterations - 2,
        "LOOP_KH_ITERATIONS": loop_kh_iterations - 2,
        "LOOP_KW_ITERATIONS": loop_kw_iterations - 2,
        "LOOP_SIMD_ITERATIONS": loop_simd_iterations - 2,
        "HEAD_INCR_SIMD": addr_incr_end_simd,
        "HEAD_INCR_KW": addr_incr_end_window_elem,
        "HEAD_INCR_KH": addr_incr_end_window_row,
        "HEAD_INCR_W": addr_incr_end_window,
        "HEAD_INCR_H": addr_incr_end_row,
        "TAIL_INCR_W": tail_incr_w,
        "TAIL_INCR_H": tail_incr_h,
        "TAIL_INCR_LAST": tail_incr_last_window,
        "IS_DEPTHWISE": is_depthwise,
        "INNERMOST_STATE": innermost,
        "LAST_READ_ELEM": h * w * channel_factor - 1,
        "LAST_WRITE_ELEM": ((h - skip_rows - 1) * w + (w - skip_columns)) * channel_factor - 1,
        "BUF_ELEM_TOTAL": buffer_actual_size,
        "ELEM_PER_WINDOW": elem_per_window,
    }


def swg_default_schedule(p, n_feature_maps=4, hard_limit=None):
    """Per-cycle (input transaction, output transaction) of the default-style SWG.

    Runs the FSM from reset with in0_V_V_TVALID and out_V_V_TREADY tied high.
    Returns ``(schedule, restarts)`` where ``restarts`` holds the cycle indices
    at which the generator wrapped round to the next feature map -- the period is
    the spacing between two of those, once the start-up transient is past.
    """
    LAST_READ = p["LAST_READ_ELEM"]
    LAST_WRITE = p["LAST_WRITE_ELEM"]
    BUF = p["BUF_ELEM_TOTAL"]
    EPW = p["ELEM_PER_WINDOW"]
    IS_DW = p["IS_DEPTHWISE"]
    INNER = p["INNERMOST_STATE"]
    TAIL_W = p["TAIL_INCR_W"]
    TAIL_H = p["TAIL_INCR_H"]
    TAIL_LAST = p["TAIL_INCR_LAST"]
    HEAD = {
        "START": 0,
        "SIMD": p["HEAD_INCR_SIMD"],
        "KW": p["HEAD_INCR_KW"],
        "KH": p["HEAD_INCR_KH"],
        "W": p["HEAD_INCR_W"],
        "H": p["HEAD_INCR_H"],
    }
    IT_H = p["LOOP_H_ITERATIONS"]
    IT_W = p["LOOP_W_ITERATIONS"]
    IT_KH = p["LOOP_KH_ITERATIONS"]
    IT_KW = p["LOOP_KW_ITERATIONS"]
    IT_SIMD = p["LOOP_SIMD_ITERATIONS"]

    state = INNER
    c_h, c_w, c_kh, c_kw, c_simd = IT_H, IT_W, IT_KH, IT_KW, IT_SIMD
    newest, current, first_next, pos_in_window = -1, 0, 0, 0
    fetching_done = write_cmd = writing_done = 0

    # a cycle bound that cannot be hit in a healthy configuration, so a bug in
    # the FSM transcription shows up as a bounded run rather than a hang
    if hard_limit is None:
        hard_limit = 64 * (LAST_READ + 1) * (EPW + 4) + 4096

    sched = []
    restarts = []
    cycle = 0
    while cycle < hard_limit and len(restarts) < n_feature_maps:
        write_ok = write_cmd  # out_V_V_TREADY tied high
        fetch_cmd = (current <= newest) and not fetching_done
        reading_done = newest == LAST_READ
        oldest = newest - (BUF - 1)
        read_ok = (not reading_done) and (
            fetching_done or (oldest < first_next and oldest < current)
        )

        addr_incr = HEAD[state]
        if IS_DW and c_kh >= 0:
            tail_incr = 1
        elif c_w >= 0:
            tail_incr = TAIL_W
        elif c_h >= 0:
            tail_incr = TAIL_H
        else:
            tail_incr = TAIL_LAST

        if state != INNER:
            state_next = INNER
        elif c_simd < 0:
            state_next = (
                "KW"
                if c_kw >= 0
                else ("KH" if c_kh >= 0 else "W" if c_w >= 0 else "H" if c_h >= 0 else "START")
            )
        else:
            state_next = state

        sched.append((int(read_ok), int(write_ok)))

        # sequential block, in source order so that a later assignment to the
        # same register wins, as it does in the always_ff
        n_newest, n_current, n_first, n_pos = newest, current, first_next, pos_in_window
        n_fetching, n_write_cmd, n_writing = fetching_done, write_cmd, writing_done
        restarted = False

        if read_ok:
            n_newest = newest + 1
            if newest == LAST_READ - 1 and writing_done:
                n_newest, n_current, n_first = -1, 0, 0
                n_writing = n_fetching = 0
                restarted = True

        if fetch_cmd:
            n_pos = pos_in_window + 1 if pos_in_window != EPW - 1 else 0
            if pos_in_window == 0:
                n_first = first_next + tail_incr
            if current == LAST_WRITE:
                n_fetching = 1
            else:
                n_current = current + addr_incr
            n_write_cmd = 1

        if write_ok:
            n_write_cmd = 1 if fetch_cmd else 0

        if write_ok and fetching_done:
            if reading_done or (read_ok and newest == LAST_READ - 1):
                n_newest, n_current, n_first, n_fetching = -1, 0, 0, 0
                restarted = True
            else:
                n_writing = 1

        newest, current, first_next, pos_in_window = n_newest, n_current, n_first, n_pos
        fetching_done, write_cmd, writing_done = n_fetching, n_write_cmd, n_writing

        if fetch_cmd:
            # the counter cascade is gated on the state *before* the update
            if state == INNER:
                if c_simd >= 0:
                    c_simd -= 1
                else:
                    c_simd = IT_SIMD
                    if c_kw >= 0:
                        c_kw -= 1
                    else:
                        c_kw = IT_KW
                        if c_kh >= 0:
                            c_kh -= 1
                        else:
                            c_kh = IT_KH
                            if c_w >= 0:
                                c_w -= 1
                            else:
                                c_w = IT_W
                                if c_h >= 0:
                                    c_h -= 1
                                else:
                                    c_h = IT_H
            state = state_next

        if restarted:
            restarts.append(cycle)
        cycle += 1

    return sched, restarts


def swg_parallel_params(ifm_ch, simd, k, ifm_dim, stride, dilation):
    """The parameters that prepare_codegen_parallel() substitutes into the RTL."""
    k_h, k_w = k
    h, w = ifm_dim
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    channel_factor = ifm_ch // simd

    out_dim_h = compute_conv_output_dim(h, k_h, stride_h, 0, dilation_h)
    out_dim_w = compute_conv_output_dim(w, k_w, stride_w, 0, dilation_w)

    buffer_min_size = ((k_h - 1) * dilation_h * w + (k_w - 1) * dilation_w) * channel_factor + 1
    kernel_width = (k_w - 1) * dilation_w + 1
    kernel_height = (k_h - 1) * dilation_h + 1
    skip_columns = w % (kernel_width + (out_dim_w - 1) * stride_w)
    skip_rows = h % (kernel_height + (out_dim_h - 1) * stride_h)

    loop_h_iterations = out_dim_h
    loop_w_iterations = out_dim_w
    loop_kh_iterations = channel_factor
    loop_kw_iterations = 1
    loop_simd_iterations = 1

    if loop_kh_iterations == 1:
        if loop_w_iterations == 1:
            innermost = "H"
            loop_h_iterations -= 1
        else:
            innermost = "W"
            loop_w_iterations -= 1
    else:
        innermost = "KH"
        loop_kh_iterations -= 1

    addr_incr_end_window = (stride_w - 1) * channel_factor + 1
    addr_incr_end_row = ((skip_columns + (kernel_width - 1)) * channel_factor + 1) + (
        (stride_h - 1) * w * channel_factor
    )

    return {
        "LOOP_H_ITERATIONS": loop_h_iterations - 2,
        "LOOP_W_ITERATIONS": loop_w_iterations - 2,
        "LOOP_KH_ITERATIONS": loop_kh_iterations - 2,
        "LOOP_KW_ITERATIONS": loop_kw_iterations - 2,
        "LOOP_SIMD_ITERATIONS": loop_simd_iterations - 2,
        "HEAD_INCR_SIMD": 1,
        "HEAD_INCR_KW": 1,
        "HEAD_INCR_KH": 1,
        "HEAD_INCR_W": addr_incr_end_window,
        "HEAD_INCR_H": addr_incr_end_row,
        "TAIL_INCR_W": 0,
        "TAIL_INCR_H": 0,
        "TAIL_INCR_LAST": 0,
        "IS_DEPTHWISE": 0,
        "INNERMOST_STATE": innermost,
        "LAST_READ_ELEM": h * w * channel_factor - 1,
        "LAST_WRITE_ELEM": ((h - skip_rows - 1) * w + (w - skip_columns)) * channel_factor - 1,
        "FIRST_WRITE_ELEM": buffer_min_size - 1,
    }


def swg_parallel_schedule(p, n_feature_maps=4, hard_limit=None):
    """Per-cycle (input transaction, output transaction) of the parallel-style SWG.

    Same convention as swg_default_schedule. With out_V_V_TREADY tied high the
    ``Write_done`` register can never set -- ``advance`` is asserted in every
    cycle in which ``write_ok`` is -- so the output transaction is simply
    ``write_cmd``.
    """
    LAST_READ = p["LAST_READ_ELEM"]
    LAST_WRITE = p["LAST_WRITE_ELEM"]
    FIRST_WRITE = p["FIRST_WRITE_ELEM"]
    INNER = p["INNERMOST_STATE"]
    HEAD = {
        "START": 0,
        "SIMD": p["HEAD_INCR_SIMD"],
        "KW": p["HEAD_INCR_KW"],
        "KH": p["HEAD_INCR_KH"],
        "W": p["HEAD_INCR_W"],
        "H": p["HEAD_INCR_H"],
    }
    IT_H = p["LOOP_H_ITERATIONS"]
    IT_W = p["LOOP_W_ITERATIONS"]
    IT_KH = p["LOOP_KH_ITERATIONS"]
    IT_KW = p["LOOP_KW_ITERATIONS"]
    IT_SIMD = p["LOOP_SIMD_ITERATIONS"]

    state = INNER
    c_h, c_w, c_kh, c_kw, c_simd = IT_H, IT_W, IT_KH, IT_KW, IT_SIMD
    newest, current, writing_done = -1, FIRST_WRITE, 0

    if hard_limit is None:
        hard_limit = 64 * (LAST_READ + 1) + 4096

    sched = []
    restarts = []
    cycle = 0
    while cycle < hard_limit and len(restarts) < n_feature_maps:
        write_cmd = (current <= newest) and not writing_done
        write_ok = write_cmd  # out_V_V_TREADY tied high, Write_done never sets
        reading_done = newest == LAST_READ
        read_ok = (not reading_done) and (writing_done or newest <= current)

        addr_incr = HEAD[state]

        if state != INNER:
            state_next = INNER
        elif c_simd < 0:
            state_next = (
                "KW"
                if c_kw >= 0
                else ("KH" if c_kh >= 0 else "W" if c_w >= 0 else "H" if c_h >= 0 else "START")
            )
        else:
            state_next = state

        sched.append((int(read_ok), int(write_cmd)))

        n_newest, n_current, n_writing = newest, current, writing_done
        restarted = False

        if read_ok:
            n_newest = newest + 1
            if newest == LAST_READ - 1 and writing_done:
                n_newest, n_current, n_writing = -1, FIRST_WRITE, 0
                restarted = True

        if write_ok:
            if current == LAST_WRITE:
                n_writing = 1
                if reading_done or (read_ok and newest == LAST_READ - 1):
                    n_newest, n_current, n_writing = -1, FIRST_WRITE, 0
                    restarted = True
            else:
                n_current = current + addr_incr

        newest, current, writing_done = n_newest, n_current, n_writing

        # advance_controller is write_ok for the parallel style
        if write_ok:
            if state == INNER:
                if c_simd >= 0:
                    c_simd -= 1
                else:
                    c_simd = IT_SIMD
                    if c_kw >= 0:
                        c_kw -= 1
                    else:
                        c_kw = IT_KW
                        if c_kh >= 0:
                            c_kh -= 1
                        else:
                            c_kh = IT_KH
                            if c_w >= 0:
                                c_w -= 1
                            else:
                                c_w = IT_W
                                if c_h >= 0:
                                    c_h -= 1
                                else:
                                    c_h = IT_H
            state = state_next

        if restarted:
            restarts.append(cycle)
        cycle += 1

    return sched, restarts


def swg_default_tree(inst):
    """Exact CharacteristicNode for this node, or None if the fast path does not apply.

    Applies to the RTL ConvolutionInputGenerator, in either implementation
    style. Declines ``dynamic_mode``, which ``SlidingWindowRtlModel`` leaves to
    the closed-form approximations.
    """
    if inst.get_nodeattr("dynamic_mode"):
        # the dynamic template takes its loop bounds from AXI-lite at runtime
        return None
    try:
        impl_style = inst.select_impl_style()
    except (AttributeError, AssertionError):
        return None
    if impl_style not in ("default", "parallel"):
        return None

    ifm_ch = inst.get_nodeattr("IFMChannels")
    simd = inst.get_nodeattr("SIMD")
    if simd <= 0 or ifm_ch % simd != 0:
        return None

    cache_key = (
        impl_style,
        ifm_ch,
        simd,
        tuple(inst.get_nodeattr("ConvKernelDim")),
        tuple(inst.get_nodeattr("IFMDim")),
        tuple(inst.get_nodeattr("Stride")),
        tuple(inst.get_nodeattr("Dilation")),
        int(inst.get_nodeattr("depthwise")),
    )
    cached = getattr(inst, "_swg_tree_cache", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1]

    if impl_style == "default":
        p = swg_default_params(
            ifm_ch,
            simd,
            inst.get_nodeattr("ConvKernelDim"),
            inst.get_nodeattr("IFMDim"),
            inst.get_nodeattr("Stride"),
            inst.get_nodeattr("Dilation"),
            inst.get_nodeattr("depthwise"),
            inst.get_buffer_depth(),
        )
        sched, restarts = swg_default_schedule(p, n_feature_maps=4)
    else:
        p = swg_parallel_params(
            ifm_ch,
            simd,
            inst.get_nodeattr("ConvKernelDim"),
            inst.get_nodeattr("IFMDim"),
            inst.get_nodeattr("Stride"),
            inst.get_nodeattr("Dilation"),
        )
        sched, restarts = swg_parallel_schedule(p, n_feature_maps=4)
    if len(restarts) < 4:
        # the FSM did not settle into a repeating schedule; fall back rather
        # than emit a vector of the wrong length
        return None
    period = restarts[3] - restarts[2]

    # rtlsim characterises a node by running several feature maps back to back
    # and keeping two whole periods out of the middle; the window it keeps
    # starts one cycle before a period boundary. Reproduce that phase exactly,
    # otherwise the vector is a correct schedule at the wrong offset, which
    # reads to the FIFO sizer as a delay that is not there.
    start = 2 * period - 1
    if start + period > len(sched):
        return None
    window = sched[start : start + period]

    phases = []
    for rw in window:
        if phases and phases[-1][1] == rw:
            phases[-1][0] += 1
        else:
            phases.append([1, rw])
    node = CharacteristicNode("swg_default_exact", [(cnt, list(rw)) for cnt, rw in phases], True)
    inst._swg_tree_cache = (cache_key, node)
    return node


@register("ConvolutionInputGenerator")
class SlidingWindowModel(TreeModel):
    def input_need(self, op, ind=0):
        """The words of output pixel (y, x) need the input up to the last pixel
        of its window."""
        h, w = op.get_nodeattr("IFMDim")
        k_h, k_w = op.get_nodeattr("ConvKernelDim")
        s_h, s_w = op.get_nodeattr("Stride")
        d_h, d_w = op.get_nodeattr("Dilation")
        oshape = op.get_folded_output_shape()
        oh, ow = oshape[1], oshape[2]
        words_in = int(np.prod(op.get_folded_input_shape()[:-1])) // (h * w)
        words_out = int(np.prod(oshape[:-1])) // (oh * ow)
        y, x = np.divmod(np.arange(oh * ow), ow)
        last = (y * s_h + (k_h - 1) * d_h) * w + x * s_w + (k_w - 1) * d_w
        return np.repeat((np.minimum(last, h * w - 1) + 1) * words_in, words_out)


@register("ConvolutionInputGenerator_rtl")
class SlidingWindowRtlModel(SlidingWindowModel):
    def schedule(self, op):
        """Execute the sliding-window FSM where possible; approximate where not.

        Two tiers, in the order they are tried:

        1. ``swg_default_tree`` -- the RTL generator's "default" and "parallel"
           styles are small deterministic state machines, and with the input
           always valid and the output always ready their schedule follows
           entirely from the generated parameters. Running that FSM reproduces
           the rtlsim token access vector exactly, so nothing below can improve
           on it and it is tried first.
        2. hand-derived closed forms for the configurations the FSM path
           declines, i.e. ``dynamic_mode``, whose loop bounds are only set over
           AXI-lite at runtime.

        The layering is necessary because the SWG is the one operator here whose
        schedule is genuinely irregular -- output tokens come in bursts whose
        spacing depends on where the window sits in the row -- so no single closed
        form covers every folding. A folding that neither tier describes returns
        the generic ``swg`` shape built at the end of this method.
        """
        exact = swg_default_tree(op)
        if exact is not None:
            return exact

        ifm_dim_y, ifm_dim_x = op.get_nodeattr("IFMDim")
        ifm_ch = op.get_nodeattr("IFMChannels")
        simd = op.get_nodeattr("SIMD")
        k_y, k_x = op.get_nodeattr("ConvKernelDim")
        stride_y, stride_x = op.get_nodeattr("Stride")
        dilation_y, dilation_x = op.get_nodeattr("Dilation")
        parallel_window = op.get_nodeattr("parallel_window")
        depthwise = op.get_nodeattr("depthwise")
        SF = ifm_ch // simd

        def mkleaf(name, phases):
            return CharacteristicNode(name, [(cnt, vals) for cnt, vals in phases if cnt > 0], True)

        # 1x1 pass-through / depthwise-equivalent
        if k_y == 1 and k_x == 1:
            n_tok = SF * ifm_dim_y * ifm_dim_x
            return CharacteristicNode(
                "k1_pass",
                [(1, [0, 1]), (1, [1, 0]), (n_tok - 1, [1, 1])],
                True,
            )

        # depthwise, default impl, k=2 s=2, channel_factor > 1
        if (
            parallel_window == 0
            and depthwise == 1
            and simd != ifm_ch
            and k_y == 2
            and k_x == 2
            and stride_y == 2
            and stride_x == 2
            and dilation_y == 1
            and dilation_x == 1
        ):
            ofm_dim_y = math.floor((ifm_dim_y - k_y) / stride_y) + 1
            n = ifm_dim_x * SF
            pair_cnt = (SF - 2) // 2
            w = ifm_dim_x
            c = SF + 2 - (w - 2) * pair_cnt

            prefix = mkleaf(
                "dw_prefix",
                [
                    (1, [0, 1]),
                    (2, [1, 0]),
                    (1, [1, 1]),
                    (SF - 1, [1, 0]),
                    (1, [1, 1]),
                    (n - SF - 1, [1, 0]),
                    (1, [1, 1]),
                    (SF - 1, [1, 0]),
                    (n - (w - 2) * pair_cnt, [1, 1]),
                ],
            )

            a_base = mkleaf(
                "dw_row_start_base",
                [
                    (1, [0, 1]),
                    (1, [1, 1]),
                    (3, [0, 1]),
                    (w, [1, 1]),
                ],
            )
            a_pair = mkleaf(
                "dw_row_start_pair",
                [
                    (2, [0, 1]),
                    (1, [1, 1]),
                    (3, [0, 1]),
                    (1, [1, 1]),
                    (3, [0, 1]),
                    (w, [1, 1]),
                ],
            )
            tail_mid_both = 2 * n - (1 + w) - pair_cnt * (w + 2) - (4 * SF - 3)
            tail_mid = mkleaf(
                "dw_row_tail_mid",
                [
                    (3 * SF - 3, [1, 0]),
                    (1, [1, 1]),
                    (SF - 1, [1, 0]),
                    (tail_mid_both, [1, 1]),
                ],
            )
            tail_last = mkleaf(
                "dw_row_tail_last",
                [
                    (3 * SF - 3, [1, 0]),
                    (1, [1, 1]),
                    (SF - 1, [1, 0]),
                    (tail_mid_both - c, [1, 1]),
                    (n + 2 * pair_cnt, [0, 1]),
                ],
            )

            middle = CharacteristicNode(
                "dw_middle_row",
                [
                    (1, a_base),
                    (pair_cnt, a_pair),
                    (1, tail_mid),
                ],
                False,
            )
            last = CharacteristicNode(
                "dw_last_row",
                [
                    (1, a_base),
                    (pair_cnt, a_pair),
                    (1, tail_last),
                ],
                False,
            )

            return CharacteristicNode(
                "dw_pw0_k2s2",
                [
                    (1, prefix),
                    (ofm_dim_y - 2, middle),
                    (1, last),
                ],
                False,
            )

        # Special stride-3 pw=1 non-depthwise branch.
        if (
            parallel_window == 1
            and depthwise == 0
            and k_y == 2
            and k_x == 2
            and stride_y == 3
            and stride_x == 3
            and dilation_y == 1
            and dilation_x == 1
            and SF == 1
        ):
            ofm_dim_y = math.floor((ifm_dim_y - k_y) / stride_y) + 1
            ofm_dim_x = math.floor((ifm_dim_x - k_x) / stride_x) + 1
            first_valid_read = (k_y - 1) * ifm_dim_x + (k_x - 1) + 1
            prefix = first_valid_read
            gap_x = stride_x - 1
            row_gap = stride_y * ifm_dim_x - (ofm_dim_x - 1) * stride_x - 1

            ch_both = mkleaf("both", [(1, [1, 1])])
            ch_read = mkleaf("read", [(1, [1, 0])])
            step = CharacteristicNode("step", [(1, ch_both), (gap_x, ch_read)], False)
            full_row = CharacteristicNode(
                "full_row",
                [(ofm_dim_x - 1, step), (1, ch_both), (row_gap, ch_read)],
                False,
            )
            last_row = CharacteristicNode(
                "last_row",
                [(ofm_dim_x - 1, step)],
                False,
            )
            return CharacteristicNode(
                "pw1_k2_s3",
                [(1, ch_both), (prefix, ch_read), (ofm_dim_y - 1, full_row), (1, last_row)],
                False,
            )

        # Generic parallel_window=1 model for remaining multi-tap cases.
        if parallel_window == 1:
            eff_k_y = k_y + (k_y - 1) * (dilation_y - 1)
            eff_k_x = k_x + (k_x - 1) * (dilation_x - 1)
            ofm_dim_y = math.floor((ifm_dim_y - eff_k_y) / stride_y) + 1
            ofm_dim_x = math.floor((ifm_dim_x - eff_k_x) / stride_x) + 1

            start_y = eff_k_y - 1
            start_x = eff_k_x - 1
            top_zero = start_y * ifm_dim_x * SF
            left_zero = start_x * SF
            burst = SF
            gap_x = (stride_x - 1) * SF
            last_valid_x = start_x + (ofm_dim_x - 1) * stride_x
            tail_x = (ifm_dim_x - 1 - last_valid_x) * SF
            between_rows_zero = (stride_y - 1) * ifm_dim_x * SF
            trailing_rows_zero = (
                (ifm_dim_y - 1 - (start_y + (ofm_dim_y - 1) * stride_y)) * ifm_dim_x * SF
            )
            carry_write = 1 if (tail_x == 0 and trailing_rows_zero == 0) else 0

            def build_row(trim_last):
                phases = []
                if left_zero > 0:
                    phases.append((left_zero, [1, 0]))
                for ox in range(ofm_dim_x):
                    burst_cnt = burst
                    if (
                        trim_last
                        and trailing_rows_zero == 0
                        and tail_x == 0
                        and ox == ofm_dim_x - 1
                    ):
                        burst_cnt -= 1
                    if burst_cnt > 0:
                        phases.append((burst_cnt, [1, 1]))
                    if ox < ofm_dim_x - 1 and gap_x > 0:
                        phases.append((gap_x, [1, 0]))
                tail_cnt = tail_x
                if trim_last and trailing_rows_zero == 0 and tail_x > 0:
                    tail_cnt -= 1
                if tail_cnt > 0:
                    phases.append((tail_cnt, [1, 0]))
                return mkleaf("row", phases)

            row_full = build_row(False)
            row_short = build_row(True)
            top_zero_leaf = mkleaf("top_zero", [(top_zero, [1, 0])])
            between_rows_leaf = mkleaf("between_rows", [(between_rows_zero, [1, 0])])
            trailing_rows_leaf = mkleaf("trailing_rows", [(trailing_rows_zero - 1, [1, 0])])
            carry_leaf = mkleaf("carry", [(1, [0, carry_write])])
            bubble_leaf = mkleaf("bubble", [(1, [1, 0])])

            common_row_parts = [(1, row_full)]
            if between_rows_zero > 0:
                common_row_parts.append((1, between_rows_leaf))
            common_row = CharacteristicNode("common_row", common_row_parts, False)

            if trailing_rows_zero > 0:
                last_part_children = [(1, row_full)]
                if trailing_rows_zero - 1 > 0:
                    last_part_children.append((1, trailing_rows_leaf))
                last_part = CharacteristicNode("last_part", last_part_children, False)
            else:
                last_part = row_short

            scan_children = []
            if top_zero > 0:
                scan_children.append((1, top_zero_leaf))
            if ofm_dim_y > 1:
                scan_children.append((ofm_dim_y - 1, common_row))
            scan_children.append((1, last_part))
            scan_prefix = CharacteristicNode("scan_prefix", scan_children, False)

            return CharacteristicNode(
                "pw1_generic",
                [(1, carry_leaf), (1, bubble_leaf), (1, scan_prefix)],
                False,
            )

        # k=[2,2] pw=0 stride=[2,2] dw=0
        if (
            k_y == 2
            and k_x == 2
            and parallel_window == 0
            and depthwise == 0
            and stride_y == 2
            and stride_x == 2
        ):
            n_pix = ifm_dim_y * ifm_dim_x
            kernel_lines = math.ceil((ifm_dim_y - k_y + 1) / stride_y)

            main_both = SF * n_pix - 2 - 2 * SF - 4 * SF
            trailing_writes = SF * (kernel_lines - 1) + 1

            swg = CharacteristicNode(
                "k=2x2 pw=0 s=2x2",
                [
                    (1, [0, 1]),
                    (2, [1, 0]),
                    (2 * SF, [1, 1]),
                    (4 * SF, [1, 0]),
                    (main_both, [1, 1]),
                    (trailing_writes, [0, 1]),
                ],
                True,
            )
            return swg

        # Baseline branch for remaining pw=0 / other special cases
        stride_y_skips = (stride_y - 1) * ifm_dim_x

        kernels_in_line = math.ceil(
            (ifm_dim_x - (k_x - 1 + (k_x - 1) * (dilation_x - 1))) / stride_x
        )
        kernel_lines = math.ceil(
            (ifm_dim_y - ((k_y - 1) + (k_y - 1) * (dilation_y - 1))) / stride_y
        )

        shifts_x = (kernels_in_line - 1) * stride_x
        starting_index_x = k_x + (k_x - 1) * (dilation_x - 1)
        remainder_x = ifm_dim_x - (starting_index_x + shifts_x)

        shifts_y = (kernel_lines - 1) * stride_y
        starting_index_y = k_y + (k_y - 1) * (dilation_y - 1)
        remainder_y = (ifm_dim_y - (starting_index_y + shifts_y)) * ifm_dim_x

        reads_to_prepare_line = (k_x - 1) + (k_x - 1) * (dilation_x - 1)
        reads_to_prepare_first_line = ((k_y - 1) + (k_y - 1) * (dilation_y - 1)) * ifm_dim_x
        total_kernel_y = k_y + (k_y - 1) * (dilation_y - 1)
        first_line_kernel_buffer = k_x + (k_x - 1) * (dilation_x - 1)
        first_line_buffer = (total_kernel_y - 1) * ifm_dim_x

        if parallel_window == 1:
            writes_per_kernel = 1
        else:
            writes_per_kernel = k_y * k_x

        inner_line_buffer_reads = (stride_y - 1) * ifm_dim_x

        single_move_dif = writes_per_kernel - stride_x
        if single_move_dif > 0:
            do_both = stride_x
            writes_only = single_move_dif
            reads_only = 0
        else:
            do_both = writes_per_kernel
            reads_only = -single_move_dif
            writes_only = 0

        first_do_both = 0
        first_writes_only = writes_per_kernel
        first_reads_only = first_line_kernel_buffer

        absorbing_kernels = 0

        remaining_buffer_reads = inner_line_buffer_reads
        if inner_line_buffer_reads > 0 and ((kernels_in_line - 1) * writes_only) > 0:
            absorbing_kernels = min(
                math.floor((inner_line_buffer_reads) // writes_only), kernels_in_line - 1
            )
            absorbed_reads = absorbing_kernels * writes_only
            inner_line_buffer_reads -= absorbed_reads
            remaining_buffer_reads -= absorbed_reads

        first_reads = first_line_kernel_buffer + remaining_buffer_reads
        first_single_move_dif = writes_per_kernel - first_reads
        if first_single_move_dif > 0:
            first_do_both = first_reads
            first_writes_only = first_single_move_dif
            first_reads_only = 0
        else:
            first_do_both = writes_per_kernel
            first_reads_only = -first_single_move_dif
            first_writes_only = 0

        absolute_first_reads = first_line_kernel_buffer + first_line_buffer
        absolute_first_single_move_dif = writes_per_kernel - absolute_first_reads

        absolute_first_do_both = 0
        absolute_first_writes_only = writes_per_kernel
        absolute_first_reads_only = absolute_first_reads

        if depthwise == 0:
            if absolute_first_single_move_dif > 0:
                absolute_first_do_both = absolute_first_reads
                absolute_first_writes_only = absolute_first_single_move_dif
                absolute_first_reads_only = 0
            else:
                absolute_first_do_both = writes_per_kernel
                absolute_first_reads_only = -absolute_first_single_move_dif
                absolute_first_writes_only = 0

        ch_idle = CharacteristicNode("Output Write", [(SF, [0, 0])], True)
        ch_write = CharacteristicNode("Output Write", [(SF, [0, 1])], True)

        ch_read = CharacteristicNode("Streamed Read", [(SF, [1, 0])], True)
        ch_both = CharacteristicNode("Streamed Read+Write", [(SF, [1, 1])], True)

        if parallel_window == 2:
            ch_handle = CharacteristicNode("write out", [(1, ch_both)], False)

            handle_kernel = CharacteristicNode(
                "handle one kernel", [(1, ch_handle), (stride_x - 1, ch_read)], False
            )

            handle_last_kernel = CharacteristicNode(
                "handle last kernel",
                [
                    (1, ch_handle),
                    (remainder_x, ch_read),
                ],
                False,
            )

            handle_line = CharacteristicNode(
                "write_one_line",
                [
                    (reads_to_prepare_line, ch_read),
                    (kernels_in_line - 1, handle_kernel),
                    (1, handle_last_kernel),
                    (stride_y_skips, ch_read),
                ],
                False,
            )
            handle_last_line = CharacteristicNode(
                "write line without stride at end",
                [
                    (reads_to_prepare_line, ch_read),
                    (kernels_in_line, handle_kernel),
                    (remainder_y, ch_read),
                ],
                False,
            )
            swg = CharacteristicNode(
                "SlidingWindowGenerator",
                [
                    (1, ch_idle),
                    (reads_to_prepare_first_line, ch_read),
                    (kernel_lines - 1, handle_line),
                    (1, handle_last_line),
                ],
                False,
            )

        else:
            handle_absolute_kernel = CharacteristicNode(
                "handle one kernel",
                [
                    (absolute_first_do_both, ch_both),
                    (absolute_first_reads_only, ch_read),
                    (absolute_first_writes_only, ch_write),
                ],
                False,
            )

            handle_first_kernel = CharacteristicNode(
                "handle one kernel",
                [
                    (first_do_both, ch_both),
                    (first_reads_only, ch_read),
                    (first_writes_only, ch_write),
                ],
                False,
            )

            handle_kernel = CharacteristicNode(
                "handle one kernel",
                [
                    (do_both, ch_both),
                    (reads_only, ch_read),
                    (writes_only, ch_write),
                ],
                False,
            )

            handle_kernel_absorbed = CharacteristicNode(
                "handle one kernel with fused writes",
                [
                    (do_both + writes_only, ch_both),
                    (reads_only, ch_read),
                ],
                False,
            )

            handle_first_line = CharacteristicNode(
                "write first line",
                [
                    (1, handle_absolute_kernel),
                    (kernels_in_line - 1, handle_kernel),
                    (remainder_x, ch_read),
                ],
                False,
            )

            handle_line = CharacteristicNode(
                "write one inner line",
                [
                    (1, handle_first_kernel),
                    (absorbing_kernels, handle_kernel_absorbed),
                    (kernels_in_line - 1 - absorbing_kernels, handle_kernel),
                    (remainder_x, ch_read),
                ],
                False,
            )

            swg = CharacteristicNode(
                "SlidingWindowGenerator",
                [
                    (1, handle_first_line),
                    (kernel_lines - 1, handle_line),
                    (remainder_y, ch_read),
                ],
                False,
            )

        return swg
