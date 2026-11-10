# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Elaboration-time quantities and schedules of the loop-nest-driven sliding
window generator ``input_gen`` (finn-hlslib ``input_gen.hpp``,
``finn-rtllib/input_gen/hdl/input_gen.sv``), mirrored in Python for resource
estimation and FIFO sizing."""

import numpy as np


def input_gen_r_flags(fm_size, dims, coefs):
    """Free-pointer responsibility flags (``R_FLAG``): entry i tells whether
    level i advances the free pointer; entry 0 (the frame) always does."""
    d = len(dims)
    w = [fm_size] + list(coefs)
    fw_if_r = [0] * (d + 1)
    for i in range(d - 1, -1, -1):
        fw = (dims[i] - 1) * coefs[i] + fw_if_r[i + 1]
        fw_if_r[i] = fw if coefs[i] > 0 and fw <= w[i] else 0
    r_flag = [True]
    for i in range(d):
        r_flag.append(
            r_flag[-1] and coefs[i] > 0 and (dims[i] - 1) * coefs[i] + fw_if_r[i + 1] <= w[i]
        )
    return r_flag


def input_gen_buffer_depth(fm_size, dims, coefs):
    """Buffer depth finn-rtllib/input_gen/hdl/input_gen.sv derives at
    elaboration for the loop nest (dims, coefs) over a feature map of fm_size
    elements. Mirrors the
    SystemVerilog functions R_FLAG, TERMINAL_RP_INC, TERMINAL_FP_INC,
    INIT_MAX_OCCUPANCY and INIT_THROUGHPUT_OCC; keep the two in sync."""
    d = len(dims)
    w = [fm_size] + list(coefs)
    r_flag = input_gen_r_flags(fm_size, dims, coefs)

    # terminal read/free pointer increments (free pointer stored negated)
    rp_inc = [0] * (d + 1)
    fp_inc = [0] * (d + 1)
    rw = fw = 0
    for i in range(d, -1, -1):
        if i < d:
            rw = (dims[i] - 1) * coefs[i] + rw
            fw = (dims[i] - 1) * coefs[i] + fw if r_flag[i + 1] else 0
        rp_inc[i] = w[i] - rw
        fp_inc[i] = fw - w[i] if r_flag[i] else 0

    # functional bound: max backward retraction and read-free pointer gap
    max_occupancy = max([0] + [-rp_inc[i] for i in range(d)])
    rw = fw = 0
    for i in range(d - 1, -1, -1):
        rw = (dims[i] - 1) * coefs[i] + rw
        fw = (dims[i] - 1) * coefs[i] + fw if r_flag[i + 1] else 0
        max_occupancy = max(max_occupancy, rw - fw)

    # throughput-preserving occupancy (INIT_THROUGHPUT_OCC)
    neg = -(1 << 60)

    def compose(b, n, j):
        tot, pre, suf, best = b
        if n > 1:
            step = max(0, (n - 1) * (tot + j))
            return (
                n * tot + (n - 1) * j,
                pre + step,
                suf + step,
                max(best, suf + j + pre + max(0, (n - 2) * (tot + j))),
            )
        return b

    q = max(i for i in range(d + 1) if r_flag[i])
    pq, delta, gapq = 1, 0, 0
    for i in range(d - 1, q - 1, -1):
        delta += (dims[i] - 1) * max(0, coefs[i] - pq)
        gapq += (dims[i] - 1) * coefs[i]
        pq *= dims[i]
    u = delta + pq - 1
    bp = bn = (0, 0, 0, neg)
    for i in range(q - 1, -1, -1):
        j = -fp_inc[i + 1] - pq
        bp = compose(bp, dims[i], j)
        bn = compose(bn, dims[i], -j)
    jf = -fp_inc[0] - pq
    if bp[0] + jf <= 0:
        occ = max(gapq, u + max(bp[3], bp[2] + jf + bp[1]))
    else:
        occ = u + max(0, max(bn[3], bn[2] - jf + bn[1]))

    # WP_DELAY = 1: functional floor max_occupancy + 3; stall-free occ + 4
    required_depth = max(occ + 4, max_occupancy + 3)
    return 1 << (required_depth - 1).bit_length()


def pool_schedule(
    ifm_dim, kernel, stride, cf, padding=(0, 0, 0, 0), frames=5, end_latency=7
):
    """Steady-state token schedule of the fused HLS pooling kernel
    ``[pad<>] -> input_gen<> -> pool<F>`` (``Pool_hls``) with input always valid
    and output always ready.

    The read schedule of ``input_gen`` with its stall-free buffer equals the
    unbounded one, tau_k = max(tau_{k-1} + 1, A_k + 3), a running maximum over
    the read addresses A_k. Input word y is accepted once the free pointer --
    the base address of the current unit of the deepest free-pointer level --
    has passed y - BUF_SIZE. pool<F> emits a window's result when it consumes
    the window's last element.

    Returns ``(rd, wr)``: 0/1 arrays over one steady-state period (the frame
    period, max(input words, reads) per frame), phased like the stored window of
    an rtlsim characterization over ``frames`` frames (from 2 * total // frames).
    """
    (h, w), (kh, kw), (sh, sw) = ifm_dim, kernel, stride
    hp, wp = h + padding[0] + padding[2], w + padding[1] + padding[3]
    oh, ow = (hp - kh) // sh + 1, (wp - kw) // sw + 1
    fm = hp * wp * cf
    dims = [oh, ow, cf, kh, kw]
    coefs = [sh * wp * cf, sw * cf, 1, wp * cf, cf]

    idx = np.indices(dims).reshape(len(dims), -1)
    frame_ofs = fm * np.arange(frames)[:, None]
    addr = (sum(c * i for c, i in zip(coefs, idx))[None, :] + frame_ofs).ravel()
    k = np.arange(addr.size)
    tau = k + np.maximum.accumulate(addr + 3 - k)

    # free pointer after each read: base of the unit holding the next read
    r_flag = input_gen_r_flags(fm, dims, coefs)
    q = max(i for i in range(len(dims) + 1) if r_flag[i])
    base = (sum(coefs[lvl] * idx[lvl] for lvl in range(q))[None, :] + frame_ofs).ravel()
    fp_after = np.append(base[1:], fm * frames)

    buf = input_gen_buffer_depth(fm, dims, coefs)
    y = np.arange(fm * frames)
    need = y - buf + 1
    release = np.where(
        need <= 0, 0, tau[np.minimum(np.searchsorted(fp_after, need), tau.size - 1)] + 1
    )
    write = y + np.maximum.accumulate(release - y)

    # node inputs are the interior (non-pad) words of the padded stream
    pix = np.arange(fm) // cf
    row, col = pix // wp, pix % wp
    interior = (
        (row >= padding[0]) & (row < padding[0] + h) & (col >= padding[1]) & (col < padding[1] + w)
    )
    t_in = write.reshape(frames, fm)[:, interior].ravel()
    t_out = tau[k % (kh * kw) == kh * kw - 1]

    n_in = h * w * cf
    period = int(t_in[(frames - 2) * n_in] - t_in[(frames - 3) * n_in])
    start = 2 * ((int(t_out[-1]) + 1 + end_latency) // frames)
    rd = np.bincount(t_in[(t_in >= start) & (t_in < start + period)] - start, minlength=period)
    wr = np.bincount(t_out[(t_out >= start) & (t_out < start + period)] - start, minlength=period)
    return rd[:period].astype(np.int8), wr[:period].astype(np.int8)
