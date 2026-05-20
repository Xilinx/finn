# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""
PyTorch modules that match FINN hardware-layer behavior.

These modules are intended as drop-in PyTorch layers for modelling the
functional behavior of FINN hardware layers before conversion to HWCustomOps.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants matching the SystemVerilog pwpolyf module
NUM_OCTAVES = 5
EXP_BIAS = 127
EXP_BASE = 125
EXP_CLAMP = 130

SUPPORTED_FUNCS = ("gelu", "silu", "sigmoid", "tanh")

REFERENCE_FUNCS = {
    "gelu": lambda x: F.gelu(x),
    "silu": lambda x: F.silu(x),
    "sigmoid": lambda x: torch.sigmoid(x),
    "tanh": lambda x: torch.tanh(x),
}

CLAMP_CFG = {
    "gelu": {"neg_clamp": 0.0, "pos_clamp": 0.0, "pos_passthrough": True},
    "silu": {"neg_clamp": 0.0, "pos_clamp": 0.0, "pos_passthrough": True},
    "sigmoid": {"neg_clamp": 0.0, "pos_clamp": 1.0, "pos_passthrough": False},
    "tanh": {"neg_clamp": -1.0, "pos_clamp": 1.0, "pos_passthrough": False},
}


def _segment_boundaries(K):
    """Return (lo, hi) bounds for every PWPolyF segment."""
    num_subs = 1 << K
    bounds = []

    # Segment 0: near-zero
    bounds.append((-0.25, 0.25))

    # Positive segments
    for octave in range(NUM_OCTAVES):
        exp_val = EXP_BASE + octave - EXP_BIAS
        base = 2.0**exp_val
        for sub in range(num_subs):
            lo = base * (1.0 + sub / num_subs)
            hi = base * (1.0 + (sub + 1) / num_subs)
            bounds.append((lo, hi))

    # Negative segments (mirror of positive)
    for octave in range(NUM_OCTAVES):
        exp_val = EXP_BASE + octave - EXP_BIAS
        base = 2.0**exp_val
        for sub in range(num_subs):
            lo = base * (1.0 + sub / num_subs)
            hi = base * (1.0 + (sub + 1) / num_subs)
            bounds.append((-hi, -lo))

    return bounds


def _fit_coefficients(func_name, K, degree=2, num_samples=1000):
    """Fit degree-N polynomials per segment. Returns a (segments, degree+1) tensor."""
    ref_fn = REFERENCE_FUNCS[func_name]
    bounds = _segment_boundaries(K)
    num_segs = len(bounds)
    coeffs = np.zeros((num_segs, degree + 1), dtype=np.float64)

    for seg, (lo, hi) in enumerate(bounds):
        xs = np.linspace(lo, hi, num_samples, dtype=np.float64)
        with torch.no_grad():
            ys = ref_fn(torch.from_numpy(xs).float()).numpy().astype(np.float64)
        c = np.polynomial.polynomial.polyfit(xs, ys, deg=degree)
        coeffs[seg] = c[: degree + 1]

    return torch.from_numpy(coeffs.astype(np.float32))


def _segment_index(x, K, num_subs, num_segs):
    """Map each element to its polynomial segment, mirroring SV addressing."""
    abs_x = x.abs()
    is_neg = x < 0

    is_near_zero = abs_x < 0.25
    is_clamp = abs_x >= 8.0
    is_neg_clamp = is_neg & is_clamp
    is_pos_clamp = (~is_neg) & is_clamp

    safe_abs = abs_x.clamp(min=0.25)
    floor_log2 = torch.floor(torch.log2(safe_abs))
    octave = (floor_log2 + 2).long().clamp(0, NUM_OCTAVES - 1)

    pow2 = torch.exp2(floor_log2)
    frac = safe_abs / pow2 - 1.0
    sub = (frac * num_subs).long().clamp(0, num_subs - 1)

    pos_idx = 1 + octave * num_subs + sub
    neg_idx = 1 + NUM_OCTAVES * num_subs + octave * num_subs + sub

    seg_idx = torch.where(
        is_near_zero,
        torch.zeros_like(pos_idx),
        torch.where(is_neg, neg_idx, pos_idx),
    )
    seg_idx = seg_idx.clamp(0, num_segs - 1)

    return seg_idx, is_neg_clamp, is_pos_clamp


class PWPolyFFunction(torch.autograd.Function):
    """Emit a single PWPolyF ONNX node during legacy torch.onnx export."""

    @staticmethod
    def forward(ctx, x, coeffs, neg_clamp_val, pos_clamp_val, func, K, degree):
        num_subs = 1 << K
        num_segs = 1 + 2 * NUM_OCTAVES * num_subs
        degree = int(degree)
        pos_passthrough = CLAMP_CFG[func]["pos_passthrough"]

        orig_shape = x.shape
        x_flat = x.contiguous().view(-1)

        seg_idx, is_neg_clamp, is_pos_clamp = _segment_index(x_flat, K, num_subs, num_segs)

        c = coeffs[seg_idx]
        # Horner evaluation: y = c0 + x*(c1 + x*(c2 + ...))
        y = c[:, degree]
        for i in range(degree - 1, -1, -1):
            y = c[:, i] + x_flat * y

        if pos_passthrough:
            pos_val = x_flat
        else:
            pos_val = pos_clamp_val.expand_as(y)
        y = torch.where(is_pos_clamp, pos_val, y)
        y = torch.where(is_neg_clamp, neg_clamp_val.expand_as(y), y)

        return y.view(orig_shape)

    @staticmethod
    def symbolic(g, x, coeffs, neg_clamp_val, pos_clamp_val, func, K, degree):
        return g.op("PWPolyF", x, func_s=func, K_i=K, degree_i=degree)


class PiecewisePolyActivation(nn.Module):
    """
    Drop-in activation matching FINN's PWPolyF RTL behavior.

    Approximates nonlinear activations using piecewise polynomials over
    segments defined by FP32 bit extraction. The polynomial is evaluated via
    Horner's method to match the DSPFP32 FMA chain used by the RTL.
    """

    def __init__(self, func="gelu", K=3, degree=2, fit_samples=1000):
        super().__init__()
        if func not in SUPPORTED_FUNCS:
            raise ValueError("Unsupported func=%r; choose from %s" % (func, SUPPORTED_FUNCS))

        self.func = func
        self.K = K
        self.degree = degree
        self.num_subs = 1 << K
        self.num_segs = 1 + 2 * NUM_OCTAVES * self.num_subs
        self.pos_passthrough = CLAMP_CFG[func]["pos_passthrough"]

        coeffs = _fit_coefficients(func, K, degree=degree, num_samples=fit_samples)
        self.register_buffer("coeffs", coeffs)

        neg_cv = torch.tensor(CLAMP_CFG[func]["neg_clamp"], dtype=torch.float32)
        pos_cv = torch.tensor(CLAMP_CFG[func]["pos_clamp"], dtype=torch.float32)
        self.register_buffer("neg_clamp_val", neg_cv)
        self.register_buffer("pos_clamp_val", pos_cv)

    def forward(self, x):
        if torch.onnx.is_in_onnx_export():
            return PWPolyFFunction.apply(
                x,
                self.coeffs,
                self.neg_clamp_val,
                self.pos_clamp_val,
                self.func,
                self.K,
                self.degree,
            )

        orig_shape = x.shape
        x_flat = x.contiguous().view(-1)

        seg_idx, is_neg_clamp, is_pos_clamp = _segment_index(
            x_flat, self.K, self.num_subs, self.num_segs
        )

        c = self.coeffs[seg_idx]
        # Horner evaluation: y = c0 + x*(c1 + x*(c2 + ...))
        y = c[:, self.degree]
        for i in range(self.degree - 1, -1, -1):
            y = c[:, i] + x_flat * y

        if self.pos_passthrough:
            pos_val = x_flat
        else:
            pos_val = self.pos_clamp_val.expand_as(y)
        y = torch.where(is_pos_clamp, pos_val, y)
        y = torch.where(is_neg_clamp, self.neg_clamp_val.expand_as(y), y)

        return y.view(orig_shape)
