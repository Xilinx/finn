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

# Import constants from the ONNX custom op package (importing from the package
# triggers registration of PWPolyFunction with QONNX's custom op registry)
from finn.custom_op.general import (
    CLAMP_CFG,
    NUM_OCTAVES,
    SUPPORTED_FUNCS,
    _segment_boundaries,
)

# PyTorch reference functions for coefficient fitting
REFERENCE_FUNCS = {
    "gelu": lambda x: F.gelu(x),
    "silu": lambda x: F.silu(x),
    "sigmoid": lambda x: torch.sigmoid(x),
    "tanh": lambda x: torch.tanh(x),
}


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


def _pwpolyf_eval(x, coeffs, neg_clamp_val, pos_clamp_val, func, K, degree):
    """Evaluate piecewise polynomial activation. Used by both Module and Function."""
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


class PWPolyFFunction(torch.autograd.Function):
    """Emit a single PWPolyFunction ONNX node during torch.onnx export."""

    @staticmethod
    def forward(ctx, x, coeffs, neg_clamp_val, pos_clamp_val, func, K, degree):
        return _pwpolyf_eval(x, coeffs, neg_clamp_val, pos_clamp_val, func, K, degree)

    @staticmethod
    def symbolic(g, x, coeffs, neg_clamp_val, pos_clamp_val, func, K, degree):
        # Use qonnx.custom_op.general domain to match Brevitas export convention
        # and enable QONNX's registry to find the PWPolyFunction custom op
        ret = g.op(
            "qonnx.custom_op.general::PWPolyFunction",
            x,
            func_s=func,
            K_i=K,
            degree_i=degree,
        )
        ret.setType(x.type())
        return ret


class PWPolyFActivation(nn.Module):
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
        return PWPolyFFunction.apply(
            x,
            self.coeffs,
            self.neg_clamp_val,
            self.pos_clamp_val,
            self.func,
            self.K,
            self.degree,
        )
