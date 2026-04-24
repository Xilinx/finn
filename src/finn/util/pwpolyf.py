# Copyright (C) 2026, Advanced Micro Devices, Inc.
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

"""
Piecewise polynomial activation - PyTorch module and software model.

Drop-in activation that approximates GELU, SiLU, Sigmoid, and Tanh using
degree-2 polynomials, matching the pwpolyf RTL behaviour.  Emits a single
PWPolyF custom op node during ONNX export (requires dynamo=False).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants matching the SystemVerilog module
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
    """Return (lo, hi) bounds for every segment."""
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


def _fit_coefficients(func_name, K, num_samples=1000):
    """Fit degree-2 polynomials per segment.  Returns (NUM_SEGS, 3) tensor."""
    ref_fn = REFERENCE_FUNCS[func_name]
    bounds = _segment_boundaries(K)
    num_segs = len(bounds)
    coeffs = np.zeros((num_segs, 3), dtype=np.float64)

    for seg, (lo, hi) in enumerate(bounds):
        xs = np.linspace(lo, hi, num_samples, dtype=np.float64)
        with torch.no_grad():
            ys = ref_fn(torch.from_numpy(xs).float()).numpy().astype(np.float64)
        c = np.polynomial.polynomial.polyfit(xs, ys, deg=2)
        coeffs[seg] = c[:3]

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
    """Emits a single PWPolyF ONNX node during export."""

    @staticmethod
    def forward(ctx, x, coeffs, neg_clamp_val, pos_clamp_val, func, K):
        num_subs = 1 << K
        num_segs = 1 + 2 * NUM_OCTAVES * num_subs
        pos_passthrough = CLAMP_CFG[func]["pos_passthrough"]

        orig_shape = x.shape
        x_flat = x.contiguous().view(-1)

        seg_idx, is_neg_clamp, is_pos_clamp = _segment_index(x_flat, K, num_subs, num_segs)

        c = coeffs[seg_idx]
        a0 = c[:, 0]
        a1 = c[:, 1]
        a2 = c[:, 2]

        y = a0 + x_flat * (a1 + a2 * x_flat)

        if pos_passthrough:
            pos_val = x_flat
        else:
            pos_val = pos_clamp_val.expand_as(y)
        y = torch.where(is_pos_clamp, pos_val, y)
        y = torch.where(is_neg_clamp, neg_clamp_val.expand_as(y), y)

        return y.view(orig_shape)

    @staticmethod
    def symbolic(g, x, coeffs, neg_clamp_val, pos_clamp_val, func, K):
        return g.op("PWPolyF", x, func_s=func, K_i=K)


class PiecewisePolyActivation(nn.Module):
    """
    Drop-in activation matching the pwpolyf hardware behaviour.

    Approximates nonlinear activations using degree-2 polynomials over
    segments defined by FP32 bit-extraction.  Evaluated via Horner's method.
    Emits a single PWPolyF custom op node during ONNX export.
    """

    def __init__(self, func="gelu", K=3, fit_samples=1000):
        super().__init__()
        if func not in SUPPORTED_FUNCS:
            raise ValueError("Unsupported func=%r; choose from %s" % (func, SUPPORTED_FUNCS))

        self.func = func
        self.K = K
        self.num_subs = 1 << K
        self.num_segs = 1 + 2 * NUM_OCTAVES * self.num_subs
        self.pos_passthrough = CLAMP_CFG[func]["pos_passthrough"]

        coeffs = _fit_coefficients(func, K, fit_samples)
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
            )

        orig_shape = x.shape
        x_flat = x.contiguous().view(-1)

        seg_idx, is_neg_clamp, is_pos_clamp = _segment_index(
            x_flat, self.K, self.num_subs, self.num_segs
        )

        c = self.coeffs[seg_idx]
        a0 = c[:, 0]
        a1 = c[:, 1]
        a2 = c[:, 2]

        # Horner: y = a0 + x*(a1 + a2*x)
        y = a0 + x_flat * (a1 + a2 * x_flat)

        if self.pos_passthrough:
            pos_val = x_flat
        else:
            pos_val = self.pos_clamp_val.expand_as(y)
        y = torch.where(is_pos_clamp, pos_val, y)
        y = torch.where(is_neg_clamp, self.neg_clamp_val.expand_as(y), y)

        return y.view(orig_shape)
