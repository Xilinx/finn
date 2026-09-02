# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""
PWPolyFunction ONNX custom op for piecewise polynomial activations.

This module provides the ONNX graph execution layer for PWPolyF, analogous to
how qonnx.custom_op.general provides execution for Brevitas-exported ops.
"""

import numpy as np
from qonnx.core.datatype import DataType
from qonnx.custom_op.base import CustomOp

# Constants matching the SystemVerilog pwpolyf module
NUM_OCTAVES = 5
EXP_BIAS = 127
EXP_BASE = 125
EXP_CLAMP = 130

SUPPORTED_FUNCS = ("gelu", "silu", "sigmoid", "tanh")

# ONNX domain and opset for exported PWPolyF nodes
PWPOLYF_ONNX_DOMAIN = "finn.pwpolyf"
PWPOLYF_ONNX_OPSET = 1

# Clamping configuration per activation function
CLAMP_CFG = {
    "gelu": {"neg_clamp": 0.0, "pos_clamp": 0.0, "pos_passthrough": True},
    "silu": {"neg_clamp": 0.0, "pos_clamp": 0.0, "pos_passthrough": True},
    "sigmoid": {"neg_clamp": 0.0, "pos_clamp": 1.0, "pos_passthrough": False},
    "tanh": {"neg_clamp": -1.0, "pos_clamp": 1.0, "pos_passthrough": False},
}


def _segment_boundaries(K):
    """Return (lo, hi) bounds for every PWPolyF segment.

    Args:
        K: Number of mantissa subdivision bits.

    Returns:
        List of (lo, hi) tuples defining segment boundaries.
    """
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


def _reference_func_numpy(func_name, x):
    """Evaluate reference activation function using numpy.

    Args:
        func_name: One of "gelu", "silu", "sigmoid", "tanh".
        x: Input numpy array.

    Returns:
        Output numpy array.
    """
    if func_name == "gelu":
        # GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
        return x * 0.5 * (1.0 + np.vectorize(np.math.erf)(x / np.sqrt(2.0)))
    elif func_name == "silu":
        # SiLU: x * sigmoid(x)
        return x / (1.0 + np.exp(-x))
    elif func_name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    elif func_name == "tanh":
        return np.tanh(x)
    else:
        raise ValueError(f"Unknown function: {func_name}")


def _fit_coefficients(func_name, K, degree=2, num_samples=1000):
    """Fit degree-N polynomials per segment.

    Args:
        func_name: Activation function name.
        K: Number of mantissa subdivision bits.
        degree: Polynomial degree.
        num_samples: Samples per segment for fitting.

    Returns:
        Numpy array of shape (num_segments, degree+1) with coefficients.
    """
    bounds = _segment_boundaries(K)
    num_segs = len(bounds)
    coeffs = np.zeros((num_segs, degree + 1), dtype=np.float64)

    for seg, (lo, hi) in enumerate(bounds):
        xs = np.linspace(lo, hi, num_samples, dtype=np.float64)
        ys = _reference_func_numpy(func_name, xs)
        c = np.polynomial.polynomial.polyfit(xs, ys, deg=degree)
        coeffs[seg] = c[: degree + 1]

    return coeffs.astype(np.float32)


def _segment_index_numpy(x, K):
    """Map each element to its polynomial segment using numpy.

    Args:
        x: Input numpy array (flattened).
        K: Number of mantissa subdivision bits.

    Returns:
        Tuple of (seg_idx, is_neg_clamp, is_pos_clamp) numpy arrays.
    """
    num_subs = 1 << K
    num_segs = 1 + 2 * NUM_OCTAVES * num_subs

    abs_x = np.abs(x)
    is_neg = x < 0

    is_near_zero = abs_x < 0.25
    is_clamp = abs_x >= 8.0
    is_neg_clamp = is_neg & is_clamp
    is_pos_clamp = (~is_neg) & is_clamp

    safe_abs = np.clip(abs_x, 0.25, None)
    floor_log2 = np.floor(np.log2(safe_abs))
    octave = np.clip((floor_log2 + 2).astype(np.int64), 0, NUM_OCTAVES - 1)

    pow2 = np.exp2(floor_log2)
    frac = safe_abs / pow2 - 1.0
    sub = np.clip((frac * num_subs).astype(np.int64), 0, num_subs - 1)

    pos_idx = 1 + octave * num_subs + sub
    neg_idx = 1 + NUM_OCTAVES * num_subs + octave * num_subs + sub

    seg_idx = np.where(is_near_zero, 0, np.where(is_neg, neg_idx, pos_idx))
    seg_idx = np.clip(seg_idx, 0, num_segs - 1)

    return seg_idx, is_neg_clamp, is_pos_clamp


def _horner_eval_numpy(x, coeffs, seg_idx, degree, func_name):
    """Evaluate piecewise polynomial using Horner's method in numpy.

    Args:
        x: Input values (flattened numpy array).
        coeffs: Coefficient array of shape (num_segs, degree+1).
        seg_idx: Segment indices for each input element.
        degree: Polynomial degree.
        func_name: Activation function name (for clamping config).

    Returns:
        Output numpy array.
    """
    c = coeffs[seg_idx]

    # Horner evaluation: y = c0 + x*(c1 + x*(c2 + ...))
    y = c[:, degree].copy()
    for i in range(degree - 1, -1, -1):
        y = c[:, i] + x * y

    return y


class PWPolyFunction(CustomOp):
    """ONNX custom op for piecewise polynomial activation functions.

    This op is exported by the PWPolyFActivation PyTorch module and can be
    executed during ONNX graph simulation. It approximates GELU, SiLU,
    Sigmoid, and Tanh using piecewise polynomials.
    """

    def get_nodeattr_types(self):
        return {
            "func": ("s", True, ""),
            "K": ("i", False, 3),
            "degree": ("i", False, 2),
        }

    def make_shape_compatible_op(self, model):
        """Return a standard op that produces the same output shape."""
        node = self.onnx_node
        return super().make_const_shape_op(model.get_tensor_shape(node.input[0]))

    def infer_node_datatype(self, model):
        """Infer and set output datatype (always FLOAT32)."""
        node = self.onnx_node
        model.set_tensor_datatype(node.output[0], DataType["FLOAT32"])

    def execute_node(self, context, graph):
        """Execute the PWPolyF operation using numpy."""
        node = self.onnx_node
        inp = context[node.input[0]]

        func = self.get_nodeattr("func")
        K = self.get_nodeattr("K")
        degree = self.get_nodeattr("degree")

        cfg = CLAMP_CFG[func]

        # Fit coefficients (could be cached for performance)
        coeffs = _fit_coefficients(func, K, degree=degree)

        # Flatten input for processing
        orig_shape = inp.shape
        x_flat = inp.flatten().astype(np.float32)

        # Get segment indices
        seg_idx, is_neg_clamp, is_pos_clamp = _segment_index_numpy(x_flat, K)

        # Horner evaluation
        y = _horner_eval_numpy(x_flat, coeffs, seg_idx, degree, func)

        # Apply clamping
        if cfg["pos_passthrough"]:
            y = np.where(is_pos_clamp, x_flat, y)
        else:
            y = np.where(is_pos_clamp, cfg["pos_clamp"], y)
        y = np.where(is_neg_clamp, cfg["neg_clamp"], y)

        context[node.output[0]] = y.reshape(orig_shape).astype(np.float32)

    def verify_node(self):
        """Verify node attributes are valid."""
        info_messages = []

        func = self.get_nodeattr("func")
        if func in SUPPORTED_FUNCS:
            info_messages.append("Attribute func is set correctly")
        else:
            info_messages.append(
                "Attribute func must be one of %s, got %s" % (SUPPORTED_FUNCS, func)
            )

        return info_messages
