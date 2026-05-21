# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility imports for PWPolyF PyTorch utilities.

The canonical home for PyTorch modules that match FINN hardware behavior is
``finn.util.torch_hw_modules``. This module is kept to avoid breaking existing
imports while downstream code moves to the new location.
"""

from finn.util.torch_hw_modules import (
    CLAMP_CFG,
    EXP_BASE,
    EXP_BIAS,
    EXP_CLAMP,
    NUM_OCTAVES,
    REFERENCE_FUNCS,
    SUPPORTED_FUNCS,
    PiecewisePolyActivation,
    PWPolyFFunction,
    _fit_coefficients,
    _segment_boundaries,
    _segment_index,
)

__all__ = [
    "CLAMP_CFG",
    "EXP_BIAS",
    "EXP_BASE",
    "EXP_CLAMP",
    "NUM_OCTAVES",
    "PWPolyFFunction",
    "PiecewisePolyActivation",
    "REFERENCE_FUNCS",
    "SUPPORTED_FUNCS",
    "_fit_coefficients",
    "_segment_boundaries",
    "_segment_index",
]
