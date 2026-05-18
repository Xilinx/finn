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
