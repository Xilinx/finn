# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""
FINN general-purpose ONNX custom ops.

This package contains custom ops that are used during ONNX graph execution
but are not hardware-specific (unlike finn.custom_op.fpgadataflow).

Domain Registration
-------------------
Brevitas export_qonnx exports all custom ops with domain "qonnx.custom_op.general".
For QONNX to find and execute these custom ops, we register them with QONNX's
registry at import time using add_op_to_domain().

The registration flow:
1. User imports a PyTorch module (e.g., PWPolyFActivation) from finn.util
2. That module imports from this package (finn.custom_op.general)
3. This __init__.py executes and registers all custom ops with QONNX's registry
4. QONNX can now find and execute these ops when loading/transforming models

Any constants or helpers needed by the PyTorch modules should be re-exported here
to ensure this __init__.py is always executed (triggering the registration).
"""

from qonnx.custom_op.base import CustomOp
from qonnx.custom_op.registry import add_op_to_domain

# Dictionary of CustomOp implementations
custom_op = dict()

# flake8: noqa
# Disable linting from here, as all imports will be flagged E402 and maybe F401

from finn.custom_op.general.pwpolyfunction import (
    CLAMP_CFG,
    NUM_OCTAVES,
    SUPPORTED_FUNCS,
    PWPolyFunction,
    _segment_boundaries,
)

custom_op["PWPolyFunction"] = PWPolyFunction

# Register with QONNX's registry (see module docstring for details)
add_op_to_domain("qonnx.custom_op.general", PWPolyFunction)
