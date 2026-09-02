# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from qonnx.custom_op.base import CustomOp

# Dictionary of CustomOp implementations
custom_op = dict()

# flake8: noqa
# Disable linting from here, as all imports will be flagged E402 and maybe F401

from finn.custom_op.general.pwpolyfunction import PWPolyFunction

custom_op["PWPolyFunction"] = PWPolyFunction
