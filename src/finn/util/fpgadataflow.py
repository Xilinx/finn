############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

from qonnx.custom_op.registry import is_custom_op
from qonnx.util.basic import get_by_name

# Supported backend attribute values for fpgadataflow nodes
SUPPORTED_BACKENDS = {"fpgadataflow", "hls", "rtl"}


def _get_backend_value(node):
    """Helper to extract backend value from a node. Returns None if not found."""
    if node is None:
        return None
    n_backend = get_by_name(node.attribute, "backend")
    return n_backend.s.decode("UTF-8") if n_backend is not None else None


def is_fpgadataflow_node(node):
    """Returns True if given node has backend 'fpgadataflow', 'hls', or 'rtl'."""
    if is_custom_op(node.domain) is False:
        return False
    backend_value = _get_backend_value(node)
    return backend_value in SUPPORTED_BACKENDS


def is_backend_node(node, backend_name):
    """Returns True if given node is of specified backend."""
    if is_custom_op(node.domain) is False:
        return False

    backend_value = _get_backend_value(node)
    if backend_value is None:
        return False

    # Direct backend match
    if backend_value == backend_name:
        return True

    # Legacy approach: finn domain indicates implementation style
    if backend_value == "fpgadataflow":
        return node.domain == f"finn.custom_op.fpgadataflow.{backend_name}"

    return False


def is_hls_node(node):
    """Returns True if given node is hls node. Otherwise False."""
    return is_backend_node(node, "hls")


def is_rtl_node(node):
    """Returns True if given node is rtl node. Otherwise False."""
    return is_backend_node(node, "rtl")
