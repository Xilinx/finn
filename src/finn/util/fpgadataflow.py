############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

import os
import warnings
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp, is_custom_op
from qonnx.util.basic import get_by_name

# Supported backend attribute values for fpgadataflow nodes
SUPPORTED_BACKENDS = {"fpgadataflow", "hls", "rtl"}


def _get_backend_value(node):
    """Helper to extract backend value from a node. Returns None if not found."""
    if node is None:
        return None
    n_backend = get_by_name(node.attribute, "backend")
    return n_backend.s.decode("UTF-8") if n_backend is not None else None


def is_mlo(model: ModelWrapper) -> bool:
    """Returns True if the model is an MLO model (contains FINNLoop), False otherwise."""
    for node in model.graph.node:
        if node.op_type == "FINNLoop":
            return True
    return False


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


def detect_hls_rtl_dsp_conflict(model, check_subgraphs=True):
    """
    Detect if model contains both floating-point HLS ops and RTL LayerNorm.

    This combination causes incorrect simulation results in xsim due to DSP
    primitive initialization conflicts. The hardware is correct - only
    simulation is affected.

    HLS ops that use floating-point and trigger the conflict:
    - HLS Elementwise ops with FLOAT32 datatypes
    - HWSoftmax_hls (uses hls::exp)
    - LayerNorm_hls (uses hls::rsqrt)
    - Requant_hls with FLOAT32 input

    Args:
        model: ModelWrapper to check
        check_subgraphs: If True, also check inside FINNLoop bodies

    Returns:
        Tuple of (has_conflict, hls_fp_ops, rtl_dsp_ops)
        - has_conflict: bool, True if both types of ops are present
        - hls_fp_ops: list of floating-point HLS node names
        - rtl_dsp_ops: list of RTL LayerNorm node names
    """
    # RTL ops that use DSPFP32 primitive (via binopf.sv)
    RTL_DSP_OPS = {
        "LayerNorm_rtl",
    }

    # HLS ops that trigger DSP conflict with RTL LayerNorm (via hls_math.h)
    # Note: HWSoftmax_hls does NOT trigger the conflict despite using FP ops
    HLS_FP_OPS = {
        "LayerNorm_hls",
        "Requant_hls",
    }

    HLS_DOMAIN = "finn.custom_op.fpgadataflow.hls"

    hls_fp_ops = []
    rtl_dsp_ops = []

    def check_nodes(nodes, prefix=""):
        for node in nodes:
            full_name = f"{prefix}{node.name}" if prefix else node.name

            # Check for HLS ops that always use floating-point
            if node.op_type in HLS_FP_OPS:
                hls_fp_ops.append(full_name)

            # Check for HLS Elementwise ops with floating-point datatypes
            # (integer-only Elementwise ops don't use FP DSP)
            elif node.op_type.startswith("Elementwise") and node.domain == HLS_DOMAIN:
                try:
                    node_inst = getCustomOp(node)
                    # Check if any of the datatypes are floating-point
                    lhs_dtype = DataType[node_inst.get_nodeattr("lhs_dtype")]
                    rhs_dtype = DataType[node_inst.get_nodeattr("rhs_dtype")]
                    out_dtype = DataType[node_inst.get_nodeattr("out_dtype")]
                    if (
                        lhs_dtype.get_canonical_name().startswith("FLOAT")
                        or rhs_dtype.get_canonical_name().startswith("FLOAT")
                        or out_dtype.get_canonical_name().startswith("FLOAT")
                    ):
                        hls_fp_ops.append(full_name)
                except (KeyError, AttributeError):
                    # If we can't check datatypes, assume it could be floating-point
                    hls_fp_ops.append(full_name)

            # Check for RTL ops using DSPFP32
            if node.op_type in RTL_DSP_OPS:
                rtl_dsp_ops.append(full_name)

            # Check inside FINNLoop bodies
            if check_subgraphs and node.op_type == "FINNLoop":
                try:
                    loop_inst = getCustomOp(node)
                    loop_body = loop_inst.get_nodeattr("body")
                    check_nodes(loop_body.graph.node, prefix=f"{full_name}/")
                except (KeyError, AttributeError):
                    pass

    check_nodes(model.graph.node)

    has_conflict = len(hls_fp_ops) > 0 and len(rtl_dsp_ops) > 0
    return has_conflict, hls_fp_ops, rtl_dsp_ops


def warn_hls_rtl_dsp_conflict(model, verification_type, output_dir=None):
    """
    Check for HLS+RTL DSP conflict and issue warning if detected.

    This is used to warn users before running rtlsim verification when the
    model contains both HLS floating-point ops and RTL LayerNorm.
    This combination causes incorrect simulation results in xsim due to
    conflicting DSP primitive initializations.

    Args:
        model: ModelWrapper to check.
        verification_type: String describing the verification type.
        output_dir: Optional directory for verification outputs (writes warning file if provided).

    Returns:
        bool: True if conflict was detected (and verification should be skipped)
    """
    has_conflict, hls_ops, rtl_ops = detect_hls_rtl_dsp_conflict(model)

    if has_conflict:
        warning_msg = (
            f"\n{'='*70}\n"
            f"HLS+RTL DSP CONFLICT DETECTED - SKIPPING {verification_type.upper()}\n"
            f"{'='*70}\n"
            f"The model contains both HLS floating-point ops and RTL LayerNorm.\n"
            f"This causes INCORRECT simulation results in xsim (Vivado version <= 2025.2).\n"
            f"\n"
            f"HLS floating-point ops: {hls_ops}\n"
            f"RTL LayerNorm ops: {rtl_ops}\n"
            f"\n"
            f"The HARDWARE implementation is CORRECT - only xsim is currently affected.\n"
            f"Skipping {verification_type} verification.\n"
            f"{'='*70}\n"
        )

        warnings.warn(warning_msg, UserWarning)

        # Also save warning to file in output directory
        if output_dir is not None:
            log_file = os.path.join(output_dir, f"{verification_type}_SKIPPED_DSP_CONFLICT.txt")
            try:
                with open(log_file, "w") as f:
                    f.write(warning_msg)
            except (IOError, OSError):
                pass  # Don't fail if we can't write the log file

        return True
    return False
