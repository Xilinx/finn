# Copyright (c) 2020 Xilinx, Inc.
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
# * Neither the name of Xilinx nor the names of its
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

import os
import warnings
from qonnx.core.datatype import DataType
from qonnx.custom_op.registry import getCustomOp, is_custom_op
from qonnx.util.basic import get_by_name


def is_fpgadataflow_node(node):
    """Returns True if given node is fpgadataflow node. Otherwise False."""
    is_node = False
    if node is not None:
        if is_custom_op(node.domain):
            n_backend = get_by_name(node.attribute, "backend")
            if n_backend is not None:
                backend_value = n_backend.s.decode("UTF-8")
                if backend_value == "fpgadataflow":
                    is_node = True

    return is_node


def is_hls_node(node):
    """Returns True if given node is hls node. Otherwise False."""
    is_node = False
    if node is not None:
        if node.domain == "finn.custom_op.fpgadataflow.hls":
            n_backend = get_by_name(node.attribute, "backend")
            if n_backend is not None:
                backend_value = n_backend.s.decode("UTF-8")
                if backend_value == "fpgadataflow":
                    is_node = True

    return is_node


def is_rtl_node(node):
    """Returns True if given node is rtl node. Otherwise False."""
    is_node = False
    if node is not None:
        if node.domain == "finn.custom_op.fpgadataflow.rtl":
            n_backend = get_by_name(node.attribute, "backend")
            if n_backend is not None:
                backend_value = n_backend.s.decode("UTF-8")
                if backend_value == "fpgadataflow":
                    is_node = True

    return is_node


def detect_hls_rtl_dsp_conflict(model, check_subgraphs=True):
    """
    Detect if model contains both floating-point HLS Elementwise ops and RTL ops using DSPFP32.

    This combination causes incorrect simulation results in xsim due to DSP
    primitive initialization conflicts. The hardware is correct - only
    simulation is affected.

    Note: Only HLS Elementwise ops using floating-point datatypes are flagged,
    as integer-only HLS Elementwise ops don't use DSP primitives.

    Args:
        model: ModelWrapper to check
        check_subgraphs: If True, also check inside FINNLoop bodies

    Returns:
        Tuple of (has_conflict, hls_elementwise_ops, rtl_dsp_ops)
        - has_conflict: bool, True if both types of ops are present
        - hls_elementwise_ops: list of floating-point HLS Elementwise node names
        - rtl_dsp_ops: list of RTL DSP node names (LayerNorm_rtl, Elementwise*_rtl)
    """
    # RTL ops that use DSPFP32 primitive (via binopf.sv)
    RTL_DSP_OPS = {
        "LayerNorm_rtl",
        "ElementwiseAdd_rtl",
        "ElementwiseSub_rtl",
        "ElementwiseMul_rtl",
    }

    HLS_DOMAIN = "finn.custom_op.fpgadataflow.hls"

    hls_elementwise_ops = []
    rtl_dsp_ops = []

    def check_nodes(nodes, prefix=""):
        for node in nodes:
            full_name = f"{prefix}{node.name}" if prefix else node.name

            # Check for HLS Elementwise ops with floating-point datatypes
            if node.op_type.startswith("Elementwise") and node.domain == HLS_DOMAIN:
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
                        hls_elementwise_ops.append(full_name)
                except (KeyError, AttributeError):
                    # If we can't check datatypes, assume it could be floating-point
                    hls_elementwise_ops.append(full_name)

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

    has_conflict = len(hls_elementwise_ops) > 0 and len(rtl_dsp_ops) > 0
    return has_conflict, hls_elementwise_ops, rtl_dsp_ops


def warn_hls_rtl_dsp_conflict(model, verification_type, output_dir=None):
    """
    Check for HLS+RTL DSP conflict and issue warning if detected.

    This is used to warn users before running rtlsim verification when the
    model contains both HLS Elementwise ops and RTL ops that use DSPFP32.
    This combination causes incorrect simulation results in xsim due to
    conflicting DSP primitive initializations.

    Args:
        model: ModelWrapper to check
        verification_type: String describing the verification type
        output_dir: Directory where verification outputs would be saved (optional)
                    If provided, writes warning to a .txt file there

    Returns:
        bool: True if conflict was detected (and verification should be skipped)
    """
    has_conflict, hls_ops, rtl_ops = detect_hls_rtl_dsp_conflict(model)

    if has_conflict:
        warning_msg = (
            f"\n{'='*70}\n"
            f"HLS+RTL DSP CONFLICT DETECTED - SKIPPING {verification_type.upper()}\n"
            f"{'='*70}\n"
            f"The model contains both HLS Elementwise ops and RTL ops using DSPFP32.\n"
            f"This causes INCORRECT simulation results in xsim.\n"
            f"\n"
            f"HLS Elementwise ops: {hls_ops}\n"
            f"RTL DSP ops (DSPFP32): {rtl_ops}\n"
            f"\n"
            f"The HARDWARE implementation is CORRECT - only xsim is affected.\n"
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
