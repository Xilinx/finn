############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# SPDX-License-Identifier: MIT
############################################################################

"""
BERT (6-layer MLO + V80) dataflow build benchmark.

This test reproduces the Brainsmith ``l6_bert_demo.yaml`` blueprint flow as a
plain FINN ``DataflowBuildConfig`` build. There is no blueprint concept in FINN:
the blueprint's step pipeline is expressed directly as the ``steps`` list below
(mixing stock ``"step_*"`` names with the ported custom steps in
``custom_steps.py``), and the blueprint's ``finn_config`` block is translated
into ``DataflowBuildConfig`` arguments.

The ONNX model is *not* committed to the repo; the user is responsible for
copying it in and pointing ``BERT_MODEL_PATH`` at it. The test is skipped if the
model (or the required ``LOOP_BODY_RANGE``) is not available.
"""

import pytest

import custom_steps
import os
import tempfile

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_config import (
    DataflowOutputType,
    ShellFlowType,
    VerificationStepType,
)

# from collections import namedtuple  # MLO only: re-enable for a multi-layer BERT


# Path to the (user-supplied) quantized BERT ONNX model. Not committed.
MODEL_PATH = "tests/benchmark/bert/l1_bert_quantized_int8.onnx"

# ---------------------------------------------------------------------------
# MLO loop-rolling markers -- DISABLED for the 1-layer BERT (nothing to roll).
# To build a *multi-layer* BERT, uncomment the ``namedtuple`` import above, the
# constants below, and the ``mlo`` / ``loop_body_*`` args + MLO folding config in
# test_build_bert(), and swap the folding config back to mlo_high_folding.json.
# ---------------------------------------------------------------------------
# MLO loop-body marker. ``loop_body_hierarchy`` selects the PyTorch metadata
# hierarchy for the loop body; ``loop_body_range`` marks the (start, end) node
# range of that loop body. The range is required because this fork cannot
# propagate node metadata from the (non-dynamo) ONNX export, so the hierarchy
# alone is not enough to locate the loop body.
#
# IMPORTANT: when ``loop_body_range`` is set, ``step_loop_rolling`` first runs
# ``SetLoopBoundary`` which stamps the in-range nodes with a *hardcoded*
# ``pkg.torch.onnx.name_scopes = "['', 'layers.0']"`` metadata, then runs
# ``LoopExtraction(cfg.loop_body_hierarchy)`` which prefix-matches against that
# stamped hierarchy. The two MUST agree, so ``loop_body_hierarchy`` has to be
# ``[["", "layers.0"]]`` (the value the end2end MLO test
# tests/fpgadataflow/test_fpgadataflow_finnloop.py uses), NOT the original
# PyTorch module path. A mismatch yields an empty subgraph and the
# "GraphPattern must have at least one output" error during loop rolling.
# LOOP_BODY_HIERARCHY = [["", "layers.0"]]

# ``loop_body_range`` is consumed by ``SetLoopBoundary`` inside
# ``step_loop_rolling``; that transform only reads ``.name`` off each endpoint
# (see set_loop_boundary.py). The end2end MLO test
# (tests/fpgadataflow/test_fpgadataflow_finnloop.py) passes real graph nodes,
# e.g. ``loop_body_range=(model.graph.node[0], model.graph.node[9])``. Because
# the DataflowBuildConfig here is constructed before the model is loaded, we
# instead supply lightweight name holders: SetLoopBoundary matches the first
# node whose ``.name`` equals the start name and the first whose ``.name``
# equals the end name. The names must be the node names present at
# step_loop_rolling time (i.e. after streamlining / convert_to_hw /
# specialize_layers, so FINN HW-layer names).
# NodeRef = namedtuple("NodeRef", ["name"])

# Loop body of the 6-layer model spans the first encoder layer, from the
# DuplicateStreams at its input to the residual ElementwiseAdd at its output.
# LOOP_BODY_RANGE = (NodeRef("DuplicateStreams_hls_0"), NodeRef("ElementwiseAdd_rtl_9"))


def select_build_steps():
    """Ordered step pipeline translated from ``l6_bert_demo.yaml``.

    at_start (custom) + base ``bert.yaml`` steps (stock FINN) + at_end (custom).
    """
    return [
        # --- at_start (custom pre-processing) ---
        custom_steps.step_bert_cleanup,
        custom_steps.step_remove_head,
        custom_steps.step_generate_reference_io,
        # --- base BERT pipeline (phase-based) ---
        "phase_prepare_model",  # step_qonnx_to_finn + step_tidy_up
        custom_steps.step_bert_streamlining,  # replaces phase_optimize_model's step_streamline
        "phase_convert_to_hardware",  # convert_to_hw + partition + specialize + loop_rolling (MLO)
        "phase_optimize_hardware",  # fps + folding + minimize_bit_width + transpose + estimates
        "phase_build_hardware",  # MLO:loop-body set_fifo_depths+ipgen/stitch, then main fifo+ipgen
        "phase_generate_outputs",  # stitched IP + rtlsim perf + bitfile + driver + deploy pkg
        # --- at_end (custom V80 shell integration + build) ---
        # custom_steps.step_stage_reference_io,
        # custom_steps.step_v80_hw_build,
        # custom_steps.step_v80_sw_build,
    ]


@pytest.mark.slow
@pytest.mark.vivado
def test_build_bert():
    if not MODEL_PATH or not os.path.isfile(MODEL_PATH):
        pytest.skip("Set BERT_MODEL_PATH to a quantized BERT ONNX model to run this build.")
    # MLO only (multi-layer BERT): re-enable the loop-body range skip guard.
    # if LOOP_BODY_RANGE is None:
    #     pytest.skip("Set LOOP_BODY_RANGE (the loop-body [start, end] node range) to run.")

    build_dir = os.environ.get("FINN_BUILD_DIR", tempfile.gettempdir())
    output_dir = os.path.join(build_dir, "output_bert_l6_v80")
    config_dir = os.path.join(os.path.dirname(__file__), "configs")

    cfg = build_cfg.DataflowBuildConfig(
        output_dir=output_dir,
        steps=select_build_steps(),
        # clock / board / shell (VCK190 -> VIVADO_VERSAL; fpga_part resolved from board).
        # VCK190 is DDR-only (not in hbm_boards), so FINNLoop mem_type auto-resolves
        # to "DDR" in step_set_fifo_depths; no explicit mem_type needed.
        # The Versal flow reads the golden reference shell from $FINN_VERSAL_GOLDEN_DIR.
        synth_clk_period_ns=5.0,  # 200 MHz
        board="VCK190",
        shell_flow_type=ShellFlowType.VIVADO_VERSAL,
        # MLO -- DISABLED for the 1-layer BERT (single encoder layer, nothing to
        # roll). Re-enable these three args for a multi-layer BERT.
        # mlo=True,
        # loop_body_hierarchy=LOOP_BODY_HIERARCHY,
        # loop_body_range=LOOP_BODY_RANGE,
        # folding / thresholds / fifos
        folding_config_file=os.path.join(config_dir, "unrolled_high_folding.json"),
        # MLO (multi-layer) folding config:
        # folding_config_file=os.path.join(config_dir, "mlo_high_folding.json"),
        standalone_thresholds=True,
        preserve_thresh_shape=True,
        auto_fifo_depths=True,
        fifosim_n_inferences=2,
        # fifosim_save_waveform=True,
        # debug_fifo=True,
        # stitched_ip_gen_dcp=True,
        verification_atol=0.1,
        mute_config_assertions=True,
        # verification (reference IO produced by step_generate_reference_io)
        verify_steps=[
            VerificationStepType.QONNX_TO_FINN_PYTHON,  # "finn_onnx_python"
            VerificationStepType.STITCHED_IP_RTLSIM,  # "stitched_ip_rtlsim"
        ],
        verify_input_npy=os.path.join(output_dir, "input.npy"),
        verify_expected_output_npy=os.path.join(output_dir, "expected_output.npy"),
        # outputs (VCK190 .pdi produced by step_synthesize_bitfile into bitfile/).
        # PYNQ_DRIVER + DEPLOYMENT_PACKAGE produce driver/ and deploy/ (the deploy
        # package bundles the existing bitfile/ with the generated driver/).
        generate_outputs=[
            DataflowOutputType.ESTIMATE_REPORTS,
            DataflowOutputType.STITCHED_IP,
            DataflowOutputType.BITFILE,
            DataflowOutputType.PYNQ_DRIVER,
            DataflowOutputType.DEPLOYMENT_PACKAGE,
        ],
        save_intermediate_models=True,
    )

    build.build_dataflow_cfg(MODEL_PATH, cfg)
