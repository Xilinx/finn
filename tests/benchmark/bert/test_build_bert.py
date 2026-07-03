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
from collections import namedtuple

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_config import (
    DataflowOutputType,
    ShellFlowType,
    VerificationStepType,
)

# Path to the (user-supplied) quantized BERT ONNX model. Not committed.
MODEL_PATH = "tests/benchmark/bert/quantized_int8_model_cleaned.onnx"

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
LOOP_BODY_HIERARCHY = [["", "layers.0"]]

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
NodeRef = namedtuple("NodeRef", ["name"])

# Loop body of the 6-layer model spans the first encoder layer, from the
# DuplicateStreams at its input to the residual ElementwiseAdd at its output.
LOOP_BODY_RANGE = (NodeRef("DuplicateStreams_hls_0"), NodeRef("ElementwiseAdd_rtl_9"))


def select_build_steps():
    """Ordered step pipeline translated from ``l6_bert_demo.yaml``.

    at_start (custom) + base ``bert.yaml`` steps (stock FINN) + at_end (custom).
    """
    return [
        # --- at_start (custom pre-processing) ---
        custom_steps.step_bert_cleanup,
        custom_steps.step_remove_head,
        custom_steps.step_generate_reference_io,
        # --- base BERT pipeline ---
        "step_qonnx_to_finn",  # stock; runs ConvertQONNXtoFINN
        custom_steps.step_bert_streamlining,
        "step_convert_to_hw",  # = Brainsmith infer_kernels; also runs InferShuffle
        "step_create_dataflow_partition",
        "step_specialize_layers",
        "step_loop_rolling",  # MLO
        "step_target_fps_parallelization",
        "step_apply_folding_config",
        "step_minimize_bit_width",
        "step_transpose_decomposition",
        "step_generate_estimate_reports",
        "step_hw_codegen",
        "step_hw_ipgen",
        "step_set_fifo_depths",
        "step_create_stitched_ip",
        "step_measure_rtlsim_performance",
        # --- at_end (custom V80 shell integration + build) ---
        custom_steps.step_stage_reference_io,
        custom_steps.step_v80_hw_build,
        custom_steps.step_v80_sw_build,
    ]


@pytest.mark.slow
@pytest.mark.vivado
def test_build_bert():
    if not MODEL_PATH or not os.path.isfile(MODEL_PATH):
        pytest.skip("Set BERT_MODEL_PATH to a quantized BERT ONNX model to run this build.")
    if LOOP_BODY_RANGE is None:
        pytest.skip("Set LOOP_BODY_RANGE (the loop-body [start, end] node range) to run.")

    build_dir = os.environ.get("FINN_BUILD_DIR", tempfile.gettempdir())
    output_dir = os.path.join(build_dir, "output_bert_l6_v80")
    config_dir = os.path.join(os.path.dirname(__file__), "configs")

    cfg = build_cfg.DataflowBuildConfig(
        output_dir=output_dir,
        steps=select_build_steps(),
        # clock / board / shell (V80 -> SLASH_ALVEO; fpga_part resolved from board)
        synth_clk_period_ns=5.0,  # 200 MHz
        board="V80",
        shell_flow_type=ShellFlowType.SLASH_ALVEO,
        # MLO
        mlo=True,
        loop_body_hierarchy=LOOP_BODY_HIERARCHY,
        loop_body_range=LOOP_BODY_RANGE,
        # folding / thresholds / fifos
        folding_config_file=os.path.join(config_dir, "mlo_high_folding.json"),
        standalone_thresholds=True,
        preserve_thresh_shape=True,
        split_large_fifos=True,
        auto_fifo_depths=True,
        fifosim_n_inferences=2,
        fifosim_save_waveform=True,
        debug_fifo=True,
        stitched_ip_gen_dcp=True,
        verification_atol=0.1,
        mute_config_assertions=True,
        # verification (reference IO produced by step_generate_reference_io)
        verify_steps=[
            VerificationStepType.QONNX_TO_FINN_PYTHON,  # "finn_onnx_python"
            VerificationStepType.STITCHED_IP_RTLSIM,  # "stitched_ip_rtlsim"
        ],
        verify_input_npy=os.path.join(output_dir, "input.npy"),
        verify_expected_output_npy=os.path.join(output_dir, "expected_output.npy"),
        # outputs (the bitfile is produced by the custom V80 steps into deployment/)
        generate_outputs=[
            DataflowOutputType.ESTIMATE_REPORTS,
            DataflowOutputType.STITCHED_IP,
            DataflowOutputType.RTLSIM_PERFORMANCE,
        ],
        save_intermediate_models=True,
    )

    build.build_dataflow_cfg(MODEL_PATH, cfg)
