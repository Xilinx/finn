# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file are derived from Brainsmith:
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
BERT (1-layer, int8) dataflow build benchmark.

This test reproduces the Brainsmith BERT blueprint flow as a plain FINN
``DataflowBuildConfig`` build. The blueprint's step pipeline is expressed
directly as the ``steps`` list below (mixing stock phases with the ported custom
steps in ``custom_steps_bert.py``), and the blueprint's ``finn_config`` block is
translated into ``DataflowBuildConfig`` arguments.
"""

import pytest

import os
from benchmark_helpers import (
    benchmark_root,
    check_build_outputs,
    get_verify_steps,
    make_benchmark_cfg,
    verification_io_dir,
)
from custom_steps_bert import (
    step_bert_cleanup,
    step_bert_streamlining,
    step_remove_head,
)

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import make_build_dir

# model
model_name = "l1_bert_quantized_int8"
model_file = os.path.join(benchmark_root(), "models", model_name + ".onnx")

# verification parameters
# The verification I/O is a fixed-seed uniform [0, 1000) input for the
# head-removed model (input = output of the first LayerNorm) and the
# corresponding output of the head-removed model before step_qonnx_to_finn.
verify_input_npy = os.path.join(verification_io_dir(), model_name + "_input.npy")
verify_expected_output_npy = os.path.join(verification_io_dir(), model_name + "_output.npy")

verif_steps = [
    "finn_onnx_python",
    "stitched_ip_rtlsim",
]

# Build steps: the embedding head (everything up to and including the first
# LayerNorm) is removed before the QONNX->FINN conversion, and the BERT-specific
# streamline (softmax Mul handling, thresholds kept per-tensor) replaces the
# standard streamline phase. The rest uses the phase-based default flow.
bert_build_steps = [
    step_bert_cleanup,
    step_remove_head,
    "phase_prepare_model",
    step_bert_streamlining,
    "phase_convert_to_hardware",
    "phase_optimize_hardware",
    "phase_build_hardware",
    "phase_generate_outputs",
]

# build output products
# rtlsim performance measurement is skipped for BERT to keep the build time down.
# TEMP(stitched-ip-only): the VCK190 bitfile flow needs the PYNQ golden reference
# design (FINN_VCK190_GOLDEN=1 ./run-docker.sh), so for now we stop at stitched IP.
# The output products gate which steps actually execute, so leaving
# BITFILE/PYNQ_DRIVER/DEPLOYMENT_PACKAGE out skips synth/driver/deployment. Re-add
# them once the golden reference is in place.
build_outputs = [
    build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
    build_cfg.DataflowOutputType.STITCHED_IP,
]


def configure_build(board, output_dir, **overrides):
    # Folding is hand-tuned for the VCK190; no specialize config, the default
    # layer specialization is used.
    folding = os.path.join(
        benchmark_root(), "bert", "folding_config", f"bert_folding_config_{board}.json"
    )
    cfg = dict(
        steps=bert_build_steps,
        generate_outputs=build_outputs,
        folding_config_file=folding,
        synth_clk_period_ns=5.0,
        standalone_thresholds=True,
        auto_fifo_depths=True,
        fifosim_n_inferences=2,
        verify_steps=get_verify_steps(verif_steps),
        verify_input_npy=verify_input_npy,
        verify_expected_output_npy=verify_expected_output_npy,
    )
    cfg.update(overrides)
    return make_benchmark_cfg(board, output_dir, **cfg)


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.parametrize("board", ["VCK190"])
def test_bert(board, bench_recorder):
    output_dir = make_build_dir(f"build_bert_{board}_")

    # Run build flow
    cfg = configure_build(board, output_dir)
    build.build_dataflow_cfg(model_file, cfg)
    bench_recorder(model_name, board, output_dir)

    # Check that all expected output products are present, reporting every
    # missing artifact at once instead of aborting on the first one.
    # TEMP(stitched-ip-only): bitfile/driver/deployment artifacts are omitted
    # while the VCK190 golden reference is unavailable; re-add
    # bitfile_output_files(board) and driver/driver.py once enabled.
    build_output_files = [
        "time_per_step.json",
        "final_hw_config.json",
        "template_specialize_layers_config.json",
        "stitched_ip/ip/component.xml",
        "report/estimate_layer_cycles.json",
        "report/estimate_layer_resources.json",
        "report/estimate_network_performance.json",
    ]
    check_build_outputs(output_dir, build_output_files)
