# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import functools
import os
from benchmark_helpers import (
    CORE_BUILD_OUTPUT_FILES,
    benchmark_config_paths,
    benchmark_root,
    bitfile_output_files,
    check_build_outputs,
    get_verify_steps,
    make_benchmark_cfg,
    run_feature_build,
    verification_io_dir,
)

import finn.builder.build_dataflow as build
from finn.util.basic import make_build_dir

build_flow_folder = benchmark_root()

# model
model_name = "unsw_nb15-mlp-w2a2"
model_file = os.path.join(build_flow_folder, "models", model_name + ".onnx")

# verification parameters
verify_input_npy = os.path.join(verification_io_dir(), model_name + "_input.npy")
verify_expected_output_npy = os.path.join(verification_io_dir(), model_name + "_output.npy")

verif_steps = [
    "finn_onnx_python",
    "initial_python",
    "streamlined_python",
    "folded_hls_cppsim",
    "node_by_node_rtlsim",
    "stitched_ip_rtlsim",
]


def configure_build(board, output_dir, **overrides):
    folding, specialize = benchmark_config_paths(
        "cybersecurity-mlp",
        f"cybersecurity_folding_config_{board}",
        "cybersecurity_specialize_layers",
    )
    cfg = dict(
        folding_config_file=folding,
        specialize_layers_config_file=specialize,
        synth_clk_period_ns=10.0,
        mvau_wwidth_max=80,
        stitched_ip_gen_dcp=True,
        verify_steps=get_verify_steps(verif_steps),
        verify_input_npy=verify_input_npy,
        verify_expected_output_npy=verify_expected_output_npy,
    )
    cfg.update(overrides)
    return make_benchmark_cfg(board, output_dir, **cfg)


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.parametrize("board", ["AUP-ZU3_8GB"])
def test_cybersecuritymlp(board, bench_recorder):
    output_dir = make_build_dir("build_cybersecurity-mlp_")
    # Run build flow
    cfg = configure_build(board, output_dir)
    build.build_dataflow_cfg(model_file, cfg)
    bench_recorder(model_name, board, output_dir)

    # Check that all expected output products are present, reporting every
    # missing artifact at once instead of aborting on the first one. This model
    # builds on AUP-ZU3_8GB via the Vivado/Zynq flow (.bit/.hwh).
    build_output_files = CORE_BUILD_OUTPUT_FILES + bitfile_output_files(board)
    check_build_outputs(output_dir, build_output_files)


# --- feature tests: FIFO sizing ----------------------------------------------
# The committed folding with FIFOs sized by each strategy; see "FIFO sizing tests"
# in tests/benchmark/README.md and run_feature_build in benchmark_helpers.py.


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.fifo_sizing
@pytest.mark.xdist_group("cybersecurity-mlp")
@pytest.mark.parametrize("strategy", ["heuristic_analytical", "heuristic_rtlsim"])
@pytest.mark.parametrize("board", ["AUP-ZU3_8GB"])
def test_cybersecuritymlp_fifo_sizing(board, strategy):
    run_feature_build(
        f"cybersecurity-mlp_{board}",
        functools.partial(configure_build, board),
        model_file,
        f"fifo-{strategy}",
        auto_fifo_depths=True,
        auto_fifo_strategy=strategy,
    )
