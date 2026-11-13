# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

# custom steps for the finn-transformers models
import os
from benchmark_helpers import (
    CORE_BUILD_OUTPUT_FILES,
    benchmark_root,
    bitfile_output_files,
    check_build_outputs,
    get_verify_steps,
    make_benchmark_cfg,
    verification_io_dir,
)
from custom_steps_transformer import (
    step_transformer_prepare,
    step_transformer_streamline,
    step_transformer_tidy,
)

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import make_build_dir


# model
# The models and their verification I/O were created by the FINN+ team (Christoph
# Berganski, Paderborn University) in finn-transformers
# (https://github.com/iksnagreb/finn-transformers, rev 3a21c22) and are the QONNX
# exports also used by the FINN+ regression (https://github.com/eki-project/finn-plus).
# They are not part of this repository and are fetched separately: "benchmark" is a
# synthetic 8-bit attention model; "radioml" (RadioML 2018 modulation classification,
# 1 layer), "vision" (CIFAR-10 classification, 2 layers) and "language" (GPT-style
# masked language model, 2 layers) are 4-bit encoders with 8-bit embedding and
# output layers.
def get_model_file(model):
    return os.path.join(benchmark_root(), "models", model + ".onnx")


# verification parameters
# The verification I/O is the (input, output) pair shipped with each model. The
# language model input stays int64 (token ids), as required by its embedding Gather.
def get_verify_io(model):
    return (
        os.path.join(verification_io_dir(), model + "_input.npy"),
        os.path.join(verification_io_dir(), model + "_output.npy"),
    )


verif_steps = [
    "finn_onnx_python",
    "initial_python",
    "streamlined_python",
    "folded_hls_cppsim",
    "node_by_node_rtlsim",
    "stitched_ip_rtlsim",
]

# Per-model settings on top of the shared config. The 8-bit activations of the
# 4-bit models' embedding and output layers (e.g. the 2048-way language model output)
# are implemented as Requant instead of 255-threshold Thresholding layers, which
# would otherwise dominate the LUT budget.
MODEL_OVERRIDES = {
    "transformer_benchmark": {},
    "transformer_radioml": {"requant_bitwidth_threshold": 8},
    "transformer_vision": {"requant_bitwidth_threshold": 8},
    "transformer_language": {"requant_bitwidth_threshold": 8},
}

# Build steps: the export no-ops are removed and the inputs annotated before the
# QONNX->FINN conversion, which (like the transformer-specific streamline, moving the
# scales through the multi-head attention) runs with rank-checked layout inference.
# The rest uses the phase-based default flow.
transformer_build_steps = [
    step_transformer_tidy,
    step_transformer_prepare,
    step_transformer_streamline,
    "phase_convert_to_hardware",
    "phase_optimize_hardware",
    "phase_build_hardware",
    "phase_generate_outputs",
]

# build output products
# rtlsim performance measurement is skipped for the transformers to keep the build
# time down (the stitched-IP simulation of the larger models takes many hours).
full_outputs = [
    build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
    build_cfg.DataflowOutputType.STITCHED_IP,
    build_cfg.DataflowOutputType.BITFILE,
    build_cfg.DataflowOutputType.PYNQ_DRIVER,
    build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
]
# TEMP(stitched-ip-only): the VCK190 bitfile flow needs the PYNQ golden reference
# design (FINN_VCK190_GOLDEN=1 ./run-docker.sh), so for now the VCK190 builds stop at
# stitched IP. Re-add BITFILE/PYNQ_DRIVER/DEPLOYMENT_PACKAGE once the golden
# reference is in place.
stitched_ip_outputs = [
    build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
    build_cfg.DataflowOutputType.STITCHED_IP,
]
BOARD_OVERRIDES = {
    "ZCU104": {"generate_outputs": full_outputs},
    "VCK190": {"generate_outputs": stitched_ip_outputs},
}


def configure_build(board, model, output_dir, **overrides):
    # Folding was derived from auto-folding to the finn-plus throughput targets
    # (benchmark: 1500 fps, vision: 1525 fps, language: 1000 fps). RadioML is folded
    # to 20000 fps instead of 95000 fps, which was set for the larger RFSoC2x2 and
    # exceeds the ZCU104 DSP/BRAM budget.
    folding = os.path.join(
        benchmark_root(), "transformer", "folding_config", f"{model}_folding_config_{board}.json"
    )
    verify_input_npy, verify_expected_output_npy = get_verify_io(model)
    cfg = dict(
        steps=transformer_build_steps,
        folding_config_file=folding,
        synth_clk_period_ns=10.0,
        auto_fifo_depths=True,
        standalone_thresholds=True,
        mvau_wwidth_max=2048,
        verify_steps=get_verify_steps(verif_steps),
        verify_input_npy=verify_input_npy,
        verify_expected_output_npy=verify_expected_output_npy,
        verification_atol=0.5,
    )
    cfg.update(MODEL_OVERRIDES[model])
    cfg.update(BOARD_OVERRIDES[board])
    cfg.update(overrides)
    return make_benchmark_cfg(board, output_dir, **cfg)


TRANSFORMER_BUILDS = [
    ("ZCU104", "transformer_benchmark"),
    ("ZCU104", "transformer_radioml"),
    ("ZCU104", "transformer_vision"),
    ("VCK190", "transformer_language"),
]


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.parametrize("board,model", TRANSFORMER_BUILDS)
def test_transformer(board, model, bench_recorder):
    output_dir = make_build_dir(f"build_transformer_{model}_{board}_")

    # Run build flow
    cfg = configure_build(board, model, output_dir)
    build.build_dataflow_cfg(get_model_file(model), cfg)
    bench_recorder(model, board, output_dir)

    # Check that all expected output products are present, reporting every
    # missing artifact at once instead of aborting on the first one.
    # no rtlsim performance report (RTLSIM_PERFORMANCE is not requested)
    build_output_files = [
        f for f in CORE_BUILD_OUTPUT_FILES if f != "report/rtlsim_performance.json"
    ]
    if build_cfg.DataflowOutputType.BITFILE in cfg.generate_outputs:
        build_output_files += bitfile_output_files(board)
    else:
        # TEMP(stitched-ip-only): no bitfile/driver/deployment artifacts
        build_output_files = [f for f in build_output_files if f != "driver/driver.py"]
    check_build_outputs(output_dir, build_output_files)
