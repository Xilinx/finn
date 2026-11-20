# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Minimal SigLIP W4A6 bitfile (PDI) build for the VCK190"""

import pytest

import os
from benchmark_helpers import (
    benchmark_root,
    bitfile_output_files,
    check_build_outputs,
    make_benchmark_cfg,
)
from custom_steps_siglip import (
    step_assign_ddr_weight_offsets_for_loops,
    step_force_uram_attention_fifos,
)

import finn.builder.build_dataflow as build
from finn.builder.build_dataflow_config import DataflowOutputType
from finn.util.basic import make_build_dir

os.environ.setdefault("LIVENESS_THRESHOLD", "1000000")


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
def test_siglip_w4a6(bench_recorder):
    prepared_model = os.environ.get("SIGLIP_W4A6_PREPARED_MODEL", "").strip()
    assert os.path.isfile(prepared_model), (
        "set SIGLIP_W4A6_PREPARED_MODEL to the prepared siglip FINNLoop onnx "
        f"(got: {prepared_model!r})"
    )
    output_dir = make_build_dir("build_siglip_w4a6_")
    print(f"siglip_w4a6 output_dir={output_dir}")
    folding_file = os.path.join(
        benchmark_root(), "siglip_w4a6", "folding_config", "siglip_w4a6_folding_config_VCK190.json"
    )
    # the prepared model is already rolled, so the flow starts at the optimize phase
    steps = ["phase_optimize_hardware", "phase_build_hardware", "phase_generate_outputs"]
    cfg = make_benchmark_cfg(
        "VCK190",
        output_dir,
        steps=steps,
        folding_config_file=folding_file,
        synth_clk_period_ns=4.0,
        standalone_thresholds=True,
        auto_fifo_depths=False,  # outside-loop FIFO depths come from the folding config
        mlo=False,  # already rolled into a FINNLoop
        mute_config_assertions=True,
        generate_outputs=[
            DataflowOutputType.ESTIMATE_REPORTS,
            DataflowOutputType.STITCHED_IP,
            DataflowOutputType.BITFILE,
            DataflowOutputType.PYNQ_DRIVER,
            DataflowOutputType.DEPLOYMENT_PACKAGE,
        ],
        inject_steps_after={
            "step_assign_ddr_weight_offsets": [step_assign_ddr_weight_offsets_for_loops],
        },
        inject_steps_before=(
            {}
            if os.environ.get("SIGLIP_W4A6_NO_URAM", "0") == "1"
            else {"step_hw_codegen": [step_force_uram_attention_fifos]}
        ),
    )
    assert build.build_dataflow_cfg(prepared_model, cfg) == 0
    bench_recorder("siglip-w4a6", "VCK190", output_dir)

    # VCK190 is Versal, so the accelerator is a .pdi
    check_build_outputs(
        output_dir,
        bitfile_output_files("VCK190")
        + [
            "final_hw_config.json",
            "stitched_ip/ip/component.xml",
            "report/estimate_layer_cycles.json",
            "report/estimate_layer_resources.json",
            "driver/driver.py",
            "deploy/bitfile/finn-accel.pdi",
        ],
    )
