# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""TinyDeiT W3A3 PDI build for the VCK190.

Input is the prepared model, the QONNX checkpoint converted to fpgadataflow layers and rolled
into one FINNLoop (``tinydeit_mlo.onnx``). Set ``TINYDEIT_W3A3_PREPARED_MODEL`` to its path.
"""

import pytest

import os
from benchmark_helpers import (
    benchmark_root,
    bitfile_output_files,
    check_build_outputs,
    make_benchmark_cfg,
)
from custom_steps_tinydeit import (
    step_assign_ddr_weight_offsets_for_loops,
    step_make_loop_inputs_lazy,
    step_round_loop_thresholds,
)

import finn.builder.build_dataflow as build
from finn.builder.build_dataflow_config import DataflowOutputType
from finn.util.basic import make_build_dir


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
def test_tinydeit_w3a3_bitfile(bench_recorder):
    prepared_model = os.environ.get("TINYDEIT_W3A3_PREPARED_MODEL", "").strip()
    assert os.path.isfile(prepared_model), (
        "set TINYDEIT_W3A3_PREPARED_MODEL to the prepared tinydeit_mlo.onnx "
        f"(got: {prepared_model!r})"
    )
    output_dir = make_build_dir("build_tinydeit_w3a3_")
    cfg = make_benchmark_cfg(
        "VCK190",
        output_dir,
        # the model is already prepared, so the flow starts at the optimize phase
        steps=["phase_optimize_hardware", "phase_build_hardware", "phase_generate_outputs"],
        folding_config_file=os.environ.get("TINYDEIT_W3A3_FOLDING_CONFIG", "").strip()
        or os.path.join(
            benchmark_root(),
            "tinydeit_w3a3",
            "folding_config",
            "tinydeit_w3a3_folding_config_VCK190.json",
        ),
        synth_clk_period_ns=4.0,
        standalone_thresholds=True,
        infer_shuffle_skip_first=False,
        auto_fifo_depths=False,  # FIFO depths come from the folding config
        mlo=False,  # already rolled into a FINNLoop
        generate_outputs=[
            DataflowOutputType.ESTIMATE_REPORTS,
            DataflowOutputType.BITFILE,
            DataflowOutputType.PYNQ_DRIVER,
            DataflowOutputType.DEPLOYMENT_PACKAGE,
        ],
        inject_steps_after={
            "step_minimize_bit_width": [step_make_loop_inputs_lazy, step_round_loop_thresholds],
            "step_assign_ddr_weight_offsets": [step_assign_ddr_weight_offsets_for_loops],
        },
    )
    assert build.build_dataflow_cfg(prepared_model, cfg) == 0
    bench_recorder("tinydeit-w3a3", "VCK190", output_dir)

    check_build_outputs(
        output_dir,
        bitfile_output_files("VCK190")
        + ["driver/driver.py", "final_hw_config.json", "deploy/bitfile/finn-accel.pdi"],
    )
