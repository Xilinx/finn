##
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
##

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg

from platform_build_steps import (
    test_step_gen_vitis_xo,
    test_step_gen_instrumentation_wrapper,
    test_step_insert_tlastmarker,
    test_step_gen_instrwrap_sim,
    test_step_export_xo,
    test_step_build_platform,
)

model_name = "tfc_w1a1"
platform_name = "VMK180"
fpga_part = "xcvm1802-vsva2197-2MP-e-S"

base_output_dir="output_%s_%s" % (model_name, platform_name)


build_steps = build_cfg.default_build_dataflow_steps + [
    test_step_gen_vitis_xo,
    test_step_gen_instrumentation_wrapper,
    test_step_gen_instrwrap_sim,
    test_step_export_xo,
    test_step_build_platform,
]

step_stitchedip_ind = build_steps.index("step_create_stitched_ip")
build_steps.insert(step_stitchedip_ind, test_step_insert_tlastmarker)
build_steps.remove("step_specialize_to_rtl")

cfg = build.DataflowBuildConfig(
    steps=build_steps,
    board=platform_name,
    fpga_part=fpga_part,
    output_dir=base_output_dir,
    synth_clk_period_ns=3.3,
    folding_config_file="folding_config.json",
    stitched_ip_gen_dcp=False,
    generate_outputs=[
        build_cfg.DataflowOutputType.STITCHED_IP,
    ],
    save_intermediate_models=True,
)
model_file = "model.onnx"
build.build_dataflow_cfg(model_file, cfg)
