import os
import shutil
import numpy as np

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg

inp = np.zeros((1, 3, 32, 32)).astype(np.float32)
out = np.zeros((1)).astype(np.float32)

np.save(open("input.npy", "wb"), inp)
np.save(open("expected_output.npy", "wb"), out)


model_dir = os.environ['FINN_ROOT'] + "/pretrained_models"
model_file = model_dir + "/end2end_cnv_w1a1_folded.onnx"

estimates_output_dir = "output_estimates_only"

os.environ["RTLSIM_TRACE_DEPTH"] = "3"

steps = [
    # "step_qonnx_to_finn",
    # "step_tidy_up",
    # "step_streamline",
    # "step_convert_to_hls",
    # "step_create_dataflow_partition",
    # "step_target_fps_parallelization",
    # "step_apply_folding_config",
    "step_minimize_bit_width",
    "step_assign_partition_ids",
    "step_insert_accl",
    "step_split_dataflow",
    "step_generate_estimate_reports",
    "step_hls_codegen",
    "step_hls_ipgen",
    "step_set_fifo_depths",
    "step_create_stitched_ip",
]

cfg_splits = build.DataflowBuildConfig(
    verbose             = True,
    output_dir          = estimates_output_dir,
    steps               = steps,
    # mvau_wwidth_max     = 64,
    # target_fps          = 1000000,
    synth_clk_period_ns = 10.0,
    generate_outputs    = [
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        build_cfg.DataflowOutputType.STITCHED_IP,
    ],
    stitched_ip_gen_dcp = True,
    verify_steps        = [build_cfg.VerificationStepType.FOLDED_HLS_CPPSIM],
    shell_flow_type     = build_cfg.ShellFlowType.VITIS_ALVEO,
    board               = "U55C",
    num_boards          = 2,
    save_intermediate_models = True,
    # start_step="step_insert_accl",
)

build.build_distributed_dataflow_cfg(model_file, cfg_splits)

