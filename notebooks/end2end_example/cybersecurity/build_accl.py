import os
import shutil
import numpy as np

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg

unsw_nb15_data = np.load("./unsw_nb15_binarized.npz")

inp = unsw_nb15_data["train"][:, :-1]
inp = np.concatenate([inp, np.zeros((inp.shape[0], 7))], -1).astype(np.float32)
out = unsw_nb15_data["train"][:, -1].astype(np.float32)

indices = np.where(out == 0)[:10]

inp = 2 * inp[indices] - 1
out = 2 * out[indices] - 1

np.save(open("input.npy", "wb"), inp)
np.save(open("expected_output.npy", "wb"), out)

model_dir = os.environ['FINN_ROOT'] + "/notebooks/end2end_example/cybersecurity"
model_file = model_dir + "/cybsec-mlp-ready.onnx"

estimates_output_dir = "output_estimates_only"

#Delete previous run results if exist
if os.path.exists(estimates_output_dir):
    shutil.rmtree(estimates_output_dir)
    print("Previous run results deleted!")

cfg_estimates = build.DataflowBuildConfig(
    verbose             = True,
    output_dir          = estimates_output_dir,
    mvau_wwidth_max     = 80,
    target_fps          = 1000000,
    synth_clk_period_ns = 10.0,
    fpga_part           = "xc7z020clg400-1",
    steps               = build_cfg.estimate_only_dataflow_steps,
    generate_outputs=[
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
    ],
    verify_steps=[build_cfg.VerificationStepType.FOLDED_HLS_CPPSIM],
    board                = 'U250',
    num_boards           = 2,
    save_intermediate_models = True,
)

build.build_dataflow_cfg(model_file, cfg_estimates)

