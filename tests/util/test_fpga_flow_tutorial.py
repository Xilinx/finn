# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import numpy as np
import os
import runpy
from shutil import copytree

from finn.util.basic import make_build_dir


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.end2end
def test_fpga_flow_tutorial():
    # Copy the tutorial folder to a temporary build dir so we don't pollute the
    # repo and so build.py's relative paths (input.npy, expected_output.npy,
    # folding_config.json, templates/) resolve against a writable directory.
    test_dir = make_build_dir("test_fpga_flow_tutorial_")
    target_dir = test_dir + "/fpga_flow"
    tutorial_dir = os.environ["FINN_ROOT"] + "/tutorials/fpga_flow"
    copytree(tutorial_dir, target_dir)

    # build.py uses relative paths and a relative output_dir, so run it with the
    # copied tutorial folder as the working directory. This mirrors invoking
    # `python build.py` from within tutorials/fpga_flow as the README describes.
    old_wd = os.getcwd()
    os.chdir(target_dir)
    try:
        runpy.run_path("build.py", run_name="__main__")
    finally:
        os.chdir(old_wd)

    # output_dir is hardcoded in build.py as output_<model>_<platform>
    output_dir = target_dir + "/output_tfc_w1a1_fpga"
    # standard build artifacts
    assert os.path.isfile(output_dir + "/build_dataflow.log")
    assert os.path.isfile(output_dir + "/time_per_step.json")
    assert os.path.isfile(output_dir + "/final_hw_config.json")
    # STITCHED_IP output product
    assert os.path.isfile(output_dir + "/stitched_ip/ip/component.xml")

    # artifacts produced by the custom_step_gen_tb_and_io step
    sim_dir = output_dir + "/sim"
    assert os.path.isfile(sim_dir + "/input.dat")
    assert os.path.isfile(sim_dir + "/expected_output.dat")
    assert os.path.isfile(sim_dir + "/finn_testbench.sv")
    assert os.path.isfile(sim_dir + "/make_sim_proj.tcl")

    # verification outputs for the verify_steps configured in build.py
    verif_batchsize = np.load(target_dir + "/input.npy").shape[0]
    verify_out_dir = output_dir + "/verification_output"
    for i in range(verif_batchsize):
        assert os.path.isfile(verify_out_dir + f"/verify_initial_python_{i}_SUCCESS.npy")
        assert os.path.isfile(verify_out_dir + f"/verify_streamlined_python_{i}_SUCCESS.npy")
        assert os.path.isfile(verify_out_dir + f"/verify_folded_hls_cppsim_{i}_SUCCESS.npy")
        assert os.path.isfile(verify_out_dir + f"/verify_stitched_ip_rtlsim_{i}_SUCCESS.npy")
