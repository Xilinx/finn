# Copyright (c) 2021, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import os
import shutil
from finn.util.test import get_build_env, load_test_checkpoint_or_skip
import pytest
from finn.util.basic import make_build_dir
import pkg_resources as pk
import wget
import subprocess

target_clk_ns = 10
build_kind = "zynq"
build_dir = os.environ["FINN_BUILD_DIR"]
onnx_zip_url = "https://github.com/Xilinx/finn-examples"
onnx_zip_url += "/releases/download/v0.0.1a/onnx-models-bnn-pynq.zip"
onnx_zip_local = build_dir + "/onnx-models-bnn-pynq.zip"
onnx_dir_local = build_dir + "/onnx-models-bnn-pynq"


def get_checkpoint_name(step):
    if step == "build":
        # checkpoint for build step is an entire dir
        return build_dir + "/end2end_ext_weights_build"
    elif step == "download":
        return onnx_dir_local + "/tfc-w1a1.onnx"
    else:
        # other checkpoints are onnx files
        return build_dir + "/end2end_ext_weights_%s.onnx" % (step)


def test_end2end_ext_weights_download():
    if not os.path.isfile(onnx_zip_local):
        wget.download(onnx_zip_url, out=onnx_zip_local)
    assert os.path.isfile(onnx_zip_local)
    subprocess.check_output(["unzip", "-o", onnx_zip_local, "-d", onnx_dir_local])
    assert os.path.isfile(get_checkpoint_name("download"))


def test_end2end_ext_weights_build():
    model_file = get_checkpoint_name("download")
    load_test_checkpoint_or_skip(model_file)
    build_env = get_build_env(build_kind, target_clk_ns)
    folding_config_file = pk.resource_filename(
        "finn.qnn-data", "test_ext_weights/tfc-w1a1-extw.json"
    )
    output_dir = make_build_dir("test_end2end_ext_weights_build")
    cfg = build.DataflowBuildConfig(
        output_dir=output_dir,
        folding_config_file=folding_config_file,
        synth_clk_period_ns=target_clk_ns,
        board=build_env["board"],
        shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.BITFILE,
            build_cfg.DataflowOutputType.PYNQ_DRIVER,
            build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
        ],
    )
    build.build_dataflow_cfg(model_file, cfg)
    assert os.path.isfile(output_dir + "/deploy/bitfile/finn-accel.bit")
    assert os.path.isfile(output_dir + "/deploy/bitfile/finn-accel.hwh")
    assert os.path.isfile(output_dir + "/deploy/driver/driver.py")
    assert os.path.isfile(output_dir + "/deploy/driver/runtime_weights/idma0.npy")
    shutil.copytree(output_dir + "/deploy", get_checkpoint_name("build"))


def test_end2end_ext_weights_run_on_hw():
    build_env = get_build_env(build_kind, target_clk_ns)
    deploy_dir = get_checkpoint_name("build")
    if not os.path.isdir(deploy_dir):
        pytest.skip(deploy_dir + " not found from previous test step, skipping")
    driver_dir = deploy_dir + "/driver"
    assert os.path.isdir(driver_dir)
    # create a shell script for running validation: 10 batches x 10 imgs
    with open(driver_dir + "/validate.sh", "w") as f:
        f.write(
            """#!/bin/bash
cd %s/driver
echo %s | sudo -S python3.6 validate.py --dataset mnist --bitfile %s
        """
            % (
                build_env["target_dir"] + "/end2end_ext_weights_build",
                build_env["password"],
                "../bitfile/finn-accel.bit",
            )
        )
    # set up rsync command
    remote_target = "%s@%s:%s" % (
        build_env["username"],
        build_env["ip"],
        build_env["target_dir"],
    )
    rsync_res = subprocess.run(
        [
            "sshpass",
            "-p",
            build_env["password"],
            "rsync",
            "-avz",
            deploy_dir,
            remote_target,
        ]
    )
    assert rsync_res.returncode == 0
    remote_verif_cmd = [
        "sshpass",
        "-p",
        build_env["password"],
        "ssh",
        "%s@%s" % (build_env["username"], build_env["ip"]),
        "sh",
        build_env["target_dir"] + "/end2end_ext_weights_build/driver/validate.sh",
    ]
    verif_res = subprocess.run(
        remote_verif_cmd,
        stdout=subprocess.PIPE,
        universal_newlines=True,
        input=build_env["password"],
    )
    assert verif_res.returncode == 0
    log_output = verif_res.stdout.split("\n")
    assert log_output[-3] == "batch 100 / 100 : total OK 9296 NOK 704"
    assert log_output[-2] == "Final accuracy: 92.960000"
