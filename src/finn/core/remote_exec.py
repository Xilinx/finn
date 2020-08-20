# Copyright (c) 2020, Xilinx
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

import os
import subprocess
import warnings
import numpy as np


def remote_exec(model, execution_context):
    """Executes the given model remotely on the pynq board. The metadata properties
    related to the pynq board have to be set. The execution context contains the
    input values."""
    # TODO fix for multi input-output
    pynq_ip = model.get_metadata_prop("pynq_ip")
    pynq_port = int(model.get_metadata_prop("pynq_port"))
    pynq_username = model.get_metadata_prop("pynq_username")
    pynq_password = model.get_metadata_prop("pynq_password")
    pynq_target_dir = model.get_metadata_prop("pynq_target_dir")
    deployment_dir = model.get_metadata_prop("pynq_deploy_dir")
    platform = model.get_metadata_prop("platform")
    assert platform in ["alveo", "zynq", "zynq-iodma"]
    bitfile = model.get_metadata_prop("bitfile")
    bitfile = os.path.basename(bitfile)
    if pynq_password == "":
        if "zynq" in platform:
            raise Exception("PYNQ board remote exec needs password for sudo")
        else:
            local_prefix = ""  # assume we are using an ssh key
            warnings.warn("Empty password, make sure you've set up an ssh key")
    else:
        local_prefix = "sshpass -p %s " % pynq_password

    if platform == "alveo":
        # Alveo can run without sudo
        remote_prefix = ""
    elif "zynq" in platform:
        # PYNQ Zynq boards need to execute with sudo
        remote_prefix = "echo %s | sudo -S " % pynq_password

    inp = execution_context[model.graph.input[0].name]
    # make copy of array before saving it
    inp = inp.copy()
    batchsize = inp.shape[0]
    np.save(os.path.join(deployment_dir, "input.npy"), inp)
    # extracting last folder of absolute path (deployment_dir)
    deployment_folder = os.path.basename(os.path.normpath(deployment_dir))
    # copy input to PYNQ board
    cmd = local_prefix + "scp -P{} -r {}/input.npy {}@{}:{}/{}".format(
        pynq_port,
        deployment_dir,
        pynq_username,
        pynq_ip,
        pynq_target_dir,
        deployment_folder,
    )
    bash_command = ["/bin/bash", "-c", cmd]
    process_scp_in = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
    process_scp_in.communicate()

    # use platform attribute for correct remote execution
    if platform == "alveo":
        remote_cmd = "bash -i alveo_run.sh execute %d" % (batchsize)
    else:
        remote_cmd = (
            "python3.6 driver.py --exec_mode=execute --batchsize={} "
            "--bitfile={} --inputfile=input.npy --outputfile=output.npy "
            '--platform={} "'
        ).format(batchsize, bitfile, platform)
    cmd = (
        local_prefix + "ssh {}@{} -p {} " '"cd {}/{}; ' + remote_prefix + remote_cmd
    ).format(pynq_username, pynq_ip, pynq_port, pynq_target_dir, deployment_folder)
    bash_command = ["/bin/bash", "-c", cmd]
    process_exec_accel = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
    process_exec_accel.communicate()
    # remove stale output file from local dir, if any
    try:
        os.remove("{}/output.npy".format(deployment_dir))
    except FileNotFoundError:
        pass
    # copy generated output to local
    cmd = local_prefix + "scp -P{} {}@{}:{}/{}/output.npy {}".format(
        pynq_port,
        pynq_username,
        pynq_ip,
        pynq_target_dir,
        deployment_folder,
        deployment_dir,
    )
    bash_command = ["/bin/bash", "-c", cmd]
    process_scp_out = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
    process_scp_out.communicate()
    outp = np.load("{}/output.npy".format(deployment_dir))
    execution_context[model.graph.output[0].name] = outp
