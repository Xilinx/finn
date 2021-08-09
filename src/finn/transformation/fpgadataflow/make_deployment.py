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
from distutils.dir_util import copy_tree
from shutil import copy

import finn.transformation.fpgadataflow.templates as templates
from finn.transformation.base import Transformation
from finn.util.basic import make_build_dir


class DeployToPYNQ(Transformation):
    """Collects all necessary files for deployment and copies them to the PYNQ board.
    Expects information about PYNQ board to make scp possible:

    IP address of board, username and password for board and target directory where
    the files are stored on the board"""

    def __init__(self, ip, port, username, password, target_dir):
        super().__init__()
        self.ip = ip
        self.port = port
        self.username = username
        self.password = password
        self.target_dir = target_dir

    def apply(self, model):
        # set metadata properties accordingly to user input specifications
        model.set_metadata_prop("pynq_ip", self.ip)
        model.set_metadata_prop("pynq_port", str(self.port))
        model.set_metadata_prop("pynq_username", self.username)
        model.set_metadata_prop("pynq_password", self.password)
        model.set_metadata_prop("pynq_target_dir", self.target_dir)

        # create directory for deployment files
        deployment_dir = make_build_dir(prefix="pynq_deployment_")
        model.set_metadata_prop("pynq_deployment_dir", deployment_dir)

        # get and copy necessary files
        # .bit and .hwh file
        bitfile = model.get_metadata_prop("bitfile")
        hwh_file = model.get_metadata_prop("hw_handoff")
        deploy_files = [bitfile, hwh_file]

        for dfile in deploy_files:
            if dfile is not None:
                copy(dfile, deployment_dir)

        # helper script for Alveo
        platform = model.get_metadata_prop("platform")
        if platform == "alveo":
            alveo_run_sh = templates.alveo_run_sh_template
            fill_dict = {
                "$REMOTE_DEPLOY_DIR$": self.target_dir
                + "/"
                + os.path.basename(deployment_dir),
                "$CONDA_ENV_NAME$": "finn-pynq-alveo",
                "$REMOTE_XRT$": os.environ["XILINX_XRT"],
                "$REMOTE_PLATFORM_REPO_PATHS$": os.environ["PLATFORM_REPO_PATHS"],
                "$BITFILE$": os.path.basename(bitfile),
            }
            for key, value in fill_dict.items():
                alveo_run_sh = alveo_run_sh.replace(key, value)
            alveo_run_sh_path = deployment_dir + "/alveo_run.sh"
            with open(alveo_run_sh_path, "w") as f:
                f.write(alveo_run_sh)

        # driver.py and python libraries
        pynq_driver_dir = model.get_metadata_prop("pynq_driver_dir")
        copy_tree(pynq_driver_dir, deployment_dir)
        model.set_metadata_prop("pynq_deploy_dir", deployment_dir)
        model.set_metadata_prop("exec_mode", "remote_pynq")

        # create target directory on PYNQ board
        cmd = 'ssh {}@{} -p {} "mkdir -p {}"'.format(
            self.username, self.ip, self.port, self.target_dir
        )
        bash_command = ["/bin/bash", "-c", cmd]
        process_compile = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
        process_compile.communicate()
        # copy directory to PYNQ board using scp
        cmd = "scp -P{} -r {} {}@{}:{}".format(
            self.port, deployment_dir, self.username, self.ip, self.target_dir
        )
        bash_command = ["/bin/bash", "-c", cmd]
        process_compile = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
        process_compile.communicate()

        return (model, False)
