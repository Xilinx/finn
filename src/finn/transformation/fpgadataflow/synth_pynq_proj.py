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

from finn.transformation import Transformation


class SynthPYNQProject(Transformation):
    """Run synthesis for the PYNQ project for this graph. The MakePYNQProject
    transformation must be applied prior to this transformation."""

    def __init__(self):
        super().__init__()

    def apply(self, model):
        vivado_pynq_proj_dir = model.get_metadata_prop("vivado_pynq_proj")
        if vivado_pynq_proj_dir is None or (not os.path.isdir(vivado_pynq_proj_dir)):
            raise Exception("No synthesis project, apply MakePYNQProject first.")
        synth_project_sh = vivado_pynq_proj_dir + "/synth_project.sh"
        if not os.path.isfile(synth_project_sh):
            raise Exception("No synthesis script, apply MakePYNQProject first.")
        bash_command = ["bash", synth_project_sh]
        process_compile = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
        process_compile.communicate()
        # set bitfile attribute
        model.set_metadata_prop("bitfile", vivado_pynq_proj_dir + "/resizer.bit")
        # TODO pull out synthesis statistics and put them in as attributes
        return (model, False)
