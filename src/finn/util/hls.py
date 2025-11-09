# Copyright (c) 2021 Xilinx, Inc.
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
# * Neither the name of Xilinx nor the names of its
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


import logging
import os
import re

from finn.util.basic import launch_process_helper, which


class CallHLS:
    """Call vitis_hls to run HLS build tcl scripts."""

    def __init__(self):
        self.tcl_script = ""
        self.ipgen_path = ""
        self.code_gen_dir = ""
        self.ipgen_script = ""

    def append_tcl(self, tcl_script):
        """Sets the tcl script to be executed."""
        self.tcl_script = tcl_script

    def set_ipgen_path(self, path):
        """Sets member variable ipgen_path to given path."""
        self.ipgen_path = path

    def build(self, code_gen_dir):
        """Runs HLS synthesis using vitis_hls or vitis-run based on Vivado version."""
        self.code_gen_dir = code_gen_dir

        # Determine command based on Vivado version (renamed in 2024.2)
        vivado_path = os.environ.get("XILINX_VIVADO")
        match = re.search(r"\b(20\d{2})\.(1|2)\b", vivado_path)
        year, minor = int(match.group(1)), int(match.group(2))

        if (year, minor) > (2024, 2):
            assert which("vitis-run") is not None, "vitis-run not found in PATH"
            cmd = ["vitis-run", "--mode", "hls", "--tcl", self.tcl_script]
        else:
            assert which("vitis_hls") is not None, "vitis_hls not found in PATH"
            cmd = ["vitis_hls", self.tcl_script]

        logger = logging.getLogger("finn.vitis.hls")
        exitcode = launch_process_helper(
            cmd,
            cwd=code_gen_dir,
            logger=logger,
            stdout_level=logging.INFO,
            stderr_level=logging.WARNING,
            raise_on_error=False,
            generate_script=os.path.join(code_gen_dir, "ipgen.sh"),
        )
        if exitcode != 0:
            logger.warning("HLS synthesis returned non-zero exit code: %d", exitcode)
