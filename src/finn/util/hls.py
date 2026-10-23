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


from finn.util.basic import require_vivado_version, resolve_xilinx_tool
from finn.util.launcher import run_tool


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
        """Runs Vitis HLS on the tcl script, with code_gen_dir as the working
        directory. Also writes ipgen.sh in code_gen_dir, which repeats the run by
        hand. The output of HLS goes to ipgen.stdout.log and ipgen.stderr.log in
        code_gen_dir. Raises ToolFailed if HLS fails."""
        if require_vivado_version() > (2024, 2):
            command = [resolve_xilinx_tool("vitis-run"), "--mode", "hls", "--tcl", self.tcl_script]
        else:
            command = [resolve_xilinx_tool("vitis_hls"), self.tcl_script]
        self.code_gen_dir = code_gen_dir
        self.ipgen_script = str(self.code_gen_dir) + "/ipgen.sh"
        run_tool(command, cwd=code_gen_dir, replay=self.ipgen_script)
