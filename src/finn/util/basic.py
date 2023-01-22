# Copyright (c) 2020 Xilinx, Inc.
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

import os
import subprocess
import sys
import tempfile

# mapping from PYNQ board names to FPGA part names
pynq_part_map = dict()
pynq_part_map["Ultra96"] = "xczu3eg-sbva484-1-e"
pynq_part_map["Ultra96-V2"] = "xczu3eg-sbva484-1-i"
pynq_part_map["Pynq-Z1"] = "xc7z020clg400-1"
pynq_part_map["Pynq-Z2"] = "xc7z020clg400-1"
pynq_part_map["ZCU102"] = "xczu9eg-ffvb1156-2-e"
pynq_part_map["ZCU104"] = "xczu7ev-ffvc1156-2-e"
pynq_part_map["ZCU111"] = "xczu28dr-ffvg1517-2-e"
pynq_part_map["RFSoC2x2"] = "xczu28dr-ffvg1517-2-e"

# native AXI HP port width (in bits) for PYNQ boards
pynq_native_port_width = dict()
pynq_native_port_width["Pynq-Z1"] = 64
pynq_native_port_width["Pynq-Z2"] = 64
pynq_native_port_width["Ultra96"] = 128
pynq_native_port_width["Ultra96-V2"] = 128
pynq_native_port_width["ZCU102"] = 128
pynq_native_port_width["ZCU104"] = 128
pynq_native_port_width["ZCU111"] = 128
pynq_native_port_width["RFSoC2x2"] = 128

# Alveo device and platform mappings
alveo_part_map = dict()
alveo_part_map["U50"] = "xcu50-fsvh2104-2L-e"
alveo_part_map["U200"] = "xcu200-fsgd2104-2-e"
alveo_part_map["U250"] = "xcu250-figd2104-2L-e"
alveo_part_map["U280"] = "xcu280-fsvh2892-2L-e"

alveo_default_platform = dict()
alveo_default_platform["U50"] = "xilinx_u50_gen3x16_xdma_201920_3"
alveo_default_platform["U200"] = "xilinx_u200_xdma_201830_2"
alveo_default_platform["U250"] = "xilinx_u250_gen3x16_xdma_2_1_202010_1"
alveo_default_platform["U280"] = "xilinx_u280_xdma_201920_3"


def get_rtlsim_trace_depth():
    """Return the trace depth for rtlsim via PyVerilator. Controllable
    via the RTLSIM_TRACE_DEPTH environment variable. If the env.var. is
    undefined, the default value of 1 is returned. A trace depth of 1
    will only show top-level signals and yield smaller .vcd files.

    The following depth values are of interest for whole-network stitched IP
    rtlsim:
    - level 1 shows top-level input/output streams
    - level 2 shows per-layer input/output streams
    - level 3 shows per full-layer I/O including FIFO count signals
    """

    try:
        return int(os.environ["RTLSIM_TRACE_DEPTH"])
    except KeyError:
        return 1


def get_remote_vivado():
    """Return the address of the remote Vivado synthesis server as set by the,
    REMOTE_VIVADO environment variable, otherwise return None"""

    try:
        return os.environ["REMOTE_VIVADO"]
    except KeyError:
        return None


def get_finn_root():
    "Return the root directory that FINN is cloned into."

    try:
        return os.environ["FINN_ROOT"]
    except KeyError:
        raise Exception(
            """Environment variable FINN_ROOT must be set
        correctly. Please ensure you have launched the Docker contaier correctly.
        """
        )


def pyverilate_get_liveness_threshold_cycles():
    """Return the number of no-output cycles rtlsim will wait before assuming
    the simulation is not finishing and throwing an exception."""

    return int(os.getenv("LIVENESS_THRESHOLD", 10000))


def make_build_dir(prefix=""):
    """Creates a folder with given prefix to be used as a build dir.
    Use this function instead of tempfile.mkdtemp to ensure any generated files
    will survive on the host after the FINN Docker container exits."""
    try:
        tmpdir = tempfile.mkdtemp(prefix=prefix)
        newdir = tmpdir.replace("/tmp", os.environ["FINN_BUILD_DIR"])
        os.makedirs(newdir)
        return newdir
    except KeyError:
        raise Exception(
            """Environment variable FINN_BUILD_DIR must be set
        correctly. Please ensure you have launched the Docker contaier correctly.
        """
        )


class CppBuilder:
    """Builds the g++ compiler command to produces the executable of the c++ code
    in code_gen_dir which is passed to the function build() of this class."""

    def __init__(self):
        self.include_paths = []
        self.cpp_files = []
        self.executable_path = ""
        self.code_gen_dir = ""
        self.compile_components = []
        self.compile_script = ""

    def append_includes(self, library_path):
        """Adds given library path to include_paths list."""
        self.include_paths.append(library_path)

    def append_sources(self, cpp_file):
        """Adds given c++ file to cpp_files list."""
        self.cpp_files.append(cpp_file)

    def set_executable_path(self, path):
        """Sets member variable "executable_path" to given path."""
        self.executable_path = path

    def build(self, code_gen_dir):
        """Builds the g++ compiler command according to entries in include_paths
        and cpp_files lists. Saves it in bash script in given folder and
        executes it."""
        # raise error if includes are empty
        self.code_gen_dir = code_gen_dir
        self.compile_components.append("g++ -o " + str(self.executable_path))
        for cpp_file in self.cpp_files:
            self.compile_components.append(cpp_file)
        for lib in self.include_paths:
            self.compile_components.append(lib)
        bash_compile = ""
        for component in self.compile_components:
            bash_compile += str(component) + " "
        self.compile_script = str(self.code_gen_dir) + "/compile.sh"
        with open(self.compile_script, "w") as f:
            f.write("#!/bin/bash \n")
            f.write(bash_compile + "\n")
        bash_command = ["bash", self.compile_script]
        process_compile = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
        process_compile.communicate()


def launch_process_helper(args, proc_env=None, cwd=None):
    """Helper function to launch a process in a way that facilitates logging
    stdout/stderr with Python loggers.
    Returns (cmd_out, cmd_err)."""
    if proc_env is None:
        proc_env = os.environ.copy()
    with subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=proc_env, cwd=cwd
    ) as proc:
        (cmd_out, cmd_err) = proc.communicate()
    if cmd_out is not None:
        cmd_out = cmd_out.decode("utf-8")
        sys.stdout.write(cmd_out)
    if cmd_err is not None:
        cmd_err = cmd_err.decode("utf-8")
        sys.stderr.write(cmd_err)
    return (cmd_out, cmd_err)


def which(program):
    "Python equivalent of the shell cmd 'which'."

    # source:
    # https://stackoverflow.com/questions/377017/test-if-executable-exists-in-python
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None
