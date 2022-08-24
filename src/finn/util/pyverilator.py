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

import pkg_resources as pk

import os
import shutil
from pyverilator import PyVerilator

from finn.util.basic import get_rtlsim_trace_depth, make_build_dir


def pyverilate_stitched_ip(
    model,
    read_internal_signals=True,
    disable_common_warnings=True,
    extra_verilator_args=[],
):
    """Given a model with stitched IP, return a PyVerilator sim object.
    Trace depth is also controllable, see get_rtlsim_trace_depth()

    :param read_internal_signals  If set, it will be possible to examine the
        internal (not only port) signals of the Verilog module, but this may
        slow down compilation and emulation.

    :param disable_common_warnings If set, disable the set of warnings that
        Vivado-HLS-generated Verilog typically triggers in Verilator
        (which can be very verbose otherwise)

    """
    if PyVerilator is None:
        raise ImportError("Installation of PyVerilator is required.")

    vivado_stitch_proj_dir = model.get_metadata_prop("vivado_stitch_proj")
    with open(vivado_stitch_proj_dir + "/all_verilog_srcs.txt", "r") as f:
        all_verilog_srcs = f.read().split()

    def file_to_dir(x):
        return os.path.dirname(os.path.realpath(x))

    def file_to_basename(x):
        return os.path.basename(os.path.realpath(x))

    top_module_file_name = file_to_basename(model.get_metadata_prop("wrapper_filename"))
    top_module_name = top_module_file_name.strip(".v")
    build_dir = make_build_dir("pyverilator_ipstitched_")

    # dump all Verilog code to a single file
    # this is because large models with many files require
    # a verilator command line too long for bash on most systems
    # NOTE: there are duplicates in this list, and some files
    # are identical but in multiple directories (regslice_core.v)

    # remove duplicates from list by doing list -> set -> list
    src_exts = [".v", ".sv"]

    all_verilog_src_files = list(
        set(
            filter(
                lambda x: any(map(lambda y: x.endswith(y), src_exts)), all_verilog_srcs
            )
        )
    )

    verilog_header_dir = make_build_dir("pyverilator_vh_")
    # use custom version of axis infrastructure vh
    custom_vh = pk.resource_filename(
        "finn.qnn-data", "verilog/custom_axis_infrastructure.vh"
    )
    shutil.copy(custom_vh, verilog_header_dir + "/axis_infrastructure_v1_1_0.vh")
    for fn in all_verilog_srcs:
        if fn.endswith(".vh"):
            if "axis_infrastructure_v1_1_0.vh" in fn:
                # skip, we use a custom version for this file without recursive gcd
                continue
            else:
                shutil.copy(fn, verilog_header_dir)

    # remove all but one instances of regslice_core.v
    filtered_verilog_files = []
    remove_entry = False
    for vfile in all_verilog_src_files:
        if "regslice_core" in vfile:
            if not remove_entry:
                filtered_verilog_files.append(vfile)
            remove_entry = True
        else:
            filtered_verilog_files.append(vfile)

    # concatenate all verilog code into a single file
    with open(vivado_stitch_proj_dir + "/" + top_module_file_name, "w") as wf:
        for vfile in filtered_verilog_files:
            with open(vfile) as rf:
                wf.write("//Added from " + vfile + "\n\n")
                lines = rf.read()
                for line in lines.split("\n"):
                    # break down too-long lines, Verilator complains otherwise
                    if len(line) > 20000:
                        line = line.replace("&", "\n&")
                    wf.write("\n" + line)

    verilator_args = []
    # disable common verilator warnings that should be harmless but commonly occur
    # in large quantities for Vivado HLS-generated verilog code
    if disable_common_warnings:
        verilator_args += ["-Wno-STMTDLY"]
        verilator_args += ["-Wno-PINMISSING"]
        verilator_args += ["-Wno-IMPLICIT"]
        verilator_args += ["-Wno-WIDTH"]
        verilator_args += ["-Wno-COMBDLY"]
    # force inlining of all submodules to ensure we can read internal signals properly
    if read_internal_signals:
        verilator_args += ["--inline-mult", "0"]
    # add defines to make certain XPM src files work with Verilator
    verilator_args.append("-DDISABLE_XPM_ASSERTIONS")
    verilator_args.append("-DOBSOLETE")
    verilator_args.append("-DONESPIN")
    verilator_args.append("--bbox-unsup")
    vivado_path = os.environ["VIVADO_PATH"]
    # additional SystemVerilog modules to make XPMs work with Verilator
    xpm_memory = f"{vivado_path}/data/ip/xpm/xpm_memory/hdl/xpm_memory.sv"
    xpm_cdc = f"{vivado_path}/data/ip/xpm/xpm_cdc/hdl/xpm_cdc.sv"
    xpm_fifo = f"{vivado_path}/data/ip/xpm/xpm_fifo/hdl/xpm_fifo.sv"

    sim = PyVerilator.build(
        [top_module_file_name, xpm_fifo, xpm_memory, xpm_cdc],
        verilog_path=[vivado_stitch_proj_dir, verilog_header_dir],
        build_dir=build_dir,
        trace_depth=get_rtlsim_trace_depth(),
        top_module_name=top_module_name,
        auto_eval=False,
        read_internal_signals=read_internal_signals,
        extra_args=verilator_args + extra_verilator_args,
    )
    return sim
