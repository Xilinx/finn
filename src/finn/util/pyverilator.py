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

import numpy as np
import os
import shutil
from pyverilator import PyVerilator
from qonnx.custom_op.registry import getCustomOp

from finn.util.basic import (
    get_rtlsim_trace_depth,
    launch_process_helper,
    make_build_dir,
)


def make_single_source_file(filtered_verilog_files, target_file):
    """Dump all Verilog code used by stitched IP into a single file.
    This is because large models with many files require a verilator
    command line too long for bash on most systems"""

    # concatenate all verilog code into a single file
    with open(target_file, "w") as wf:
        for vfile in filtered_verilog_files:
            with open(vfile) as rf:
                wf.write("//Added from " + vfile + "\n\n")
                lines = rf.read()
                for line in lines.split("\n"):
                    # break down too-long lines, Verilator complains otherwise
                    if len(line) > 20000:
                        line = line.replace("&", "\n&")
                    wf.write("\n" + line)


def prepare_stitched_ip_for_verilator(model):
    """Prepare sources from given stitched IP for verilator simulation, including
    generating a single source file and replacing certain Vivado infrastructure
    headers with Verilator-compatible ones"""

    vivado_stitch_proj_dir = model.get_metadata_prop("vivado_stitch_proj")
    with open(vivado_stitch_proj_dir + "/all_verilog_srcs.txt", "r") as f:
        all_verilog_srcs = f.read().split()

    def file_to_dir(x):
        return os.path.dirname(os.path.realpath(x))

    def file_to_basename(x):
        return os.path.basename(os.path.realpath(x))

    top_module_file_name = file_to_basename(model.get_metadata_prop("wrapper_filename"))

    # dump all Verilog code to a single file
    # this is because large models with many files require
    # a verilator command line too long for bash on most systems
    # NOTE: there are duplicates in this list, and some files
    # are identical but in multiple directories (regslice_core.v)

    # remove duplicates from list by doing list -> set -> list
    src_exts = [".v", ".sv"]

    all_verilog_files = list(
        set(filter(lambda x: any(map(lambda y: x.endswith(y), src_exts)), all_verilog_srcs))
    )

    verilog_header_dir = vivado_stitch_proj_dir + "/pyverilator_vh"
    os.makedirs(verilog_header_dir, exist_ok=True)

    # use custom version of axis infrastructure vh
    # to enable Verilator to simulate AMD/Xilinx components (e.g DWC)
    custom_vh = os.environ["FINN_ROOT"] + "/src/finn/qnn-data/verilog/custom_axis_infrastructure.vh"
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
    for vfile in all_verilog_files:
        if "regslice_core" in vfile:
            if not remove_entry:
                filtered_verilog_files.append(vfile)
            remove_entry = True
        elif "swg_pkg" in vfile:
            continue
        else:
            filtered_verilog_files.append(vfile)

    target_file = vivado_stitch_proj_dir + "/" + top_module_file_name
    make_single_source_file(filtered_verilog_files, target_file)

    return vivado_stitch_proj_dir


def verilator_fifosim(model, n_inputs, max_iters=100000000):
    """Create a Verilator model of stitched IP and use a simple C++
    driver to drive the input stream. Useful for FIFO sizing, latency
    and throughput measurement."""

    vivado_stitch_proj_dir = prepare_stitched_ip_for_verilator(model)
    verilog_header_dir = vivado_stitch_proj_dir + "/pyverilator_vh"
    build_dir = make_build_dir("verilator_fifosim_")
    fifosim_cpp_fname = os.environ["FINN_ROOT"] + "/src/finn/qnn-data/cpp/verilator_fifosim.cpp"
    with open(fifosim_cpp_fname, "r") as f:
        fifosim_cpp_template = f.read()
    assert len(model.graph.input) == 1, "Only a single input stream is supported"
    assert len(model.graph.output) == 1, "Only a single output stream is supported"
    iname = model.graph.input[0].name
    first_node = model.find_consumer(iname)
    oname = model.graph.output[0].name
    last_node = model.find_producer(oname)
    assert (first_node is not None) and (last_node is not None), "Failed to find first/last nodes"
    fnode_inst = getCustomOp(first_node)
    lnode_inst = getCustomOp(last_node)
    ishape_folded = fnode_inst.get_folded_input_shape()
    oshape_folded = lnode_inst.get_folded_output_shape()

    fifo_log = []
    fifo_log_templ = '    results_file << "maxcount%s" << "\\t" '
    fifo_log_templ += "<< to_string(top->maxcount%s) << endl;"
    fifo_nodes = model.get_nodes_by_op_type("StreamingFIFO")
    fifo_ind = 0
    for fifo_node in fifo_nodes:
        fifo_node = getCustomOp(fifo_node)
        if fifo_node.get_nodeattr("depth_monitor") == 1:
            suffix = "" if fifo_ind == 0 else "_%d" % fifo_ind
            fifo_log.append(fifo_log_templ % (suffix, suffix))
            fifo_ind += 1
    fifo_log = "\n".join(fifo_log)

    template_dict = {
        "ITERS_PER_INPUT": np.prod(ishape_folded[:-1]),
        "ITERS_PER_OUTPUT": np.prod(oshape_folded[:-1]),
        "N_INPUTS": n_inputs,
        "MAX_ITERS": max_iters,
        "FIFO_DEPTH_LOGGING": fifo_log,
    }

    for key, val in template_dict.items():
        fifosim_cpp_template = fifosim_cpp_template.replace(f"@{key}@", str(val))

    with open(build_dir + "/verilator_fifosim.cpp", "w") as f:
        f.write(fifosim_cpp_template)

    which_verilator = shutil.which("verilator")
    if which_verilator is None:
        raise Exception("'verilator' executable not found")

    # add defines to make certain XPM src files work with Verilator
    xpm_args = []
    xpm_args.append("-DDISABLE_XPM_ASSERTIONS")
    xpm_args.append("-DOBSOLETE")
    xpm_args.append("-DONESPIN")
    xpm_args.append("--bbox-unsup")
    vivado_path = os.environ["VIVADO_PATH"]
    # additional SystemVerilog modules to make XPMs work with Verilator
    xpm_memory = f"{vivado_path}/data/ip/xpm/xpm_memory/hdl/xpm_memory.sv"
    xpm_cdc = f"{vivado_path}/data/ip/xpm/xpm_cdc/hdl/xpm_cdc.sv"
    xpm_fifo = f"{vivado_path}/data/ip/xpm/xpm_fifo/hdl/xpm_fifo.sv"
    swg_pkg = os.environ["FINN_ROOT"] + "/finn-rtllib/swg/swg_pkg.sv"
    verilog_file_arg = [swg_pkg, "finn_design_wrapper.v", xpm_memory, xpm_cdc, xpm_fifo]

    verilator_args = [
        "perl",
        which_verilator,
        "-Wno-fatal",
        "-Mdir",
        build_dir,
        "-y",
        vivado_stitch_proj_dir,
        "-y",
        verilog_header_dir,
        "--CFLAGS",
        "--std=c++11",
        "-O3",
        "--x-assign",
        "fast",
        "--x-initial",
        "fast",
        "--noassert",
        "--cc",
        *verilog_file_arg,
        "--top-module",
        "finn_design_wrapper",
        "--exe",
        "verilator_fifosim.cpp",
        "--threads",
        "4",
        *xpm_args,
    ]

    proc_env = os.environ.copy()
    gcc_args = "-O3 -march=native"
    proc_env["OPT_FAST"] = gcc_args
    make_args = [
        "make",
        "-j4",
        "-C",
        build_dir,
        "-f",
        "Vfinn_design_wrapper.mk",
        "Vfinn_design_wrapper",
    ]

    with open(build_dir + "/compile.sh", "w") as f:
        f.write("#!/bin/bash" + "\n")
        f.write("export OPT_FAST='%s'\n" % gcc_args)
        f.write(" ".join(verilator_args) + "\n")
        f.write(" ".join(make_args) + "\n")

    launch_process_helper(verilator_args, cwd=build_dir)
    launch_process_helper(make_args, proc_env=proc_env, cwd=build_dir)

    sim_launch_args = ["./Vfinn_design_wrapper"]
    launch_process_helper(sim_launch_args, cwd=build_dir)

    with open(build_dir + "/results.txt", "r") as f:
        results = f.read().strip().split("\n")
    ret_dict = {}
    for result_line in results:
        key, val = result_line.split("\t")
        ret_dict[key] = int(val)
    return ret_dict


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

    vivado_stitch_proj_dir = prepare_stitched_ip_for_verilator(model)
    verilog_header_dir = vivado_stitch_proj_dir + "/pyverilator_vh"

    def file_to_basename(x):
        return os.path.basename(os.path.realpath(x))

    top_module_file_name = file_to_basename(model.get_metadata_prop("wrapper_filename"))
    top_module_name = top_module_file_name.strip(".v")
    build_dir = make_build_dir("pyverilator_ipstitched_")

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

    swg_pkg = os.environ["FINN_ROOT"] + "/finn-rtllib/swg/swg_pkg.sv"

    sim = PyVerilator.build(
        [swg_pkg, top_module_file_name, xpm_fifo, xpm_memory, xpm_cdc],
        verilog_path=[vivado_stitch_proj_dir, verilog_header_dir],
        build_dir=build_dir,
        trace_depth=get_rtlsim_trace_depth(),
        top_module_name=top_module_name,
        auto_eval=False,
        read_internal_signals=read_internal_signals,
        extra_args=verilator_args + extra_verilator_args,
    )
    return sim
