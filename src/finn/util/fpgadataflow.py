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

try:
    from pyverilator import PyVerilator
except ModuleNotFoundError:
    PyVerilator = None
from finn.util.basic import get_by_name, make_build_dir, get_rtlsim_trace_depth


class IPGenBuilder:
    """Builds the bash script to generate IP blocks using Vivado HLS."""

    def __init__(self):
        self.tcl_script = ""
        self.ipgen_path = ""
        self.code_gen_dir = ""
        self.ipgen_script = ""

    def append_tcl(self, tcl_script):
        """Sets member variable "tcl_script" to given tcl script."""
        self.tcl_script = tcl_script

    def set_ipgen_path(self, path):
        """Sets member variable ipgen_path to given path."""
        self.ipgen_path = path

    def build(self, code_gen_dir):
        """Builds the bash script with given parameters and saves it in given folder.
        To guarantee the generation in the correct folder the bash script contains a
        cd command."""
        self.code_gen_dir = code_gen_dir
        self.ipgen_script = str(self.code_gen_dir) + "/ipgen.sh"
        working_dir = os.environ["PWD"]
        f = open(self.ipgen_script, "w")
        f.write("#!/bin/bash \n")
        f.write("cd {}\n".format(code_gen_dir))
        f.write("vivado_hls {}\n".format(self.tcl_script))
        f.write("cd {}\n".format(working_dir))
        f.close()
        bash_command = ["bash", self.ipgen_script]
        process_compile = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
        process_compile.communicate()


def pyverilate_stitched_ip(model):
    "Given a model with stitched IP, return a PyVerilator sim object."
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
    all_verilog_files = list(set(filter(lambda x: x.endswith(".v"), all_verilog_srcs)))

    # remove all but one instances of regslice_core.v
    filtered_verilog_files = []
    remove_entry = False
    for vfile in all_verilog_files:
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
                wf.write(rf.read())

    sim = PyVerilator.build(
        top_module_file_name,
        verilog_path=[vivado_stitch_proj_dir],
        build_dir=build_dir,
        trace_depth=get_rtlsim_trace_depth(),
        top_module_name=top_module_name,
        auto_eval=False,
    )
    return sim


def pyverilate_get_liveness_threshold_cycles():
    """Return the number of no-output cycles rtlsim will wait before assuming
    the simulation is not finishing and throwing an exception."""

    return int(os.getenv("LIVENESS_THRESHOLD", 10000))


def is_fpgadataflow_node(node):
    """Returns True if given node is fpgadataflow node. Otherwise False."""
    is_node = False
    if node is not None:
        if node.domain == "finn":
            n_backend = get_by_name(node.attribute, "backend")
            if n_backend is not None:
                backend_value = n_backend.s.decode("UTF-8")
                if backend_value == "fpgadataflow":
                    is_node = True

    return is_node


def rtlsim_multi_io(sim, io_dict, num_out_values, trace_file=""):
    """Runs the pyverilator simulation by passing the input values to the simulation,
    toggle the clock and observing the execution time. Function contains also an
    observation loop that can abort the simulation if no output value is produced
    after a set number of cycles. Can handle multiple i/o streams. See function
    implementation for details on how the top-level signals should be named.

    sim: the PyVerilator object for simulation
    io_dict: a dict of dicts in the following format:
            {"inputs" : {"in0" : <input_data>, "in1" : <input_data>},
             "outputs" : {"out0" : [], "out1" : []} }
            <input_data> is a list of Python arbitrary-precision ints indicating
            what data to push into the simulation, and the output lists are
            similarly filled when the simulation is complete
    num_out_values: number of total values to be read from the simulation to
                    finish the simulation and return.

    returns: number of clock cycles elapsed for completion

    """

    if trace_file != "":
        sim.start_vcd_trace(trace_file)

    for outp in io_dict["outputs"]:
        sim.io[outp + "_V_V_TREADY"] = 1

    # observe if output is completely calculated
    # total_cycle_count will contain the number of cycles the calculation ran
    output_done = False
    total_cycle_count = 0
    output_count = 0
    old_output_count = 0

    # avoid infinite looping of simulation by aborting when there is no change in
    # output values after 100 cycles
    no_change_count = 0
    liveness_threshold = pyverilate_get_liveness_threshold_cycles()

    while not (output_done):
        for inp in io_dict["inputs"]:
            inputs = io_dict["inputs"][inp]
            sim.io[inp + "_V_V_TVALID"] = 1 if len(inputs) > 0 else 0
            sim.io[inp + "_V_V_TDATA"] = inputs[0] if len(inputs) > 0 else 0
            if sim.io[inp + "_V_V_TREADY"] == 1 and sim.io[inp + "_V_V_TVALID"] == 1:
                inputs = inputs[1:]
            io_dict["inputs"][inp] = inputs

        for outp in io_dict["outputs"]:
            outputs = io_dict["outputs"][outp]
            if sim.io[outp + "_V_V_TVALID"] == 1 and sim.io[outp + "_V_V_TREADY"] == 1:
                outputs = outputs + [sim.io[outp + "_V_V_TDATA"]]
                output_count += 1
            io_dict["outputs"][outp] = outputs

        sim.io.ap_clk = 1
        sim.io.ap_clk = 0

        total_cycle_count = total_cycle_count + 1

        if output_count == old_output_count:
            no_change_count = no_change_count + 1
        else:
            no_change_count = 0
            old_output_count = output_count

        # check if all expected output words received
        if output_count == num_out_values:
            output_done = True

        # end sim on timeout
        if no_change_count == liveness_threshold:
            if trace_file != "":
                sim.flush_vcd_trace()
                sim.stop_vcd_trace()
            raise Exception(
                "Error in simulation! Takes too long to produce output. "
                "Consider setting the LIVENESS_THRESHOLD env.var. to a "
                "larger value."
            )

    if trace_file != "":
        sim.flush_vcd_trace()
        sim.stop_vcd_trace()

    return total_cycle_count
