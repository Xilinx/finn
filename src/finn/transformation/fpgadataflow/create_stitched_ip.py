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

import json
import multiprocessing as mp
import os
import subprocess
import warnings

from finn.custom_op.registry import getCustomOp
from finn.transformation.base import Transformation
from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)
from finn.util.basic import get_num_default_workers, make_build_dir
from finn.util.fpgadataflow import is_fpgadataflow_node


def is_external_input(model, node, i):
    # indicate whether input i of node should be made external
    # True only if input is unconnected and has no initializer
    # Only esception is second input of FC layers when mem_mode is external
    node_inst = getCustomOp(node)
    producer = model.find_producer(node.input[i])
    if producer is None:
        if model.get_initializer(node.input[i]) is None:
            return True
        else:
            if node.op_type == "StreamingFCLayer_Batch":
                if node_inst.get_nodeattr("mem_mode") == "external":
                    return True
    return False


def is_external_output(model, node, i):
    # indicate whether output i of node should be made external
    # True only if output is unconnected
    consumers = model.find_consumers(node.output[i])
    if consumers is None:
        return True
    return False


class CreateStitchedIP(Transformation):
    """Create a Vivado IP Block Design project from all the generated IPs of a
    graph. All nodes in the graph must have the fpgadataflow backend attribute,
    and the PrepareIP transformation must have been previously run on
    the graph. The resulting block design is also packaged as IP. The
    transformation gets the fpgapart as a string.

    Outcome if successful: sets the vivado_stitch_proj attribute in the ONNX
    ModelProto's metadata_props field, with the created project dir as the
    value. A make_project.tcl script is also placed under the same folder,
    which is called to instantiate the per-layer IPs and stitch them together.
    The packaged block design IP can be found under the ip subdirectory.
    """

    def __init__(self, fpgapart, clk_ns, ip_name="finn_design", vitis=False):
        super().__init__()
        self.fpgapart = fpgapart
        self.clk_ns = clk_ns
        self.ip_name = ip_name
        self.vitis = vitis
        self.has_aximm = False
        self.has_m_axis = False
        self.m_axis_idx = 0
        self.has_s_axis = False
        self.s_axis_idx = 0
        self.clock_reset_are_external = False
        self.create_cmds = []
        self.connect_cmds = []
        # keep track of top-level interface names
        self.intf_names = {
            "clk": [],
            "rst": [],
            "s_axis": [],
            "m_axis": [],
            "aximm": [],
            "axilite": [],
        }

    def connect_clk_rst(self, node):
        inst_name = node.name
        node_inst = getCustomOp(node)
        clock_intf_name = node_inst.get_verilog_top_module_intf_names()["clk"][0]
        reset_intf_name = node_inst.get_verilog_top_module_intf_names()["rst"][0]
        # make clock and reset external, if they aren't already
        if not self.clock_reset_are_external:
            self.connect_cmds.append(
                "make_bd_pins_external [get_bd_pins %s/%s]"
                % (inst_name, clock_intf_name)
            )
            self.connect_cmds.append("set_property name ap_clk [get_bd_ports ap_clk_0]")
            self.connect_cmds.append(
                "make_bd_pins_external [get_bd_pins %s/%s]"
                % (inst_name, reset_intf_name)
            )
            self.connect_cmds.append(
                "set_property name ap_rst_n [get_bd_ports ap_rst_n_0]"
            )
            self.clock_reset_are_external = True
            self.intf_names["clk"] = ["ap_clk"]
            self.intf_names["rst"] = ["ap_rst_n"]
        # otherwise connect clock and reset
        else:
            self.connect_cmds.append(
                "connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins %s/%s]"
                % (inst_name, reset_intf_name)
            )
            self.connect_cmds.append(
                "connect_bd_net [get_bd_ports ap_clk] [get_bd_pins %s/%s]"
                % (inst_name, clock_intf_name)
            )

    def connect_axi(self, node):
        inst_name = node.name
        node_inst = getCustomOp(node)
        axilite_intf_name = node_inst.get_verilog_top_module_intf_names()["axilite"]
        aximm_intf_name = node_inst.get_verilog_top_module_intf_names()["aximm"]
        if len(axilite_intf_name) != 0:
            self.connect_cmds.append(
                "make_bd_intf_pins_external "
                "[get_bd_intf_pins %s/%s]" % (inst_name, axilite_intf_name[0])
            )
            ext_if_name = "%s_%d" % (
                axilite_intf_name[0],
                len(self.intf_names["axilite"]),
            )
            self.intf_names["axilite"].append(ext_if_name)
        if len(aximm_intf_name) != 0:
            self.connect_cmds.append(
                "make_bd_intf_pins_external [get_bd_intf_pins %s/%s]"
                % (inst_name, aximm_intf_name[0][0])
            )
            self.connect_cmds.append(
                "set_property name m_axi_gmem0 [get_bd_intf_ports m_axi_gmem_0]"
            )
            self.intf_names["aximm"] = [("m_axi_gmem0", aximm_intf_name[0][1])]
            assert self.has_aximm is False, "Currently limited to one AXI-MM interface"
            self.has_aximm = True

    def connect_m_axis_external(self, node, idx=None):
        inst_name = node.name
        node_inst = getCustomOp(node)
        output_intf_names = node_inst.get_verilog_top_module_intf_names()["m_axis"]
        # make output axis external
        for i in range(len(output_intf_names)):
            if idx is not None and idx != i:
                continue
            output_intf_name = output_intf_names[i][0]
            self.connect_cmds.append(
                "make_bd_intf_pins_external [get_bd_intf_pins %s/%s]"
                % (inst_name, output_intf_name)
            )
            self.connect_cmds.append(
                "set_property name m_axis_%d [get_bd_intf_ports %s_0]"
                % (self.m_axis_idx, output_intf_name)
            )
            self.has_m_axis = True
            self.intf_names["m_axis"].append(
                ("m_axis_%d" % self.m_axis_idx, output_intf_names[i][1])
            )
            self.m_axis_idx += 1

    def connect_s_axis_external(self, node, idx=None):
        inst_name = node.name
        node_inst = getCustomOp(node)
        input_intf_names = node_inst.get_verilog_top_module_intf_names()["s_axis"]
        # make input axis external
        for i in range(len(input_intf_names)):
            if idx is not None and idx != i:
                continue
            input_intf_name = input_intf_names[i][0]
            self.connect_cmds.append(
                "make_bd_intf_pins_external [get_bd_intf_pins %s/%s]"
                % (inst_name, input_intf_name)
            )
            self.connect_cmds.append(
                "set_property name s_axis_%d [get_bd_intf_ports %s_0]"
                % (self.s_axis_idx, input_intf_name)
            )
            self.has_s_axis = True
            self.intf_names["s_axis"].append(
                ("s_axis_%d" % self.s_axis_idx, input_intf_names[i][1])
            )
            self.s_axis_idx += 1

    def apply(self, model):
        # ensure non-relative readmemh .dat files
        model = model.transform(ReplaceVerilogRelPaths())
        ip_dirs = ["list"]
        # add RTL streamer IP
        ip_dirs.append("/workspace/finn/finn-rtllib/memstream")
        if model.graph.node[0].op_type not in ["StreamingFIFO", "IODMA"]:
            warnings.warn(
                """First node is not StreamingFIFO or IODMA.
                You may experience incorrect stitched-IP rtlsim or hardware
                behavior. It is strongly recommended to insert FIFOs prior to
                calling CreateStitchedIP."""
            )
        # ensure that all nodes are fpgadataflow, and that IPs are generated
        for node in model.graph.node:
            assert is_fpgadataflow_node(
                node
            ), "All nodes must be FINN fpgadataflow nodes."
            node_inst = getCustomOp(node)
            ip_dir_value = node_inst.get_nodeattr("ip_path")
            assert os.path.isdir(ip_dir_value), "IP generation directory doesn't exist."
            ip_dirs += [ip_dir_value]
            self.create_cmds += node_inst.code_generation_ipi()
            self.connect_clk_rst(node)
            self.connect_axi(node)
            for i in range(len(node.input)):
                if is_external_input(model, node, i):
                    self.connect_s_axis_external(node, idx=i)
                else:
                    producer = model.find_producer(node.input[i])
                    if producer is None:
                        continue
                    j = list(producer.output).index(node.input[i])
                    src_intf_name = getCustomOp(
                        producer
                    ).get_verilog_top_module_intf_names()["m_axis"][j][0]
                    dst_intf_name = node_inst.get_verilog_top_module_intf_names()[
                        "s_axis"
                    ][i][0]
                    self.connect_cmds.append(
                        "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                        "[get_bd_intf_pins %s/%s]"
                        % (producer.name, src_intf_name, node.name, dst_intf_name)
                    )
            for i in range(len(node.output)):
                if is_external_output(model, node, i):
                    self.connect_m_axis_external(node, idx=i)

        # create a temporary folder for the project
        prjname = "finn_vivado_stitch_proj"
        vivado_stitch_proj_dir = make_build_dir(prefix="vivado_stitch_proj_")
        model.set_metadata_prop("vivado_stitch_proj", vivado_stitch_proj_dir)
        # start building the tcl script
        tcl = []
        # create vivado project
        tcl.append(
            "create_project %s %s -part %s"
            % (prjname, vivado_stitch_proj_dir, self.fpgapart)
        )
        # add all the generated IP dirs to ip_repo_paths
        ip_dirs_str = " ".join(ip_dirs)
        tcl.append("set_property ip_repo_paths [%s] [current_project]" % ip_dirs_str)
        tcl.append("update_ip_catalog")
        # create block design and instantiate all layers
        block_name = self.ip_name
        tcl.append('create_bd_design "%s"' % block_name)
        tcl.extend(self.create_cmds)
        tcl.extend(self.connect_cmds)
        fclk_mhz = 1 / (self.clk_ns * 0.001)
        fclk_hz = fclk_mhz * 1000000
        model.set_metadata_prop("clk_ns", str(self.clk_ns))
        tcl.append("set_property CONFIG.FREQ_HZ %f [get_bd_ports /ap_clk]" % fclk_hz)
        tcl.append("regenerate_bd_layout")
        tcl.append("validate_bd_design")
        tcl.append("save_bd_design")
        # create wrapper hdl (for rtlsim later on)
        bd_base = "%s/%s.srcs/sources_1/bd/%s" % (
            vivado_stitch_proj_dir,
            prjname,
            block_name,
        )
        bd_filename = "%s/%s.bd" % (bd_base, block_name)
        tcl.append("make_wrapper -files [get_files %s] -top" % bd_filename)
        wrapper_filename = "%s/hdl/%s_wrapper.v" % (bd_base, block_name)
        tcl.append("add_files -norecurse %s" % wrapper_filename)
        model.set_metadata_prop("wrapper_filename", wrapper_filename)
        # synthesize to DCP and export stub, DCP and constraints
        if self.vitis:
            tcl.append(
                "set_property SYNTH_CHECKPOINT_MODE Hierarchical [ get_files %s ]"
                % bd_filename
            )
            tcl.append(
                "set_property -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} "
                "-value {-mode out_of_context} -objects [get_runs synth_1]"
            )
            num_workers = get_num_default_workers()
            assert num_workers >= 0, "Number of workers must be nonnegative."
            if num_workers == 0:
                num_workers = mp.cpu_count()
            tcl.append("launch_runs synth_1 -jobs %s" % str(num_workers))
            tcl.append("wait_on_run [get_runs synth_1]")
            tcl.append("open_run synth_1 -name synth_1")
            tcl.append("write_verilog -force -mode synth_stub %s.v" % block_name)
            tcl.append("write_checkpoint %s.dcp" % block_name)
            tcl.append("write_xdc %s.xdc" % block_name)
            tcl.append("report_utilization -file %s_partition_util.rpt" % block_name)
        # export block design itself as an IP core
        block_vendor = "xilinx_finn"
        block_library = "finn"
        block_vlnv = "%s:%s:%s:1.0" % (block_vendor, block_library, block_name)
        model.set_metadata_prop("vivado_stitch_vlnv", block_vlnv)
        model.set_metadata_prop("vivado_stitch_ifnames", json.dumps(self.intf_names))
        tcl.append(
            (
                "ipx::package_project -root_dir %s/ip -vendor %s "
                "-library %s -taxonomy /UserIP -module %s -import_files"
            )
            % (vivado_stitch_proj_dir, block_vendor, block_library, block_name)
        )
        tcl.append("set_property core_revision 2 [ipx::find_open_core %s]" % block_vlnv)
        tcl.append("ipx::create_xgui_files [ipx::find_open_core %s]" % block_vlnv)
        # mark bus interface params as user-resolvable to avoid FREQ_MHZ mismatches
        tcl.append(
            "set_property value_resolve_type user [ipx::get_bus_parameters "
            "-of [ipx::get_bus_interfaces -of [ipx::current_core ]]]"
        )
        # if targeting Vitis, add some properties to the IP
        if self.vitis:
            # replace source code with dcp
            tcl.append(
                "set_property sdx_kernel true [ipx::find_open_core %s]" % block_vlnv
            )
            tcl.append(
                "set_property sdx_kernel_type rtl [ipx::find_open_core %s]" % block_vlnv
            )
            tcl.append(
                "set_property supported_families { } [ipx::find_open_core %s]"
                % block_vlnv
            )
            tcl.append(
                "set_property xpm_libraries {XPM_CDC XPM_MEMORY XPM_FIFO} "
                "[ipx::find_open_core %s]" % block_vlnv
            )
            tcl.append(
                "set_property auto_family_support_level level_2 "
                "[ipx::find_open_core %s]" % block_vlnv
            )
            # remove all files from synthesis and sim groups
            # we'll replace with DCP, stub, and xdc
            tcl.append(
                "ipx::remove_all_file "
                "[ipx::get_file_groups xilinx_anylanguagebehavioralsimulation]"
            )
            tcl.append(
                "ipx::remove_all_file "
                "[ipx::get_file_groups xilinx_anylanguagesynthesis]"
            )
            tcl.append(
                "ipx::remove_file_group "
                "xilinx_anylanguagebehavioralsimulation [ipx::current_core]"
            )
            tcl.append(
                "ipx::remove_file_group "
                "xilinx_anylanguagesynthesis [ipx::current_core]"
            )
            # remove sim and src folders
            tcl.append("file delete -force %s/ip/sim" % vivado_stitch_proj_dir)
            tcl.append("file delete -force %s/ip/src" % vivado_stitch_proj_dir)
            # copy and add DCP, stub, and xdc
            tcl.append("file mkdir %s/ip/dcp" % vivado_stitch_proj_dir)
            tcl.append("file mkdir %s/ip/impl" % vivado_stitch_proj_dir)
            tcl.append(
                "file copy -force %s.dcp %s/ip/dcp"
                % (block_name, vivado_stitch_proj_dir)
            )
            tcl.append(
                "file copy -force %s.xdc %s/ip/impl"
                % (block_name, vivado_stitch_proj_dir)
            )
            tcl.append("ipx::add_file_group xilinx_implementation [ipx::current_core]")
            tcl.append(
                "ipx::add_file impl/%s.xdc [ipx::get_file_groups xilinx_implementation]"
                % block_name
            )
            tcl.append(
                "set_property used_in [list implementation] "
                "[ipx::get_files impl/%s.xdc "
                "-of_objects [ipx::get_file_groups xilinx_implementation]]" % block_name
            )
            tcl.append(
                "ipx::add_file_group " "xilinx_synthesischeckpoint [ipx::current_core]"
            )
            tcl.append(
                "ipx::add_file dcp/%s.dcp "
                "[ipx::get_file_groups xilinx_synthesischeckpoint]" % block_name
            )
            tcl.append(
                "ipx::add_file_group xilinx_simulationcheckpoint [ipx::current_core]"
            )
            tcl.append(
                "ipx::add_file dcp/%s.dcp "
                "[ipx::get_file_groups xilinx_simulationcheckpoint]" % block_name
            )
        tcl.append("ipx::update_checksums [ipx::find_open_core %s]" % block_vlnv)
        tcl.append("ipx::save_core [ipx::find_open_core %s]" % block_vlnv)
        # export list of used Verilog files (for rtlsim later on)
        tcl.append(
            "set all_v_files [get_files -filter {FILE_TYPE == Verilog "
            + "&& USED_IN_SYNTHESIS == 1} ]"
        )
        v_file_list = "%s/all_verilog_srcs.txt" % vivado_stitch_proj_dir
        tcl.append("set fp [open %s w]" % v_file_list)
        # write each verilog filename to all_verilog_srcs.txt
        tcl.append("foreach vf $all_v_files {puts $fp $vf}")
        tcl.append("close $fp")
        # write the project creator tcl script
        tcl_string = "\n".join(tcl) + "\n"
        with open(vivado_stitch_proj_dir + "/make_project.tcl", "w") as f:
            f.write(tcl_string)
        # create a shell script and call Vivado
        make_project_sh = vivado_stitch_proj_dir + "/make_project.sh"
        working_dir = os.environ["PWD"]
        with open(make_project_sh, "w") as f:
            f.write("#!/bin/bash \n")
            f.write("cd {}\n".format(vivado_stitch_proj_dir))
            f.write("vivado -mode batch -source make_project.tcl\n")
            f.write("cd {}\n".format(working_dir))
        bash_command = ["bash", make_project_sh]
        process_compile = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
        process_compile.communicate()
        return (model, False)
