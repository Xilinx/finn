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
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.util.basic import get_num_default_workers
from shutil import copytree

from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)
from finn.util.basic import make_build_dir
from finn.util.fpgadataflow import is_fpgadataflow_node


def is_external_input(model, node, i):
    # indicate whether input i of node should be made external
    # True only if input is unconnected and has no initializer
    # Only esception is second input of FC layers when mem_mode is external
    node_inst = getCustomOp(node)
    op_type = node_inst.base_op_type()
    producer = model.find_producer(node.input[i])
    if producer is None:
        if model.get_initializer(node.input[i]) is None:
            return True
        else:
            if op_type == "MatrixVectorActivation":
                if node_inst.get_nodeattr("mem_mode") == "external":
                    return True
    return False


def is_external_output(model, node, i):
    # indicate whether output i of node should be made external
    # True only if output is unconnected
    consumers = model.find_consumers(node.output[i])
    if consumers == []:
        # TODO should ideally check if tensor is in top-level
        # outputs
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

    def __init__(self, fpgapart, clk_ns, ip_name="finn_design", vitis=False, signature=[]):
        super().__init__()
        self.fpgapart = fpgapart
        self.clk_ns = clk_ns
        self.ip_name = ip_name
        self.vitis = vitis
        self.signature = signature
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
                "make_bd_pins_external [get_bd_pins %s/%s]" % (inst_name, clock_intf_name)
            )
            self.connect_cmds.append("set_property name ap_clk [get_bd_ports ap_clk_0]")
            self.connect_cmds.append(
                "make_bd_pins_external [get_bd_pins %s/%s]" % (inst_name, reset_intf_name)
            )
            self.connect_cmds.append("set_property name ap_rst_n [get_bd_ports ap_rst_n_0]")
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
            ext_if_name = "m_axi_gmem%d" % (len(self.intf_names["aximm"]))
            self.connect_cmds.append(
                "set_property name %s [get_bd_intf_ports m_axi_gmem_0]" % ext_if_name
            )
            self.connect_cmds.append("assign_bd_address")
            seg_name = "%s/Data_m_axi_gmem/SEG_%s_Reg" % (inst_name, ext_if_name)
            self.connect_cmds.append("set_property offset 0 [get_bd_addr_segs {%s}]" % (seg_name))
            # TODO should propagate this information from the node instead of 4G
            self.connect_cmds.append("set_property range 4G [get_bd_addr_segs {%s}]" % (seg_name))
            self.intf_names["aximm"] = [(ext_if_name, aximm_intf_name[0][1])]
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
                "make_bd_intf_pins_external [get_bd_intf_pins %s/%s]" % (inst_name, input_intf_name)
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

    def connect_ap_none_external(self, node):
        inst_name = node.name
        node_inst = getCustomOp(node)
        input_intf_names = node_inst.get_verilog_top_module_intf_names()["ap_none"]
        # make external
        for i in range(len(input_intf_names)):
            input_intf_name = input_intf_names[i]
            self.connect_cmds.append(
                "make_bd_pins_external [get_bd_pins %s/%s]" % (inst_name, input_intf_name)
            )
            self.connect_cmds.append(
                "set_property name %s [get_bd_ports %s_0]" % (input_intf_name, input_intf_name)
            )

    def insert_signature(self, checksum_count):
        signature_vlnv = "AMD:user:axi_info_top:1.0"
        signature_name = "axi_info_top0"
        self.create_cmds.append(
            "create_bd_cell -type ip -vlnv %s %s" % (signature_vlnv, signature_name)
        )
        self.create_cmds.append(
            "set_property -dict [list "
            "CONFIG.SIG_CUSTOMER {%s} "
            "CONFIG.SIG_APPLICATION {%s} "
            "CONFIG.VERSION {%s} "
            "CONFIG.CHECKSUM_COUNT {%s} "
            "] [get_bd_cells %s]"
            % (
                self.signature[0],
                self.signature[1],
                self.signature[2],
                checksum_count,
                signature_name,
            )
        )
        # set clk and reset
        self.connect_cmds.append(
            "connect_bd_net [get_bd_ports ap_clk] [get_bd_pins %s/ap_clk]" % signature_name
        )
        self.connect_cmds.append(
            "connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins %s/ap_rst_n]" % signature_name
        )
        fclk_mhz = 1 / (self.clk_ns * 0.001)
        fclk_hz = fclk_mhz * 1000000
        self.connect_cmds.append(
            "set_property -dict [list "
            "CONFIG.FREQ_HZ {%f} "
            "CONFIG.CLK_DOMAIN {ap_clk} "
            "] [get_bd_intf_pins %s/s_axi]"
            % (
                fclk_hz,
                signature_name,
            )
        )
        # make axilite interface external
        self.connect_cmds.append(
            "make_bd_intf_pins_external [get_bd_intf_pins %s/s_axi]" % signature_name
        )
        self.connect_cmds.append("set_property name s_axilite_info [get_bd_intf_ports s_axi_0]")
        self.connect_cmds.append("assign_bd_address")

    def apply(self, model):
        # ensure non-relative readmemh .dat files
        model = model.transform(ReplaceVerilogRelPaths())
        ip_dirs = ["list"]
        # add RTL streamer IP
        ip_dirs.append("$::env(FINN_ROOT)/finn-rtllib/memstream")
        if self.signature:
            ip_dirs.append("$::env(FINN_ROOT)/finn-rtllib/axi_info")
        if model.graph.node[0].op_type not in ["StreamingFIFO", "IODMA_hls"]:
            warnings.warn(
                """First node is not StreamingFIFO or IODMA.
                You may experience incorrect stitched-IP rtlsim or hardware
                behavior. It is strongly recommended to insert FIFOs prior to
                calling CreateStitchedIP."""
            )
        if model.graph.node[0].op_type == "StreamingFIFO":
            firstfifo = getCustomOp(model.graph.node[0])
            if firstfifo.get_nodeattr("impl_style") == "vivado":
                warnings.warn(
                    """First FIFO has impl_style=vivado, which may cause
                    simulation glitches (e.g. dropping the first input sample
                    after reset)."""
                )
        for node in model.graph.node:
            # ensure that all nodes are fpgadataflow, and that IPs are generated
            assert is_fpgadataflow_node(node), "All nodes must be FINN fpgadataflow nodes."
            node_inst = getCustomOp(node)
            ip_dir_value = node_inst.get_nodeattr("ip_path")
            assert os.path.isdir(ip_dir_value), "IP generation directory doesn't exist."
            ip_dirs += [ip_dir_value]
            self.create_cmds += node_inst.code_generation_ipi()
            self.connect_clk_rst(node)
            self.connect_ap_none_external(node)
            self.connect_axi(node)
            for i in range(len(node.input)):
                if not is_external_input(model, node, i):
                    producer = model.find_producer(node.input[i])
                    if producer is None:
                        continue
                    j = list(producer.output).index(node.input[i])
                    src_intf_name = getCustomOp(producer).get_verilog_top_module_intf_names()[
                        "m_axis"
                    ][j][0]
                    dst_intf_name = node_inst.get_verilog_top_module_intf_names()["s_axis"][i][0]
                    self.connect_cmds.append(
                        "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                        "[get_bd_intf_pins %s/%s]"
                        % (producer.name, src_intf_name, node.name, dst_intf_name)
                    )

        # process external inputs and outputs in top-level graph input order
        for input in model.graph.input:
            inp_name = input.name
            inp_cons = model.find_consumers(inp_name)
            assert inp_cons != [], "No consumer for input " + inp_name
            assert len(inp_cons) == 1, "Multiple consumers for input " + inp_name
            node = inp_cons[0]
            node_inst = getCustomOp(node)
            for i in range(len(node.input)):
                if node.input[i] == inp_name:
                    self.connect_s_axis_external(node, idx=i)
        for output in model.graph.output:
            out_name = output.name
            node = model.find_producer(out_name)
            assert node is not None, "No producer for output " + out_name
            node_inst = getCustomOp(node)
            for i in range(len(node.output)):
                if node.output[i] == out_name:
                    self.connect_m_axis_external(node, idx=i)

        if self.signature:
            # extract number of checksum layer from graph
            checksum_layers = model.get_nodes_by_op_type("checksum")
            self.insert_signature(len(checksum_layers))

        # create a temporary folder for the project
        prjname = "finn_vivado_stitch_proj"
        vivado_stitch_proj_dir = make_build_dir(prefix="vivado_stitch_proj_")
        model.set_metadata_prop("vivado_stitch_proj", vivado_stitch_proj_dir)
        # start building the tcl script
        tcl = []
        # create vivado project
        tcl.append(
            "create_project %s %s -part %s" % (prjname, vivado_stitch_proj_dir, self.fpgapart)
        )
        # no warnings on long module names
        tcl.append("set_msg_config -id {[BD 41-1753]} -suppress")
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
        tcl.append("set_property CONFIG.FREQ_HZ %d [get_bd_ports /ap_clk]" % round(fclk_hz))
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
        tcl.append("set_property top %s_wrapper [current_fileset]" % block_name)
        # synthesize to DCP and export stub, DCP and constraints
        if self.vitis:
            tcl.append(
                "set_property SYNTH_CHECKPOINT_MODE Hierarchical [ get_files %s ]" % bd_filename
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
            tcl.append(
                "report_utilization -hierarchical -hierarchical_depth 5 "
                "-file %s_partition_util.rpt" % block_name
            )
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
        # Allow user to customize clock in deployment of stitched IP
        tcl.append("set_property ipi_drc {ignore_freq_hz true} [ipx::current_core]")
        # in some cases, the IP packager seems to infer an aperture of 64K or 4G,
        # preventing address assignment of the DDR_LOW and/or DDR_HIGH segments
        # the following is a hotfix to remove this aperture during IODMA packaging
        tcl.append(
            "ipx::remove_segment -quiet m_axi_gmem0:APERTURE_0 "
            "[ipx::get_address_spaces m_axi_gmem0 -of_objects [ipx::current_core]]"
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
            tcl.append("set_property sdx_kernel true [ipx::find_open_core %s]" % block_vlnv)
            tcl.append("set_property sdx_kernel_type rtl [ipx::find_open_core %s]" % block_vlnv)
            tcl.append("set_property supported_families { } [ipx::find_open_core %s]" % block_vlnv)
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
            tcl.append("ipx::remove_all_file " "[ipx::get_file_groups xilinx_anylanguagesynthesis]")
            tcl.append(
                "ipx::remove_file_group "
                "xilinx_anylanguagebehavioralsimulation [ipx::current_core]"
            )
            tcl.append("ipx::remove_file_group " "xilinx_anylanguagesynthesis [ipx::current_core]")
            # remove sim and src folders
            tcl.append("file delete -force %s/ip/sim" % vivado_stitch_proj_dir)
            tcl.append("file delete -force %s/ip/src" % vivado_stitch_proj_dir)
            # copy and add DCP, stub, and xdc
            tcl.append("file mkdir %s/ip/dcp" % vivado_stitch_proj_dir)
            tcl.append("file mkdir %s/ip/impl" % vivado_stitch_proj_dir)
            tcl.append("file copy -force %s.dcp %s/ip/dcp" % (block_name, vivado_stitch_proj_dir))
            tcl.append("file copy -force %s.xdc %s/ip/impl" % (block_name, vivado_stitch_proj_dir))
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
            tcl.append("ipx::add_file_group " "xilinx_synthesischeckpoint [ipx::current_core]")
            tcl.append(
                "ipx::add_file dcp/%s.dcp "
                "[ipx::get_file_groups xilinx_synthesischeckpoint]" % block_name
            )
            tcl.append("ipx::add_file_group xilinx_simulationcheckpoint [ipx::current_core]")
            tcl.append(
                "ipx::add_file dcp/%s.dcp "
                "[ipx::get_file_groups xilinx_simulationcheckpoint]" % block_name
            )
        # add a rudimentary driver mdd to get correct ranges in xparameters.h later on
        example_data_dir = os.environ["FINN_ROOT"] + "/src/finn/qnn-data/mdd-data"
        copytree(example_data_dir, vivado_stitch_proj_dir + "/data")

        #####
        # Core Cleanup Operations
        tcl.append(
            """
set core [ipx::current_core]

# Add rudimentary driver
file copy -force data ip/
set file_group [ipx::add_file_group -type software_driver {} $core]
set_property type mdd       [ipx::add_file data/finn_design.mdd $file_group]
set_property type tclSource [ipx::add_file data/finn_design.tcl $file_group]

# Remove all XCI references to subcores
set impl_files [ipx::get_file_groups xilinx_implementation -of $core]
foreach xci [ipx::get_files -of $impl_files {*.xci}] {
    ipx::remove_file [get_property NAME $xci] $impl_files
}

# Construct a single flat memory map for each AXI-lite interface port
foreach port [get_bd_intf_ports -filter {CONFIG.PROTOCOL==AXI4LITE}] {
    set pin $port
    set awidth ""
    while { $awidth == "" } {
        set pins [get_bd_intf_pins -of [get_bd_intf_nets -boundary_type lower -of $pin]]
        set kill [lsearch $pins $pin]
        if { $kill >= 0 } { set pins [lreplace $pins $kill $kill] }
        if { [llength $pins] != 1 } { break }
        set pin [lindex $pins 0]
        set awidth [get_property CONFIG.ADDR_WIDTH $pin]
    }
    if { $awidth == "" } {
       puts "CRITICAL WARNING: Unable to construct address map for $port."
    } {
       set range [expr 2**$awidth]
       set range [expr $range < 4096 ? 4096 : $range]
       puts "INFO: Building address map for $port: 0+:$range"
       set name [get_property NAME $port]
       set addr_block [ipx::add_address_block Reg0 [ipx::add_memory_map $name $core]]
       set_property range $range $addr_block
       set_property slave_memory_map_ref $name [ipx::get_bus_interfaces $name -of $core]
    }
}

# Finalize and Save
ipx::update_checksums $core
ipx::save_core $core

# Remove stale subcore references from component.xml
file rename -force ip/component.xml ip/component.bak
set ifile [open ip/component.bak r]
set ofile [open ip/component.xml w]
set buf [list]
set kill 0
while { [eof $ifile] != 1 } {
    gets $ifile line
    if { [string match {*<spirit:fileSet>*} $line] == 1 } {
        foreach l $buf { puts $ofile $l }
        set buf [list $line]
    } elseif { [llength $buf] > 0 } {
        lappend buf $line

        if { [string match {*</spirit:fileSet>*} $line] == 1 } {
            if { $kill == 0 } { foreach l $buf { puts $ofile $l } }
            set buf [list]
            set kill 0
        } elseif { [string match {*<xilinx:subCoreRef>*} $line] == 1 } {
            set kill 1
        }
    } else {
        puts $ofile $line
    }
}
close $ifile
close $ofile
"""
        )

        # export list of used Verilog files (for rtlsim later on)
        tcl.append(
            "set all_v_files [get_files -filter {USED_IN_SYNTHESIS == 1 "
            + "&& (FILE_TYPE == Verilog || FILE_TYPE == SystemVerilog "
            + '|| FILE_TYPE =="Verilog Header")}]'
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
        # wrapper may be created in different location depending on Vivado version
        if not os.path.isfile(wrapper_filename):
            # check in alternative location (.gen instead of .srcs)
            wrapper_filename_alt = wrapper_filename.replace(".srcs", ".gen")
            if os.path.isfile(wrapper_filename_alt):
                model.set_metadata_prop("wrapper_filename", wrapper_filename_alt)
            else:
                raise Exception(
                    """CreateStitchedIP failed, no wrapper HDL found under %s or %s.
                    Please check logs under the parent directory."""
                    % (wrapper_filename, wrapper_filename_alt)
                )

        return (model, False)
