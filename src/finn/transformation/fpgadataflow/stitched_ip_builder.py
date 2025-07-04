from finn.util.context import Context
from finn.util.kernel_util import get_node_attr
from finn.util import templates
from finn.kernels import Kernel
from finn.kernels import gkr

from qonnx.util.basic import get_num_default_workers
from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper
import os
import json
import multiprocessing as mp
import subprocess
from pathlib import Path
import importlib
from shutil import copytree


def is_external_input(model, node, i, kernel):
    # indicate whether input i of node should be made external
    # True only if input is unconnected and has no initializer
    # Only esception is second input of FC layers when mem_mode is external
    op_type = node.op_type
    producer = model.find_producer(node.input[i])
    if producer is None:
        if model.get_initializer(node.input[i]) is None:
            return True
        else:
            if op_type.startswith("MVAU"):
                if kernel.mem_mode == "external":
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


class StitchedIPBuilder(Transformation):
    """ Takes an ONNX graph and does IP stitching """

    def __init__(self, ctx: Context):
        super().__init__()
        self.ctx = ctx
        self.has_aximm = False
        self.has_m_axis = False
        self.m_axis_idx = 0
        self.has_s_axis = False
        self.s_axis_idx = 0
        self.clock_reset_are_external = False
        self.clock2x_is_external = False
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

    def is_double_pumped(self, node, kernel):
        if node.op_type.startswith("MVAU"):
            try:
                pumped_compute = kernel.pumpedCompute
            except AttributeError:
                pumped_compute = 0
            return pumped_compute or kernel.pumpedMemory

    def connect_clk_rst(self, node, kernel):
        clock_intf_name = kernel.get_verilog_top_module_intf_names()["clk"][0]
        reset_intf_name = kernel.get_verilog_top_module_intf_names()["rst"][0]
        # make clock and reset external, if they aren't already
        if not self.clock_reset_are_external:
            self.connect_cmds.append(
                "make_bd_pins_external [get_bd_pins %s/%s]" % (node.name, clock_intf_name)
            )
            self.connect_cmds.append("set_property name ap_clk [get_bd_ports ap_clk_0]")
            self.connect_cmds.append(
                "make_bd_pins_external [get_bd_pins %s/%s]" % (node.name, reset_intf_name)
            )
            self.connect_cmds.append("set_property name ap_rst_n [get_bd_ports ap_rst_n_0]")
            self.clock_reset_are_external = True
            self.intf_names["clk"] = ["ap_clk"]
            self.intf_names["rst"] = ["ap_rst_n"]
        # otherwise connect clock and reset
        else:
            self.connect_cmds.append(
                "connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins %s/%s]"
                % (node.name, reset_intf_name)
            )
            self.connect_cmds.append(
                "connect_bd_net [get_bd_ports ap_clk] [get_bd_pins %s/%s]"
                % (node.name, clock_intf_name)
            )
        # make clk2x external, if it isn't already and connect clk2x
        if self.is_double_pumped(node, kernel):
            clock2x_intf_name = kernel.get_verilog_top_module_intf_names()["clk2x"][0]
            if not self.clock2x_is_external:
                self.connect_cmds.append(
                    "make_bd_pins_external [get_bd_pins %s/%s]" % (node.name, clock2x_intf_name)
                )
                self.connect_cmds.append("set_property name ap_clk2x [get_bd_ports ap_clk2x_0]")
                self.clock2x_is_external = True
                self.intf_names["clk2x"] = ["ap_clk2x"]
            # otherwise connect clk2x
            else:
                if self.is_double_pumped(node):
                    self.connect_cmds.append(
                        "connect_bd_net [get_bd_ports ap_clk2x] [get_bd_pins %s/%s]"
                        % (node.name, clock2x_intf_name)
                    )

    def connect_axi(self, node, kernel):
        axilite_intf_name = kernel.get_verilog_top_module_intf_names()["axilite"]
        aximm_intf_name = kernel.get_verilog_top_module_intf_names()["aximm"]
        if len(axilite_intf_name) != 0:
            self.connect_cmds.append(
                "make_bd_intf_pins_external "
                "[get_bd_intf_pins %s/%s]" % (node.name, axilite_intf_name[0])
            )
            ext_if_name = "%s_%d" % (
                axilite_intf_name[0],
                len(self.intf_names["axilite"]),
            )
            self.intf_names["axilite"].append(ext_if_name)
        if len(aximm_intf_name) != 0:
            self.connect_cmds.append(
                "make_bd_intf_pins_external [get_bd_intf_pins %s/%s]"
                % (node.names, aximm_intf_name[0][0])
            )
            ext_if_name = "m_axi_gmem%d" % (len(self.intf_names["aximm"]))
            self.connect_cmds.append(
                "set_property name %s [get_bd_intf_ports m_axi_gmem_0]" % ext_if_name
            )
            self.connect_cmds.append("assign_bd_address")
            seg_name = "%s/Data_m_axi_gmem/SEG_%s_Reg" % (node.name, ext_if_name)
            self.connect_cmds.append("set_property offset 0 [get_bd_addr_segs {%s}]" % (seg_name))
            # TODO should propagate this information from the node instead of 4G
            self.connect_cmds.append("set_property range 4G [get_bd_addr_segs {%s}]" % (seg_name))
            self.intf_names["aximm"] = [(ext_if_name, aximm_intf_name[0][1])]
            self.has_aximm = True

    def connect_m_axis_external(self, node, kernel, idx=None):
        output_intf_names = kernel.get_verilog_top_module_intf_names()["m_axis"]
        # make output axis external
        for i in range(len(output_intf_names)):
            if idx is not None and idx != i:
                continue
            output_intf_name = output_intf_names[i][0]
            self.connect_cmds.append(
                "make_bd_intf_pins_external [get_bd_intf_pins %s/%s]"
                % (node.name, output_intf_name)
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

    def connect_s_axis_external(self, node, kernel, idx=None):
        input_intf_names = kernel.get_verilog_top_module_intf_names()["s_axis"]
        # make input axis external
        for i in range(len(input_intf_names)):
            if idx is not None and idx != i:
                continue
            input_intf_name = input_intf_names[i][0]
            self.connect_cmds.append(
                "make_bd_intf_pins_external [get_bd_intf_pins %s/%s]" % (node.name, input_intf_name)
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

    def connect_ap_none_external(self, node, kernel):
        input_intf_names = kernel.get_verilog_top_module_intf_names()["ap_none"]
        # make external
        for i in range(len(input_intf_names)):
            input_intf_name = input_intf_names[i]
            self.connect_cmds.append(
                "make_bd_pins_external [get_bd_pins %s/%s]" % (node.name, input_intf_name)
            )
            self.connect_cmds.append(
                "set_property name %s [get_bd_ports %s_0]" % (input_intf_name, input_intf_name)
            )

    def insert_signature(self, checksum_count, signature, clk_ns):
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
                signature[0],
                signature[1],
                signature[2],
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
        fclk_mhz = 1 / (clk_ns * 0.001)
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

    def apply(self, model: ModelWrapper):
        ip_dirs = ["list"]
        # add RTL streamer IP
        # ip_dirs.append("$::env(FINN_ROOT)/finn-rtllib/memstream")
        # if self.ctx.signature:
        #     ip_dirs.append("$::env(FINN_ROOT)/finn-rtllib/axi_info")
        global_inp_names = [inp.name for inp in model.graph.input]

        # Make cmds for kernel files.
        for kernel_file, _ in self.ctx.kernel_files.items():
            full_path = self.ctx.directory / kernel_file
            file_paths = {file for file in full_path.rglob('*') if file.is_file()}
            self.create_cmds += [f"add_files -norecurse ../{path.relative_to(self.ctx.directory)}" for path in file_paths]

        # Make cmds for shared files.
        full_path = self.ctx.shared_dir
        file_paths = {file for file in full_path.rglob('*') if file.is_file()}
        self.create_cmds += [f"add_files -norecurse ../{path.relative_to(self.ctx.directory)}" for path in file_paths]

        for node in model.graph.node:
            # Extract node attributes and fetch kernel
            kernel: Kernel = gkr.kernel(node.op_type, get_node_attr(node, model))
            node_ctx = self.ctx.get_subcontext(Path(node.name))
            ip_dir_value = node_ctx.directory
            assert os.path.isdir(ip_dir_value), "IP generation directory doesn't exist."
            ip_dirs += ['../'+str(ip_dir_value.relative_to(self.ctx.directory))]
            kernel_cmds = kernel.code_generation_ipi(node_ctx)
            kernel_cmds = [cmd.replace("$CODEGEN_DIR_IP_GEN$", '../'+str(node_ctx.directory.relative_to(self.ctx.directory))) for cmd in kernel_cmds]
            self.create_cmds += kernel_cmds
            self.connect_clk_rst(node, kernel)
            self.connect_ap_none_external(node, kernel)
            self.connect_axi(node, kernel)
            for i in range(len(node.input)):
                if not is_external_input(model, node, i, kernel):
                    producer = model.find_producer(node.input[i])
                    if producer is None:
                        continue
                    j = list(producer.output).index(node.input[i])
                    producer_kernel = gkr.kernel(producer.op_type, get_node_attr(producer, model))
                    src_intf_name = producer_kernel.get_verilog_top_module_intf_names()[
                        "m_axis"
                    ][j][0]
                    dst_intf_name = kernel.get_verilog_top_module_intf_names()["s_axis"][i][0]
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
            kernel: Kernel = gkr.kernel(node.op_type, get_node_attr(node, model))
            for i in range(len(node.input)):
                if node.input[i] == inp_name:
                    self.connect_s_axis_external(node, kernel, idx=i)
        for output in model.graph.output:
            out_name = output.name
            node = model.find_producer(out_name)
            assert node is not None, "No producer for output " + out_name
            kernel: Kernel = gkr.kernel(node.op_type, get_node_attr(node, model))
            for i in range(len(node.output)):
                if node.output[i] == out_name:
                    self.connect_m_axis_external(node, kernel, idx=i)

        if self.ctx.signature:
            # extract number of checksum layer from graph
            checksum_layers = model.get_nodes_by_op_type("CheckSum_hls")
            self.insert_signature(len(checksum_layers), self.ctx.signature, self.ctx.clk_ns)

        # create a folder for the project
        prjname = "finn_vivado_stitch_proj"
        vivado_stitch_proj_dir = self.ctx.directory / Path("vivado_stitch_proj")
        vivado_stitch_proj_dir.mkdir(exist_ok=True)
        model.set_metadata_prop("vivado_stitch_proj", str(vivado_stitch_proj_dir))

        # start building the tcl script
        tcl = []
        # create vivado project
        tcl.append(
            "create_project %s %s -part %s -force" % (prjname, '.', self.ctx.fpga_part)
        )
        # no warnings on long module names
        tcl.append("set_msg_config -id {[BD 41-1753]} -suppress")
        # add all the generated IP dirs to ip_repo_paths
        ip_dirs_str = " ".join(ip_dirs)
        tcl.append("set_property ip_repo_paths [%s] [current_project]" % ip_dirs_str)
        tcl.append("update_ip_catalog")
        # create block design and instantiate all layers
        block_name = self.ctx.ip_name
        tcl.append('create_bd_design "%s"' % block_name)
        tcl.extend(self.create_cmds)
        tcl.extend(self.connect_cmds)
        fclk_mhz = 1 / (self.ctx.clk_ns * 0.001)
        fclk_hz = fclk_mhz * 1000000
        tcl.append("set_property CONFIG.FREQ_HZ %d [get_bd_ports /ap_clk]" % round(fclk_hz))
        if self.clock2x_is_external:
            tcl.append(
                "set_property CONFIG.FREQ_HZ %d [get_bd_ports /ap_clk2x]" % round(2 * fclk_hz)
            )
        tcl.append("validate_bd_design")
        tcl.append("save_bd_design")
        # create wrapper hdl (for rtlsim later on)
        bd_base = "%s.srcs/sources_1/bd/%s" % (
            prjname,
            block_name,
        )
        bd_base_abs = "%s/%s.srcs/sources_1/bd/%s" % (
            vivado_stitch_proj_dir,
            prjname,
            block_name,
        )
        bd_filename = "%s/%s.bd" % (bd_base, block_name)
        tcl.append("make_wrapper -files [get_files %s] -top" % bd_filename)
        wrapper_filename = "%s/hdl/%s_wrapper.v" % (bd_base, block_name)
        wrapper_filename_abs = "%s/hdl/%s_wrapper.v" % (bd_base_abs, block_name)
        tcl.append("add_files -norecurse %s" % wrapper_filename)
        model.set_metadata_prop("wrapper_filename", wrapper_filename)
        tcl.append("set_property top %s_wrapper [current_fileset]" % block_name)
        # synthesize to DCP and export stub, DCP and constraints
        if self.ctx.vitis:
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
                "ipx::package_project -root_dir ip -vendor %s "
                "-library %s -taxonomy /UserIP -module %s -import_files"
            )
            % (block_vendor, block_library, block_name)
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
        if self.ctx.vitis:
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
            tcl.append("file delete -force ip/sim")
            tcl.append("file delete -force ip/src")
            # copy and add DCP, stub, and xdc
            tcl.append("file mkdir ip/dcp")
            tcl.append("file mkdir ip/impl")
            tcl.append("file copy -force %s.dcp ip/dcp" % (block_name))
            tcl.append("file copy -force %s.xdc ip/impl" % (block_name))
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
        example_data_dir = importlib.resources.files("finn") / Path("qnn-data/mdd-data")
        copytree(example_data_dir, vivado_stitch_proj_dir / Path("data"), dirs_exist_ok=True)

        #####
        # Core Cleanup Operations
        tcl.append(templates.ipstitching_template)

        # export list of used Verilog files (for rtlsim later on)
        tcl.append(
            "set all_v_files [get_files -filter {USED_IN_SYNTHESIS == 1 "
            + "&& (FILE_TYPE == Verilog || FILE_TYPE == SystemVerilog "
            + '|| FILE_TYPE =="Verilog Header")}]'
        )
        v_file_list = "all_verilog_srcs.txt"
        tcl.append("set fp [open %s w]" % v_file_list)
        # write each verilog filename to all_verilog_srcs.txt
        tcl.append("foreach vf $all_v_files {puts $fp $vf}")
        tcl.append("close $fp")
        # write the project creator tcl script
        tcl_string = "\n".join(tcl) + "\n"
        with open(vivado_stitch_proj_dir / Path("make_project.tcl"), "w") as f:
            f.write(tcl_string)
        # create a shell script and call Vivado
        make_project_sh = vivado_stitch_proj_dir / Path("make_project.sh")
        with open(make_project_sh, "w") as f:
            f.write("#!/bin/bash \n")
            f.write("pushd {}\n".format(str(vivado_stitch_proj_dir)))
            f.write("vivado -mode batch -source make_project.tcl\n")
            f.write("popd\n")
        bash_command = ["bash", make_project_sh]
        process_compile = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
        process_compile.communicate()
        # wrapper may be created in different location depending on Vivado version
        if not os.path.isfile(wrapper_filename):
            # check in alternative location (.gen instead of .srcs)
            wrapper_filename_alt = wrapper_filename_abs.replace(".srcs", ".gen")
            if os.path.isfile(wrapper_filename_alt):
                model.set_metadata_prop("wrapper_filename", wrapper_filename_alt)
            else:
                raise Exception(
                    """CreateStitchedIP failed, no wrapper HDL found under %s or %s.
                    Please check logs under the parent directory."""
                    % (wrapper_filename, wrapper_filename_alt)
                )

        return (model, False)
