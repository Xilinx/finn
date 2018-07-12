###############################################################################
 #  Copyright (c) 2016, Xilinx, Inc.
 #  All rights reserved.
 #
 #  Redistribution and use in source and binary forms, with or without
 #  modification, are permitted provided that the following conditions are met:
 #
 #  1.  Redistributions of source code must retain the above copyright notice,
 #     this list of conditions and the following disclaimer.
 #
 #  2.  Redistributions in binary form must reproduce the above copyright
 #      notice, this list of conditions and the following disclaimer in the
 #      documentation and/or other materials provided with the distribution.
 #
 #  3.  Neither the name of the copyright holder nor the names of its
 #      contributors may be used to endorse or promote products derived from
 #      this software without specific prior written permission.
 #
 #  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 #  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 #  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 #  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 #  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 #  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 #  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 #  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 #  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 #  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 #  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 #
###############################################################################
###############################################################################
 #
 #
 # @file make-vivado-proj.tcl
 #
 # tcl script for block design and bitstream generation. Automatically 
 # launched by make-hw.sh. Tested with Vivado 2016.1
 #
 #
###############################################################################

# Creates a Vivado project ready for synthesis and launches bitstream generation
if {$argc != 4} {
  puts "Expected: <jam repo> <proj name> <proj dir> <xdc_dir>"
  exit
}

# paths to donut and jam IP folders
set config_jam_repo [lindex $argv 0]

# project name, target dir and FPGA part to use
set config_proj_name [lindex $argv 1]
set config_proj_dir [lindex $argv 2]
set config_proj_part "xc7z020clg400-1"

# other project config

set xdc_dir [lindex $argv 3]

# set up project
create_project $config_proj_name $config_proj_dir -part $config_proj_part
set_property ip_repo_paths [list $config_jam_repo] [current_project]
update_ip_catalog

#Add PYNQ XDC
add_files -fileset constrs_1 -norecurse "${xdc_dir}/PYNQ-Z1_C.xdc"

# create block design
create_bd_design "procsys"
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7
set ps7 [get_bd_cells ps7]
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" Master "Disable" Slave "Disable" } $ps7
source "${xdc_dir}/pynq_revC.tcl"

set_property -dict [apply_preset $ps7] $ps7
set_property -dict [list CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} CONFIG.PCW_FPGA1_PERIPHERAL_FREQMHZ {142.86} CONFIG.PCW_FPGA2_PERIPHERAL_FREQMHZ {200} CONFIG.PCW_FPGA3_PERIPHERAL_FREQMHZ {166.67} CONFIG.PCW_EN_CLK1_PORT {1} CONFIG.PCW_EN_CLK2_PORT {1} CONFIG.PCW_EN_CLK3_PORT {1} CONFIG.PCW_USE_M_AXI_GP0 {1} CONFIG.PCW_USE_S_AXI_HP0 {1}] $ps7

# instantiate jam
create_bd_cell -type ip -vlnv xilinx.com:hls:BlackBoxJam:1.0 BlackBoxJam_0

# connect jam to ps7
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config {Master "/BlackBoxJam_0/m_axi_hostmem" Clk "Auto" }  [get_bd_intf_pins ps7/S_AXI_HP0]
delete_bd_objs [get_bd_nets ps7_FCLK_CLK0]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins rst_ps7_100M/slowest_sync_clk]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins axi_mem_intercon/ACLK]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins axi_mem_intercon/S00_ACLK]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins axi_mem_intercon/M00_ACLK]
connect_bd_net [get_bd_pins BlackBoxJam_0/ap_clk] [get_bd_pins ps7/FCLK_CLK0]
connect_bd_net [get_bd_pins ps7/S_AXI_HP0_ACLK] [get_bd_pins ps7/FCLK_CLK0]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config {Master "/ps7/M_AXI_GP0" Clk "/ps7/FCLK_CLK0 (100 MHz)" }  [get_bd_intf_pins BlackBoxJam_0/s_axi_control]


# create HDL wrapper for the block design
save_bd_design
make_wrapper -files [get_files $config_proj_dir/$config_proj_name.srcs/sources_1/bd/procsys/procsys.bd] -top
add_files -norecurse $config_proj_dir/$config_proj_name.srcs/sources_1/bd/procsys/hdl/procsys_wrapper.v
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

set_property strategy Flow_PerfOptimized_high [get_runs synth_1]

set_property STEPS.SYNTH_DESIGN.ARGS.DIRECTIVE AlternateRoutability [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING true [get_runs synth_1]

set_property strategy Performance_ExtraTimingOpt [get_runs impl_1]
set_property STEPS.OPT_DESIGN.ARGS.DIRECTIVE Explore [get_runs impl_1]
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]


# launch bitstream generation
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1

