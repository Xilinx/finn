# Copyright (c) 2023 Advanced Micro Devices, Inc.
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
# * Neither the name of AMD nor the names of its
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

set fpga_part @FPGA_PART@
set output_root ".."
# path to IP folder for instrumentation wrapper, change as needed
set instrwrp_ip_dir "$output_root/instrumentation_wrapper/project_instrwrap/sol1/impl/ip"
# path to IP folder for FINN IP, change as needed
set finn_ip_dir "$output_root/stitched_ip/ip"

create_project -force instr_sim_proj instr_sim_proj/ -part $fpga_part
create_bd_design "dut"
update_compile_order -fileset sources_1
set_property  ip_repo_paths  [list $instrwrp_ip_dir $finn_ip_dir] [current_project]
update_ip_catalog


create_bd_cell -type ip -vlnv xilinx_finn:finn:finn_design:1.0 finn_design_0
create_bd_cell -type ip -vlnv xilinx.com:hls:instrumentation_wrapper:1.0 instrumentation_wrap_0
connect_bd_intf_net [get_bd_intf_pins instrumentation_wrap_0/finnix] [get_bd_intf_pins finn_design_0/s_axis_0]
connect_bd_intf_net [get_bd_intf_pins finn_design_0/m_axis_0] [get_bd_intf_pins instrumentation_wrap_0/finnox]
make_bd_intf_pins_external  [get_bd_intf_pins instrumentation_wrap_0/s_axi_ctrl]
make_bd_pins_external  [get_bd_pins instrumentation_wrap_0/ap_clk]
make_bd_pins_external  [get_bd_pins instrumentation_wrap_0/ap_rst_n]
connect_bd_net [get_bd_ports ap_clk_0] [get_bd_pins finn_design_0/ap_clk]
connect_bd_net [get_bd_ports ap_rst_n_0] [get_bd_pins finn_design_0/ap_rst_n]

save_bd_design

update_compile_order -fileset sources_1
make_wrapper -files [get_files instr_sim_proj/instr_sim_proj.srcs/sources_1/bd/dut/dut.bd] -top
add_files -norecurse instr_sim_proj/instr_sim_proj.gen/sources_1/bd/dut/hdl/dut_wrapper.v

set_property SOURCE_SET sources_1 [get_filesets sim_1]
add_files -fileset sim_1 ./instrwrap_testbench.sv
update_compile_order -fileset sim_1

set_property synth_checkpoint_mode None [get_files instr_sim_proj/instr_sim_proj.srcs/sources_1/bd/dut/dut.bd]
generate_target Simulation [get_files instr_sim_proj/instr_sim_proj.srcs/sources_1/bd/dut/dut.bd]
launch_simulation -simset sim_1 -mode behavioral
run all
