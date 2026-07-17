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

# flake8: noqa

call_pynqshell_makefile_template = """
#!/bin/bash
cd %s
export platform=%s
export ip_config=%s
make %s
cd %s
"""

custom_zynq_shell_template = """
set FREQ_MHZ @FREQ_MHZ@
set NUM_AXILITE @NUM_AXILITE@
if {$NUM_AXILITE > 9} {
    error "Maximum 10 AXI-Lite interfaces supported"
}
set NUM_AXIMM @NUM_AXIMM@
set BOARD @BOARD@
set FPGA_PART @FPGA_PART@
create_project finn_link ./finn_link -part $FPGA_PART

# set board part repo paths to find PYNQ-Z1/Z2
set paths_prop [get_property BOARD_PART_REPO_PATHS [current_project]]
set paths_param [get_param board.repoPaths]
lappend paths_prop $::env(FINN_ROOT)/deps/board_files
lappend paths_param $::env(FINN_ROOT)/deps/board_files
set_property BOARD_PART_REPO_PATHS $paths_prop [current_project]
set_param board.repoPaths $paths_param

if {$BOARD == "ZCU104"} {
    set_property board_part xilinx.com:zcu104:part0:1.1 [current_project]
    set ZYNQ_TYPE "zynq_us+"
} elseif {$BOARD == "ZCU102"} {
    set_property board_part xilinx.com:zcu102:part0:3.3 [current_project]
    set ZYNQ_TYPE "zynq_us+"
} elseif {$BOARD == "RFSoC2x2"} {
    set_property board_part xilinx.com:rfsoc2x2:part0:1.1 [current_project]
    set ZYNQ_TYPE "zynq_us+"
} elseif {$BOARD == "RFSoC4x2"} {
    set_property board_part realdigital.org:rfsoc4x2:part0:1.0 [current_project]
    set ZYNQ_TYPE "zynq_us+"
} elseif {$BOARD == "Ultra96"} {
    set_property board_part avnet.com:ultra96v1:part0:1.2 [current_project]
    set ZYNQ_TYPE "zynq_us+"
} elseif {$BOARD == "Ultra96-V2"} {
    set_property board_part avnet.com:ultra96v2:part0:1.2 [current_project]
    set ZYNQ_TYPE "zynq_us+"
} elseif {$BOARD == "Pynq-Z2"} {
    set ZYNQ_TYPE "zynq_7000"
    set_property board_part tul.com.tw:pynq-z2:part0:1.0 [current_project]
} elseif {$BOARD == "Pynq-Z1"} {
    set ZYNQ_TYPE "zynq_7000"
    set_property board_part www.digilentinc.com:pynq-z1:part0:1.0 [current_project]
} elseif {$BOARD == "KV260_SOM"} {
    set ZYNQ_TYPE "zynq_us+"
    set_property board_part xilinx.com:kv260_som:part0:1.3 [current_project]
} elseif {$BOARD == "AUP-ZU3_8GB"} {
    set ZYNQ_TYPE "zynq_us+"
    set_property board_part realdigital.org:aup-zu3-8gb:part0:1.0 [current_project]
} else {
    puts "Unrecognized board"
}

create_bd_design "finn_link"
if {$ZYNQ_TYPE == "zynq_us+"} {
    set zynq_ps_vlnv [get_property VLNV [get_ipdefs "xilinx.com:ip:zynq_ultra_ps_e:*"]]
    create_bd_cell -type ip -vlnv $zynq_ps_vlnv zynq_ps
    apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1" }  [get_bd_cells zynq_ps]
    #activate one slave port, deactivate the second master port
    set_property -dict [list CONFIG.PSU__USE__S_AXI_GP2 {1}] [get_bd_cells zynq_ps]
    set_property -dict [list CONFIG.PSU__USE__M_AXI_GP1 {0}] [get_bd_cells zynq_ps]
    #activate one master port and deactivate third master port for AUP-ZU3
    if {$BOARD == "AUP-ZU3_8GB"} {
        set_property -dict [list CONFIG.PSU__USE__M_AXI_GP0 {1}] [get_bd_cells zynq_ps]
        set_property -dict [list CONFIG.PSU__USE__M_AXI_GP2 {0}] [get_bd_cells zynq_ps]
    }
    #set frequency of PS clock (this can't always be exactly met)
    set_property -dict [list CONFIG.PSU__OVERRIDE__BASIC_CLOCK {0}] [get_bd_cells zynq_ps]
    set_property -dict [list CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ [expr int($FREQ_MHZ)]] [get_bd_cells zynq_ps]
} elseif {$ZYNQ_TYPE == "zynq_7000"} {
    set zynq_ps_vlnv [get_property VLNV [get_ipdefs "xilinx.com:ip:processing_system7:*"]]
    create_bd_cell -type ip -vlnv $zynq_ps_vlnv zynq_ps
    apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable" }  [get_bd_cells zynq_ps]
    set_property -dict [list CONFIG.PCW_USE_S_AXI_HP0 {1}] [get_bd_cells zynq_ps]
    set_property -dict [list CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ [expr int($FREQ_MHZ)]] [get_bd_cells zynq_ps]
} else {
    puts "Unrecognized Zynq type"
}

#instantiate axi interconnect, axi smartconnect
set interconnect_vlnv [get_property VLNV [get_ipdefs -all "xilinx.com:ip:axi_interconnect:*" -filter design_tool_contexts=~*IPI*]]
set smartconnect_vlnv [get_property VLNV [get_ipdefs "xilinx.com:ip:smartconnect:*"]]
create_bd_cell -type ip -vlnv $interconnect_vlnv axi_interconnect_0
create_bd_cell -type ip -vlnv $smartconnect_vlnv smartconnect_0
#set number of axilite interfaces, and number of axi master interfaces
set_property -dict [list CONFIG.NUM_SI $NUM_AXIMM] [get_bd_cells smartconnect_0]
set_property -dict [list CONFIG.NUM_MI $NUM_AXILITE] [get_bd_cells axi_interconnect_0]

#create reset controller and connect interconnects to PS
if {$ZYNQ_TYPE == "zynq_us+"} {
    set axi_peripheral_base 0xA0000000
    connect_bd_intf_net [get_bd_intf_pins smartconnect_0/M00_AXI] [get_bd_intf_pins zynq_ps/S_AXI_HP0_FPD]
    connect_bd_intf_net [get_bd_intf_pins zynq_ps/M_AXI_HPM0_FPD] -boundary_type upper [get_bd_intf_pins axi_interconnect_0/S00_AXI]
    #connect interconnect clocks and resets
    apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ps/pl_clk0} Freq {} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins axi_interconnect_0/ACLK]
    apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ps/pl_clk0} Freq {} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins axi_interconnect_0/S00_ACLK]
    apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ps/pl_clk0} Freq {} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins zynq_ps/saxihp0_fpd_aclk]
} elseif {$ZYNQ_TYPE == "zynq_7000"} {
    set axi_peripheral_base 0x40000000
    connect_bd_intf_net -boundary_type upper [get_bd_intf_pins zynq_ps/M_AXI_GP0] [get_bd_intf_pins axi_interconnect_0/S00_AXI]
    connect_bd_intf_net [get_bd_intf_pins smartconnect_0/M00_AXI] [get_bd_intf_pins zynq_ps/S_AXI_HP0]
    apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ps/FCLK_CLK0} Freq {} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins axi_interconnect_0/ACLK]
    apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ps/FCLK_CLK0} Freq {} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins axi_interconnect_0/S00_ACLK]
    apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ps/FCLK_CLK0} Freq {} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins zynq_ps/S_AXI_HP0_ACLK]
}
connect_bd_net [get_bd_pins axi_interconnect_0/ARESETN] [get_bd_pins smartconnect_0/aresetn]

#procedure used by below IP instantiations to map BD address segments based on the axi interface aperture
proc assign_axi_addr_proc {axi_intf_path} {
    #global variable holds current base address
    global axi_peripheral_base
    #infer range
    set range [expr 2**[get_property CONFIG.ADDR_WIDTH [get_bd_intf_pins $axi_intf_path]]]
    set range [expr $range < 4096 ? 4096 : $range]
    #align base address to range
    set offset [expr ($axi_peripheral_base + ($range-1)) & ~($range-1)]
    #perform assignment
    assign_bd_address [get_bd_addr_segs $axi_intf_path/Reg*] -offset $offset -range $range
    #advance base address
    set axi_peripheral_base [expr $offset + $range]
}

#custom IP instantiations/connections start here
@CONFIG@

#MLO (Multi-Layer Offload) weight streaming
if {$ZYNQ_TYPE == "zynq_us+"} {
    set mlo_mm_pins [get_bd_intf_pins -quiet -of_objects [get_bd_cells] \
        -filter {MODE == Master && (NAME == m_axi_intermediate_frame || NAME =~ m_axi_MVAU_*)}]
    if {[llength $mlo_mm_pins] > 0} {
        set_property -dict [list CONFIG.PSU__USE__S_AXI_GP3 {1}] [get_bd_cells zynq_ps]
        create_bd_cell -type ip -vlnv $smartconnect_vlnv smartconnect_mlo
        set_property -dict [list CONFIG.NUM_SI [llength $mlo_mm_pins]] [get_bd_cells smartconnect_mlo]
        connect_bd_intf_net [get_bd_intf_pins smartconnect_mlo/M00_AXI] [get_bd_intf_pins zynq_ps/S_AXI_HP1_FPD]
        set mlo_si_idx 0
        foreach mlo_mm_pin $mlo_mm_pins {
            set mlo_si_name [format "S%02d_AXI" $mlo_si_idx]
            connect_bd_intf_net $mlo_mm_pin [get_bd_intf_pins smartconnect_mlo/$mlo_si_name]
            incr mlo_si_idx
        }
        connect_bd_net [get_bd_pins smartconnect_mlo/aclk] [get_bd_pins zynq_ps/pl_clk0]
        connect_bd_net [get_bd_pins zynq_ps/saxihp1_fpd_aclk] [get_bd_pins zynq_ps/pl_clk0]
        connect_bd_net [get_bd_pins smartconnect_mlo/aresetn] [get_bd_pins axi_interconnect_0/ARESETN]
    }
}

# set up debug
if {@ENABLE_DEBUG@ == 1} {
    set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {idma0_m_axis_0}]
    set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {StreamingDataflowPartition_1_m_axis_0}]
    set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {smartconnect_0_M00_AXI}]
    apply_bd_automation -rule xilinx.com:bd_rule:debug -dict [list \
                                                              [get_bd_intf_nets smartconnect_0_M00_AXI] {AXI_R_ADDRESS "Data and Trigger" AXI_R_DATA "Data and Trigger" AXI_W_ADDRESS "Data and Trigger" AXI_W_DATA "Data and Trigger" AXI_W_RESPONSE "Data and Trigger" CLK_SRC "/zynq_ps/FCLK_CLK0" SYSTEM_ILA "Auto" APC_EN "0" } \
                                                              [get_bd_intf_nets idma0_m_axis_0] {AXIS_SIGNALS "Data and Trigger" CLK_SRC "/zynq_ps/FCLK_CLK0" SYSTEM_ILA "Auto" APC_EN "0" } \
                                                              [get_bd_intf_nets StreamingDataflowPartition_1_m_axis_0] {AXIS_SIGNALS "Data and Trigger" CLK_SRC "/zynq_ps/FCLK_CLK0" SYSTEM_ILA "Auto" APC_EN "0" } \
                                                             ]
}

#finalize clock and reset connections for interconnects
if {$ZYNQ_TYPE == "zynq_us+"} {
    apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ps/pl_clk0} }  [get_bd_pins axi_interconnect_0/M*_ACLK]
} elseif {$ZYNQ_TYPE == "zynq_7000"} {
    apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ps/FCLK_CLK0} }  [get_bd_pins axi_interconnect_0/M*_ACLK]
}

save_bd_design
assign_bd_address
validate_bd_design

set_property SYNTH_CHECKPOINT_MODE "Hierarchical" [ get_files finn_link.bd ]
make_wrapper -files [get_files finn_link.bd] -import -fileset sources_1 -top

set_property strategy Flow_PerfOptimized_high [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.DIRECTIVE AlternateRoutability [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING true [get_runs synth_1]
set_property strategy Performance_ExtraTimingOpt [get_runs impl_1]
set_property STEPS.OPT_DESIGN.ARGS.DIRECTIVE Explore [get_runs impl_1]
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]

# out-of-context synth can't be used for bitstream generation
# set_property -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} -value {-mode out_of_context} -objects [get_runs synth_1]
launch_runs -to_step write_bitstream impl_1 -jobs @NUM_WORKERS@
wait_on_run [get_runs impl_1]

# generate synthesis report
open_run impl_1
report_utilization -hierarchical -hierarchical_depth 4 -file synth_report.xml -format xml
report_timing_summary -file timing_summary_routed.rpt
close_project
"""

# Versal (embedded, e.g. VCK190) overlay shell template.
custom_versal_shell_template = """
set FREQ_MHZ @FREQ_MHZ@
set NUM_AXILITE @NUM_AXILITE@
if {$NUM_AXILITE > 16} {
    error "Maximum 16 AXI-Lite interfaces supported"
}
set NUM_AXIMM @NUM_AXIMM@
set BOARD @BOARD@
set FPGA_PART @FPGA_PART@
set GOLDEN_DIR @GOLDEN_DIR@
set OVERLAY_NAME finn_link
set design_name $OVERLAY_NAME

# Source the golden reference design
source [file join $GOLDEN_DIR golden_ref.tcl]

# Remove the golden tie-offs on the interfaces FINN drives with real logic
delete_bd_objs [get_bd_cells pl_tieoff_fpd]
delete_bd_objs [get_bd_cells pl_tieoff_dma0]
delete_bd_objs [get_bd_cells pl_tieoff_dma1]

# Control path: M_AXI_FPD -> control SmartConnect -> kernel AXI-Lite ports
set smartconnect_vlnv [get_property VLNV [get_ipdefs "xilinx.com:ip:smartconnect:*"]]
create_bd_cell -type ip -vlnv $smartconnect_vlnv axi_interconnect_0
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI $NUM_AXILITE] [get_bd_cells axi_interconnect_0]
connect_bd_intf_net [get_bd_intf_pins versal_cips_0/M_AXI_FPD] [get_bd_intf_pins axi_interconnect_0/S00_AXI]

# DDR path: FINN I/O DMA masters -> SmartConnect -> axi_noc_pl/S00_AXI
create_bd_cell -type ip -vlnv $smartconnect_vlnv smartconnect_0
set_property -dict [list CONFIG.NUM_SI $NUM_AXIMM CONFIG.NUM_MI {1}] [get_bd_cells smartconnect_0]
connect_bd_intf_net [get_bd_intf_pins smartconnect_0/M00_AXI] [get_bd_intf_pins axi_noc_pl/S00_AXI]

# Procedure to assign AXI-Lite register apertures in the M_AXI_FPD space.
# PL peripherals live in the 0xA4000000 window in the golden address map.
set axi_peripheral_base 0xA4000000
proc assign_axi_addr_proc {axi_intf_path} {
    global axi_peripheral_base
    set range [expr 2**[get_property CONFIG.ADDR_WIDTH [get_bd_intf_pins $axi_intf_path]]]
    set range [expr $range < 4096 ? 4096 : $range]
    set offset [expr ($axi_peripheral_base + ($range-1)) & ~($range-1)]
    assign_bd_address [get_bd_addr_segs $axi_intf_path/Reg*] \
        -target_address_space [get_bd_addr_spaces versal_cips_0/M_AXI_FPD] \
        -offset $offset -range $range -force
    set axi_peripheral_base [expr $offset + $range]
}

# Procedure to map an aximm master onto DDR through the PS NoC inter-NoC port.
# Maps both DDR_LOW0 (0-2 GB) and DDR_LOW1 (32 GB+) so the DMA can reach any
# buffer the runtime CMA allocator hands out.
# TODO. look into auto assignment
proc assign_ddr_addr_proc {aximm_intf_path} {
    set space [get_bd_addr_spaces -of_objects [get_bd_intf_pins $aximm_intf_path]]
    assign_bd_address -offset 0x00000000 -range 0x80000000 \
        -target_address_space $space \
        [get_bd_addr_segs axi_noc_ps/S00_INI/C0_DDR_LOW0] -force
    assign_bd_address -offset 0x000800000000 -range 0x180000000 \
        -target_address_space $space \
        [get_bd_addr_segs axi_noc_ps/S00_INI/C0_DDR_LOW1] -force
}

# custom IP instantiations/connections start here
@CONFIG@

foreach gmem_pin [get_bd_intf_pins -quiet -of_objects [get_bd_cells] \
    -filter {MODE == Master && NAME == m_axi_gmem0}] {
    assign_ddr_addr_proc [get_property PATH $gmem_pin]
}

# MLO (Multi-Layer Offload) weight streaming -> axi_noc_pl/S01_AXI
set mlo_mm_pins [get_bd_intf_pins -quiet -of_objects [get_bd_cells] \
    -filter {MODE == Master && (NAME == m_axi_intermediate_frame || NAME =~ m_axi_MVAU_*)}]
if {[llength $mlo_mm_pins] > 0} {
    create_bd_cell -type ip -vlnv $smartconnect_vlnv smartconnect_mlo
    set_property -dict [list CONFIG.NUM_SI [llength $mlo_mm_pins] CONFIG.NUM_MI {1}] [get_bd_cells smartconnect_mlo]
    connect_bd_intf_net [get_bd_intf_pins smartconnect_mlo/M00_AXI] [get_bd_intf_pins axi_noc_pl/S01_AXI]
    set mlo_si_idx 0
    foreach mlo_mm_pin $mlo_mm_pins {
        set mlo_si_name [format "S%02d_AXI" $mlo_si_idx]
        connect_bd_intf_net $mlo_mm_pin [get_bd_intf_pins smartconnect_mlo/$mlo_si_name]
        assign_ddr_addr_proc [get_property PATH $mlo_mm_pin]
        incr mlo_si_idx
    }
    connect_bd_net [get_bd_pins smartconnect_mlo/aclk] [get_bd_pins versal_cips_0/pl0_ref_clk]
    connect_bd_net [get_bd_pins smartconnect_mlo/aresetn] [get_bd_pins rst_pl0/peripheral_aresetn]
} else {
    # keep the second NoC PL slave port driven so the locked NoC solution
    # remains valid (matches the golden 2-SI topology)
    create_bd_cell -type ip -vlnv xilinx.com:ip:axi_vip:1.1 pl_dma1_tieoff
    set_property -dict [list CONFIG.INTERFACE_MODE {MASTER} CONFIG.PROTOCOL {AXI4} CONFIG.ADDR_WIDTH {64} CONFIG.DATA_WIDTH {128}] [get_bd_cells pl_dma1_tieoff]
    connect_bd_intf_net [get_bd_intf_pins pl_dma1_tieoff/M_AXI] [get_bd_intf_pins axi_noc_pl/S01_AXI]
    connect_bd_net [get_bd_pins pl_dma1_tieoff/aclk] [get_bd_pins versal_cips_0/pl0_ref_clk]
    connect_bd_net [get_bd_pins pl_dma1_tieoff/aresetn] [get_bd_pins rst_pl0/peripheral_aresetn]
    assign_ddr_addr_proc pl_dma1_tieoff/M_AXI
}

# clock/reset for the control + DDR SmartConnects
connect_bd_net [get_bd_pins versal_cips_0/pl0_ref_clk] \
    [get_bd_pins axi_interconnect_0/aclk] \
    [get_bd_pins smartconnect_0/aclk]
connect_bd_net [get_bd_pins rst_pl0/peripheral_aresetn] \
    [get_bd_pins axi_interconnect_0/aresetn] \
    [get_bd_pins smartconnect_0/aresetn]

# set up debug
if {@ENABLE_DEBUG@ == 1} {
    set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets -quiet {idma0_m_axis_0}]
}

validate_bd_design
save_bd_design

# Build flow: wrapper, segmented configuration, lock golden NoC, implement,
# verify against the golden routed checkpoint, export the PL PDI.
make_wrapper -files [get_files $OVERLAY_NAME.bd] -import -fileset sources_1 -top
set_property top ${OVERLAY_NAME}_wrapper [current_fileset]
update_compile_order -fileset sources_1

set_property platform.default_output_type "sd_card" [current_project]
set_property platform.design_intent.embedded "true" [current_project]
set_property platform.design_intent.server_managed "false" [current_project]
set_property platform.design_intent.external_host "false" [current_project]
set_property platform.design_intent.datacenter "false" [current_project]
set_property segmented_configuration true [current_project]

# lock the NoC solution to the golden reference -- mandatory for the PLD PDI
# to be compatible with the golden boot PDI
set golden_ncr [file join $GOLDEN_DIR golden_noc.ncr]
if {[file exists $golden_ncr]} {
    set_property NOC_SOLUTION_FILE [file normalize $golden_ncr] [get_runs impl_1]
} else {
    error "golden_noc.ncr not found in $GOLDEN_DIR"
}

set_property strategy Flow_PerfOptimized_high [get_runs synth_1]
set_property strategy Performance_ExtraTimingOpt [get_runs impl_1]

launch_runs impl_1 -to_step write_device_image -jobs @NUM_WORKERS@
wait_on_run [get_runs impl_1]

set impl_status [get_property STATUS [get_runs impl_1]]
if { [string match "*Complete*" $impl_status] == 0 } {
    error "Implementation did not complete (status: $impl_status)"
}

# verify NoC/static compatibility with the golden routed checkpoint
set golden_dcp [file join $GOLDEN_DIR golden_routed.dcp]
set overlay_dcps [glob -nocomplain ./${OVERLAY_NAME}/${OVERLAY_NAME}.runs/impl_1/*_routed.dcp]
if {[file exists $golden_dcp] && [llength $overlay_dcps] > 0} {
    if {[catch {pr_verify [file normalize $golden_dcp] [lindex $overlay_dcps 0]} msg]} {
        error "pr_verify FAILED -- overlay incompatible with golden reference: $msg"
    }
    puts "pr_verify PASSED -- overlay compatible with golden reference"
} else {
    error "golden_routed.dcp or overlay routed checkpoint missing, cannot pr_verify"
}

# synthesis utilization report
open_run impl_1
report_utilization -hierarchical -hierarchical_depth 4 -file synth_report.xml -format xml
report_timing_summary -file timing_summary_routed.rpt
close_project
"""

vitis_gen_xml_report_tcl_template = """
open_project $VITIS_PROJ_PATH$/_x/link/vivado/vpl/prj/prj.xpr
open_run impl_1
report_utilization -hierarchical -hierarchical_depth 5 -file $VITIS_PROJ_PATH$/synth_report.xml -format xml
"""
