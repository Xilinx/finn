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

#PL clock: the PS clock. Double-pumped kernels derive their ap_clk2x from it
#internally (CreateStitchedIP clk2x_gen).
if {$ZYNQ_TYPE == "zynq_us+"} {
    set pl_clk /zynq_ps/pl_clk0
} else {
    set pl_clk /zynq_ps/FCLK_CLK0
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
    apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config [list Clk $pl_clk Freq {} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}]  [get_bd_pins axi_interconnect_0/ACLK]
    apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config [list Clk $pl_clk Freq {} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}]  [get_bd_pins axi_interconnect_0/S00_ACLK]
    apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config [list Clk $pl_clk Freq {} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}]  [get_bd_pins zynq_ps/saxihp0_fpd_aclk]
} elseif {$ZYNQ_TYPE == "zynq_7000"} {
    set axi_peripheral_base 0x40000000
    connect_bd_intf_net -boundary_type upper [get_bd_intf_pins zynq_ps/M_AXI_GP0] [get_bd_intf_pins axi_interconnect_0/S00_AXI]
    connect_bd_intf_net [get_bd_intf_pins smartconnect_0/M00_AXI] [get_bd_intf_pins zynq_ps/S_AXI_HP0]
    apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config [list Clk $pl_clk Freq {} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}]  [get_bd_pins axi_interconnect_0/ACLK]
    apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config [list Clk $pl_clk Freq {} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}]  [get_bd_pins axi_interconnect_0/S00_ACLK]
    apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config [list Clk $pl_clk Freq {} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}]  [get_bd_pins zynq_ps/S_AXI_HP0_ACLK]
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

#procedure that opens memory port idx of the PS for the MLO (Multi-Layer Offload)
#masters and returns it. Port 0 is S_AXI_HP0_FPD, which smartconnect_0 already uses
proc mlo_port {idx} {
    global ZYNQ_TYPE pl_clk
    if {$ZYNQ_TYPE != "zynq_us+" || $idx < 1 || $idx > 3} {
        error "No memory port $idx for MLO on $ZYNQ_TYPE"
    }
    #S_AXI_HP<idx>_FPD is S_AXI_GP<idx+2> in the PS configuration
    set_property CONFIG.PSU__USE__S_AXI_GP[expr {$idx + 2}] {1} [get_bd_cells zynq_ps]
    connect_bd_net [get_bd_pins zynq_ps/saxihp${idx}_fpd_aclk] [get_bd_pins $pl_clk]
    return [get_bd_intf_pins zynq_ps/S_AXI_HP${idx}_FPD]
}

#custom IP instantiations/connections start here
@CONFIG@

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
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config [list Clk $pl_clk]  [get_bd_pins axi_interconnect_0/M*_ACLK]

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

# The golden reference script builds into an already open project and empty block
# design, so they keep the FINN names. The board part must match the golden one.
create_project $OVERLAY_NAME ./$OVERLAY_NAME -part $FPGA_PART
set_property board_part xilinx.com:vck190:part0:3.4 [current_project]
create_bd_design $OVERLAY_NAME

# Source the golden reference design.
source [file join $GOLDEN_DIR golden_ref.tcl]
if {[get_bd_cells -quiet versal_cips_0] eq ""} {
    error "Failed to create the golden reference design from $GOLDEN_DIR/golden_ref.tcl"
}

# The whole overlay runs on pl0_ref_clk at the kernel frequency.
set_property CONFIG.PS_PMC_CONFIG [list PMC_CRP_PL0_REF_CTRL_FREQMHZ $FREQ_MHZ] \
    [get_bd_cells versal_cips_0]

# The NoC is configured at boot from the golden design, so noc_pl must keep the
# golden data width once the tie-offs are gone.
foreach noc_port [get_bd_intf_pins noc_pl/S0*_AXI] {
    set_property CONFIG.DATA_WIDTH [get_property CONFIG.DATA_WIDTH $noc_port] $noc_port
}

# Replace the golden tie-offs on the interfaces FINN drives with real logic.
delete_bd_objs [get_bd_cells tieoff_fpd]
delete_bd_objs [get_bd_cells tieoff_pl0]

# Control path: M_AXI_FPD -> control SmartConnect -> kernel AXI-Lite ports
set smartconnect_vlnv [get_property VLNV [get_ipdefs "xilinx.com:ip:smartconnect:*"]]
create_bd_cell -type ip -vlnv $smartconnect_vlnv axi_interconnect_0
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI $NUM_AXILITE] [get_bd_cells axi_interconnect_0]
connect_bd_intf_net [get_bd_intf_pins versal_cips_0/M_AXI_FPD] [get_bd_intf_pins axi_interconnect_0/S00_AXI]

# DDR path: FINN I/O DMA masters -> SmartConnect -> noc_pl/S00_AXI
create_bd_cell -type ip -vlnv $smartconnect_vlnv smartconnect_0
set_property -dict [list CONFIG.NUM_SI $NUM_AXIMM CONFIG.NUM_MI {1}] [get_bd_cells smartconnect_0]
connect_bd_intf_net [get_bd_intf_pins smartconnect_0/M00_AXI] [get_bd_intf_pins noc_pl/S00_AXI]

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

# Procedure that opens memory port idx for the MLO (Multi-Layer Offload) masters and
# returns it. The ports are the four PL NMUs of the golden NoC (noc_pl/S00..S03_AXI),
# each of which reaches DDR4, LPDDR0 and LPDDR1. Port 0 is the one smartconnect_0
# already uses, FINN puts a SmartConnect smartconnect_mlo<idx> on every further one.
proc mlo_port {idx} {
    if {$idx < 1 || $idx > 3} {
        error "No memory port $idx for MLO"
    }
    delete_bd_objs [get_bd_cells tieoff_pl$idx]
    return [get_bd_intf_pins noc_pl/S0${idx}_AXI]
}

# Procedure that adds an AXI CDMA next to the I/O DMAs. It copies the staged MLO weights
# into the fixed-address memory banks (LPDDR). Its master takes slot aximm_idx of
# smartconnect_0, its control port slot axilite_idx of axi_interconnect_0.
proc add_mlo_cdma {name data_width clk rst aximm_idx axilite_idx} {
    create_bd_cell -type ip -vlnv [get_property VLNV [get_ipdefs xilinx.com:ip:axi_cdma:*]] $name
    set_property -dict [list CONFIG.C_INCLUDE_SG {0} CONFIG.C_ADDR_WIDTH {64} CONFIG.C_M_AXI_DATA_WIDTH $data_width CONFIG.C_M_AXI_MAX_BURST_LEN {64} CONFIG.C_INCLUDE_DRE {0}] [get_bd_cells $name]
    connect_bd_intf_net [get_bd_intf_pins $name/M_AXI] [get_bd_intf_pins [format "smartconnect_0/S%02d_AXI" $aximm_idx]]
    connect_bd_intf_net [get_bd_intf_pins $name/S_AXI_LITE] [get_bd_intf_pins [format "axi_interconnect_0/M%02d_AXI" $axilite_idx]]
    assign_axi_addr_proc $name/S_AXI_LITE
    connect_bd_net [get_bd_pins $clk] [get_bd_pins $name/s_axi_lite_aclk] [get_bd_pins $name/m_axi_aclk]
    connect_bd_net [get_bd_pins $rst] [get_bd_pins $name/s_axi_lite_aresetn]
}

# custom IP instantiations/connections start here
@CONFIG@

# clock/reset for the control + DDR SmartConnects
connect_bd_net [get_bd_pins versal_cips_0/pl0_ref_clk] \
    [get_bd_pins axi_interconnect_0/aclk] \
    [get_bd_pins smartconnect_0/aclk]
connect_bd_net [get_bd_pins rst_pl0/peripheral_aresetn] \
    [get_bd_pins axi_interconnect_0/aresetn] \
    [get_bd_pins smartconnect_0/aresetn]

# Map the memories that the golden NoC routes to NMU S0<nmu> into a PL master. The
# offsets are the ones of the golden reference (PS and PL see DDR4 at the same address,
# LPDDR0 and LPDDR1 sit at fixed windows). Which NoC ports lead to which memory depends on
# the NMU: S00/S02 use noc_lpddr0/S01_INI and noc_lpddr1_pl_only/S00_INI, S01/S03 use
# noc_lpddr0/S02_INI and noc_lpddr1_pl_only/S01_INI.
proc assign_mem_segs {master_pin nmu_idx} {
    set space [get_bd_addr_spaces -of_objects [get_bd_intf_pins $master_pin]]
    if {$nmu_idx % 2 == 0} {
        set lpddr0_seg noc_lpddr0/S01_INI/C1_DDR_CH2
        set lpddr1_seg noc_lpddr1_pl_only/S00_INI/C0_DDR_CH1
    } else {
        set lpddr0_seg noc_lpddr0/S02_INI/C2_DDR_CH2
        set lpddr1_seg noc_lpddr1_pl_only/S01_INI/C1_DDR_CH1
    }
    assign_bd_address -offset 0x00000000 -range 0x80000000 \
        -target_address_space $space [get_bd_addr_segs noc_ddr/S04_INI/C0_DDR_LOW0] -force
    assign_bd_address -offset 0x000800000000 -range 0x000180000000 \
        -target_address_space $space [get_bd_addr_segs noc_ddr/S04_INI/C0_DDR_LOW1] -force
    assign_bd_address -offset 0x060000000000 -range 0x000100000000 \
        -target_address_space $space [get_bd_addr_segs $lpddr0_seg] -force
    assign_bd_address -offset 0x050000000000 -range 0x000100000000 \
        -target_address_space $space [get_bd_addr_segs $lpddr1_seg] -force
}

# Apply it to every master that is connected to a SmartConnect feeding an NMU.
proc assign_sc_masters {sc_name nmu_idx} {
    foreach si [get_bd_intf_pins -quiet $sc_name/S*_AXI] {
        set net [get_bd_intf_nets -quiet -of_objects $si]
        foreach master_pin [get_bd_intf_pins -of_objects $net -filter {MODE == Master}] {
            assign_mem_segs $master_pin $nmu_idx
        }
    }
}
assign_sc_masters smartconnect_0 0
foreach nmu_idx {1 2 3} {
    if {[get_bd_cells -quiet smartconnect_mlo$nmu_idx] ne ""} {
        assign_sc_masters smartconnect_mlo$nmu_idx $nmu_idx
    }
}

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


# OPT_DESIGN pre-hook of VitisLink:
# - constrain the kernel clock 0 of an Alveo platform (output of the UCS MMCM) to
#   @F_MHZ@ MHz relative to the MMCM's reference clock, replacing the generated clock
#   that v++ derived. Leaves a correct constraint and platforms without this clock
#   alone.
# - put the 1x and 2x clocks of each kernel-internal clock generator (CreateStitchedIP
#   clk2x_gen) into a CLOCK_DELAY_GROUP of their own. The kernels carry this
#   constraint, but Vivado does not read the implementation XDC of kernels packaged as
#   synthesized checkpoints ("[Designutils 20-1275] Could not find cell 'inst'") and
#   uses it instead of the constraints in their checkpoints.
vitis_kernel_clk_tcl = """
proc finn_gcd {a b} { while {$b} { set t $b; set b [expr {$a % $b}]; set a $t }; return $a }
set kpin [get_pins -quiet -hierarchical -filter {NAME =~ */ulp_ucs/aclk_kernel_00}]
set kclk [get_clocks -quiet -of_objects $kpin]
set target_ns [expr {1000.0 / @F_MHZ@}]
if {[llength $kclk] != 1 || ![get_property IS_GENERATED $kclk]} {
    puts "INFO: \\[FINN\\] no generated Alveo kernel clock 0 found: constraints unchanged"
} elseif {abs([get_property PERIOD $kclk] - $target_ns) < 0.001} {
    puts "INFO: \\[FINN\\] kernel clock $kclk already constrained to $target_ns ns"
} else {
    set period [get_property PERIOD $kclk]
    set src [get_property SOURCE $kclk]
    set out [get_property SOURCE_PINS $kclk]
    set ref_ns [get_property PERIOD [get_clocks -of_objects [get_pins $src]]]
    set ref_khz [expr {round(1e6 / $ref_ns)}]
    set f_khz [expr {round(@F_MHZ@ * 1000)}]
    set g [finn_gcd $f_khz $ref_khz]
    set ratio [list -multiply_by [expr {$f_khz / $g}] -divide_by [expr {$ref_khz / $g}]]
    create_generated_clock -name [get_property NAME $kclk] {*}$ratio -source [get_pins $src] [get_pins $out]
    set kclk [get_clocks -of_objects $kpin]
    puts "INFO: \\[FINN\\] kernel clock $kclk re-constrained from $period ns to [get_property PERIOD $kclk] ns"
}
set n 0
foreach gen [get_cells -quiet -hierarchical -filter {NAME =~ */clk2x_domain && IS_PRIMITIVE == 0}] {
    set pins [get_pins -quiet -hierarchical -filter "NAME =~ $gen/*clk_buf/O || NAME =~ $gen/*clk2x_buf/O"]
    set nets [get_nets -quiet -of_objects $pins]
    if {[llength $nets] == 2} {
        set_property CLOCK_DELAY_GROUP finn_clk2x_[incr n] $nets
        puts "INFO: \\[FINN\\] clock generator $gen: CLOCK_DELAY_GROUP finn_clk2x_$n"
    }
}
"""
