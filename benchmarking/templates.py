# Template strings for benchmarking


# power report scripting based on Lucas Reuter:
template_open = """
open_project  $PROJ_PATH$
open_run $RUN$
"""

template_single_test = """
set_switching_activity -toggle_rate $TOGGLE_RATE$ -static_probability $STATIC_PROB$ -hier -type lut [get_cells -r finn_design_i/.*]
set_switching_activity -toggle_rate $TOGGLE_RATE$ -static_probability $STATIC_PROB$ -hier -type register [get_cells -r finn_design_i/.*]
set_switching_activity -deassert_resets
report_power -file $REPORT_PATH$/$REPORT_NAME$.xml -format xml
reset_switching_activity -hier -type lut [get_cells -r finn_design_i/.*]
reset_switching_activity -hier -type register [get_cells -r finn_design_i/.*]
"""

# template_single_test_type = """
# set_switching_activity -toggle_rate $TOGGLE_RATE$ -static_probability $STATIC_PROB$ -hier -type $SWITCH_TARGET$ [get_cells -r finn_design_i/.*]
# set_switching_activity -deassert_resets
# report_power -file $REPORT_PATH$/$REPORT_NAME$.xml -format xml
# reset_switching_activity -hier -type $SWITCH_TARGET$ [get_cells -r finn_design_i/.*]
# """

template_sim_power = """
set_property SOURCE_SET sources_1 [get_filesets sim_1]
import_files -fileset sim_1 -norecurse $TB_FILE_PATH$
set_property top switching_simulation_tb [get_filesets sim_1]
update_compile_order -fileset sim_1

launch_simulation -mode post-implementation -type functional
restart
open_saif $SAIF_FILE_PATH$
log_saif [get_objects -r /switching_simulation_tb/dut/*]
run $SIM_DURATION_NS$ ns
close_saif

read_saif $SAIF_FILE_PATH$
report_power -file $REPORT_PATH$/$REPORT_NAME$.xml -format xml
"""

# TODO: configurable clock frequency
template_switching_simulation_tb = """
`timescale 1 ns/10 ps

module switching_simulation_tb;
reg clk;
reg rst;

//dut inputs
reg tready;
reg [$INSTREAM_WIDTH$-1:0] tdata;
reg tvalid;

//dut outputs
wire [$OUTSTREAM_WIDTH$-1:0] accel_tdata;
wire accel_tready;
wire accel_tvalid;

finn_design_wrapper dut(
        .ap_clk(clk),
        .ap_rst_n(rst),
        .m_axis_0_tdata(accel_tdata),
        .m_axis_0_tready(tready),
        .m_axis_0_tvalid(accel_tvalid),
        .s_axis_0_tdata(tdata),
        .s_axis_0_tready(accel_tready),
        .s_axis_0_tvalid(tvalid)
        );

always
    begin
        clk = 0;
        #2.5;
        clk = 1;
        #2.5;
    end

integer i;
initial
    begin
        tready = 0;
        tdata = 0;
        tvalid = 0;
        rst = 0;
        #50;
        rst = 1;
        tvalid = 1;
        tready = 1;
        while(1)
            begin
                for (i = 0; i < $INSTREAM_WIDTH$/$DTYPE_WIDTH$; i = i+1) begin
                    tdata[i*$DTYPE_WIDTH$ +: $DTYPE_WIDTH$] = $RANDOM_FUNCTION$;
                end
                #5;
            end
    end
endmodule
"""

zynq_harness_template = """
set FREQ_MHZ %s
set NUM_AXILITE %d
if {$NUM_AXILITE > 9} {
    error "Maximum 10 AXI-Lite interfaces supported"
}
set NUM_AXIMM %d
set BOARD %s
set FPGA_PART %s
create_project finn_zynq_link ./ -part $FPGA_PART

# set board part repo paths to find boards installed by FINN
set paths_prop [get_property BOARD_PART_REPO_PATHS [current_project]]
set paths_param [get_param board.repoPaths]
lappend paths_prop $::env(FINN_ROOT)/deps/board_files
lappend paths_param $::env(FINN_ROOT)/deps/board_files
set_property BOARD_PART_REPO_PATHS $paths_prop [current_project]
set_param board.repoPaths $paths_param

if {$BOARD == "RFSoC2x2"} {
    set_property board_part xilinx.com:rfsoc2x2:part0:1.1 [current_project]
    set ZYNQ_TYPE "zynq_us+"
} else {
    puts "Unrecognized board"
}

create_bd_design "top"
if {$ZYNQ_TYPE == "zynq_us+"} {
    set zynq_ps_vlnv [get_property VLNV [get_ipdefs "xilinx.com:ip:zynq_ultra_ps_e:*"]]
    create_bd_cell -type ip -vlnv $zynq_ps_vlnv zynq_ps
    apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1" }  [get_bd_cells zynq_ps]
    set_property CONFIG.PSU__DISPLAYPORT__PERIPHERAL__ENABLE {0} [get_bd_cells zynq_ps]
    #activate one slave port, deactivate the second master port
    set_property -dict [list CONFIG.PSU__USE__S_AXI_GP2 {0}] [get_bd_cells zynq_ps]
    set_property -dict [list CONFIG.PSU__USE__M_AXI_GP1 {0}] [get_bd_cells zynq_ps]
    #set frequency of PS clock (this can't always be exactly met)
    set_property -dict [list CONFIG.PSU__OVERRIDE__BASIC_CLOCK {0}] [get_bd_cells zynq_ps]
    set_property -dict [list CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ [expr int($FREQ_MHZ)]] [get_bd_cells zynq_ps]
} else {
    puts "Unrecognized Zynq type"
}

#instantiate axi interconnect, axi smartconnect
set interconnect_vlnv [get_property VLNV [get_ipdefs -all "xilinx.com:ip:axi_interconnect:*" -filter design_tool_contexts=~*IPI*]]
#set smartconnect_vlnv [get_property VLNV [get_ipdefs "xilinx.com:ip:smartconnect:*"]]
create_bd_cell -type ip -vlnv $interconnect_vlnv axi_interconnect_0
#create_bd_cell -type ip -vlnv $smartconnect_vlnv smartconnect_0
#set number of axilite interfaces, and number of axi master interfaces
#set_property -dict [list CONFIG.NUM_SI $NUM_AXIMM] [get_bd_cells smartconnect_0]
set_property -dict [list CONFIG.NUM_MI $NUM_AXILITE] [get_bd_cells axi_interconnect_0]

#create reset controller and connect interconnects to PS
if {$ZYNQ_TYPE == "zynq_us+"} {
    set axi_peripheral_base 0xA0000000
    #connect_bd_intf_net [get_bd_intf_pins smartconnect_0/M00_AXI] [get_bd_intf_pins zynq_ps/S_AXI_HP0_FPD]
    connect_bd_intf_net [get_bd_intf_pins zynq_ps/M_AXI_HPM0_FPD] -boundary_type upper [get_bd_intf_pins axi_interconnect_0/S00_AXI]
    #connect interconnect clocks and resets
    apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ps/pl_clk0} Freq {} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins axi_interconnect_0/ACLK]
    apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ps/pl_clk0} Freq {} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins axi_interconnect_0/S00_ACLK]
    #apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ps/pl_clk0} Freq {} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins zynq_ps/saxihp0_fpd_aclk]
}
#connect_bd_net [get_bd_pins axi_interconnect_0/ARESETN] [get_bd_pins smartconnect_0/aresetn]

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
%s

#finalize clock and reset connections for interconnects
if {$ZYNQ_TYPE == "zynq_us+"} {
    apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ps/pl_clk0} }  [get_bd_pins axi_interconnect_0/M*_ACLK]
}

save_bd_design
assign_bd_address
validate_bd_design

set_property SYNTH_CHECKPOINT_MODE "Hierarchical" [ get_files top.bd ]
make_wrapper -files [get_files top.bd] -import -fileset sources_1 -top

#set_property strategy Flow_PerfOptimized_high [get_runs synth_1]
#set_property STEPS.SYNTH_DESIGN.ARGS.DIRECTIVE AlternateRoutability [get_runs synth_1]
#set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING true [get_runs synth_1]
#set_property strategy Performance_ExtraTimingOpt [get_runs impl_1]
#set_property STEPS.OPT_DESIGN.ARGS.DIRECTIVE Explore [get_runs impl_1]
#set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
#set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
#set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]

# out-of-context synth can't be used for bitstream generation
# set_property -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} -value {-mode out_of_context} -objects [get_runs synth_1]
launch_runs -to_step write_bitstream impl_1
wait_on_run [get_runs impl_1]

# generate synthesis report
open_run impl_1
report_utilization -hierarchical -hierarchical_depth 4 -file synth_report.xml -format xml
close_project
"""
