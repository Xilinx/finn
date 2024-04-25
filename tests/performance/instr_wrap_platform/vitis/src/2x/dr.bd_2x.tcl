

#---------------------------
# Constant blocks
#---------------------------

#---------------------------
# Platform Parameters for vmk180_thin
#---------------------------
set icn_ctrl [get_bd_cell /icn_ctrl]
    
set_property -dict [ list \
  CONFIG.NUM_SI 1 \
  CONFIG.NUM_MI 2 \
  CONFIG.NUM_CLKS 1 \
  ] $icn_ctrl

#---------------------------
# Instantiating finn_design_0
#---------------------------
set finn_design_0 [create_bd_cell -type ip -vlnv xilinx_finn:finn:finn_design:1.0 finn_design_0]
  

#---------------------------
# Instantiating instrumentation_wrapper_0
#---------------------------
set instrumentation_wrapper_0 [create_bd_cell -type ip -vlnv xilinx.com:hls:instrumentation_wrapper:1.0 instrumentation_wrapper_0]
  

#---------------------------
# Instantiating axi_intc_0_intr_1_interrupt_concat
#---------------------------
set axi_intc_0_intr_1_interrupt_concat [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 axi_intc_0_intr_1_interrupt_concat]
  
set_property -dict [ list  \
  CONFIG.NUM_PORTS {15}  ] $axi_intc_0_intr_1_interrupt_concat

#---------------------------
# Instantiating irq_const_tieoff
#---------------------------
set irq_const_tieoff [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 irq_const_tieoff]
  
set_property -dict [ list  \
  CONFIG.CONST_WIDTH {1} \
  CONFIG.CONST_VAL {0}  ] $irq_const_tieoff
#---------------------------
# Clock/Reset Annotation Registration
#---------------------------

#::sdsoc::mark_pfm_border


#---------------------------
# Connectivity Phase 1
#---------------------------
connect_bd_intf_net \
  [get_bd_intf_pins -auto_enable /icn_ctrl/M01_AXI] \
  [get_bd_intf_pins -auto_enable /instrumentation_wrapper_0/s_axi_ctrl] \

connect_bd_intf_net \
  [get_bd_intf_pins -auto_enable /finn_design_0/m_axis_0] \
  [get_bd_intf_pins -auto_enable /instrumentation_wrapper_0/finnox] \

connect_bd_intf_net \
  [get_bd_intf_pins -auto_enable /instrumentation_wrapper_0/finnix] \
  [get_bd_intf_pins -auto_enable /finn_design_0/s_axis_0] \

connect_bd_net  \
  [get_bd_pins -auto_enable /axi_intc_0_intr_1_interrupt_concat/dout] \
  [get_bd_pins -auto_enable /axi_intc_0/intr] \

connect_bd_net  \
  [get_bd_pins -auto_enable /irq_const_tieoff/dout] \
  [get_bd_pins -auto_enable /axi_intc_0_intr_1_interrupt_concat/In1] \
  [get_bd_pins -auto_enable /axi_intc_0_intr_1_interrupt_concat/In0] \
  [get_bd_pins -auto_enable /axi_intc_0_intr_1_interrupt_concat/In2] \
  [get_bd_pins -auto_enable /axi_intc_0_intr_1_interrupt_concat/In3] \
  [get_bd_pins -auto_enable /axi_intc_0_intr_1_interrupt_concat/In4] \
  [get_bd_pins -auto_enable /axi_intc_0_intr_1_interrupt_concat/In5] \
  [get_bd_pins -auto_enable /axi_intc_0_intr_1_interrupt_concat/In6] \
  [get_bd_pins -auto_enable /axi_intc_0_intr_1_interrupt_concat/In7] \
  [get_bd_pins -auto_enable /axi_intc_0_intr_1_interrupt_concat/In8] \
  [get_bd_pins -auto_enable /axi_intc_0_intr_1_interrupt_concat/In9] \
  [get_bd_pins -auto_enable /axi_intc_0_intr_1_interrupt_concat/In10] \
  [get_bd_pins -auto_enable /axi_intc_0_intr_1_interrupt_concat/In11] \
  [get_bd_pins -auto_enable /axi_intc_0_intr_1_interrupt_concat/In12] \
  [get_bd_pins -auto_enable /axi_intc_0_intr_1_interrupt_concat/In13] \
  [get_bd_pins -auto_enable /axi_intc_0_intr_1_interrupt_concat/In14] \


#---------------------------
# Clock/Reset Annotation
#---------------------------

#set_property HDL_ATTRIBUTE.CLOCK_AUTOMATION true $finn_design_0
set_property HDL_ATTRIBUTE.ap_clk.refClockId {1} $finn_design_0
set_property HDL_ATTRIBUTE.ap_clk.FREQ_HZ {200000000} $finn_design_0
set_property HDL_ATTRIBUTE.ap_clk.FREQ_HZ_TOLERANCE {10000000} $finn_design_0
set_property HDL_ATTRIBUTE.ap_clk2x.refClockId {0} $finn_design_0
set_property HDL_ATTRIBUTE.ap_clk2x.FREQ_HZ {400000000} $finn_design_0
set_property HDL_ATTRIBUTE.ap_clk2x.FREQ_HZ_TOLERANCE {20000000} $finn_design_0

#set_property HDL_ATTRIBUTE.CLOCK_AUTOMATION true $instrumentation_wrapper_0
set_property HDL_ATTRIBUTE.ap_clk.refClockId {1} $instrumentation_wrapper_0
set_property HDL_ATTRIBUTE.ap_clk.FREQ_HZ {200000000} $instrumentation_wrapper_0
set_property HDL_ATTRIBUTE.ap_clk.FREQ_HZ_TOLERANCE {10000000} $instrumentation_wrapper_0


#---------------------------
# Invoke clock automation
#---------------------------

#::sdsoc::run_clock_reset_automation
#::sdsoc::erase_clock_properties
set_param bd.clkrstautomationv2 1
::bd::clkrst::apply_clk_rst_automation


#---------------------------
# Connectivity Phase 2
#---------------------------

delete_bd_objs [get_bd_nets /rst_clk_wizard_0_400M_peripheral_aresetn]
connect_bd_net -net rst_clk_wizard_0_200M_peripheral_aresetn [get_bd_pins finn_design_0/ap_rst_n]

#---------------------------
# Create Stream Map file
#---------------------------
set stream_subsystems [get_bd_cells * -hierarchical -quiet -filter {VLNV =~ "*:*:sdx_stream_subsystem:*"}]
if {[string length $stream_subsystems] > 0} {    
  set xmlFile $vpl_output_dir/qdma_stream_map.xml
  set fp [open ${xmlFile} w]
  puts $fp "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
  puts $fp "<xd:streamMap xmlns:xd=\"http://www.xilinx.com/xd\">"
  foreach streamSS [get_bd_cells * -hierarchical -quiet -filter {VLNV =~ "*:*:sdx_stream_subsystem:*"}] {
    set ssInstance [string trimleft $streamSS /]
    set ssRegion [get_property CONFIG.SLR_ASSIGNMENTS $streamSS]
    foreach ssIntf [get_bd_intf_pins $streamSS/* -quiet -filter {NAME=~"S??_AXIS"}] {
      set pinName [get_property NAME $ssIntf]
      set routeId [sdx_stream_subsystem::get_routeid $ssIntf]
      set flowId [sdx_stream_subsystem::get_flowid $ssIntf]
      puts $fp "  <xd:streamRoute xd:instanceRef=\"$ssInstance\" xd:portRef=\"$pinName\" xd:route=\"$routeId\" xd:flow=\"$flowId\" xd:region=\"$ssRegion\">"
      foreach connection [find_bd_objs -relation connected_to $ssIntf -thru_hier] {
        set connectedRegion [get_property CONFIG.SLR_ASSIGNMENTS [bd::utils::get_parent $connection]]
        set connectedPort [bd::utils::get_short_name $connection]
        set connectedInst [string trimleft [bd::utils::get_parent $connection] /]
        puts $fp "    <xd:connection xd:instanceRef=\"$connectedInst\" xd:portRef=\"$connectedPort\" xd:region=\"$connectedRegion\"/>"
      }
      puts $fp "  </xd:streamRoute>"
    }
    foreach ssIntf [get_bd_intf_pins $streamSS/* -quiet -filter {NAME=~"M??_AXIS"}] {
      set pinName [get_property NAME $ssIntf]
      set routeId [sdx_stream_subsystem::get_routeid $ssIntf]
      set flowId [sdx_stream_subsystem::get_flowid $ssIntf]
      puts $fp "  <xd:streamRoute xd:instanceRef=\"$ssInstance\" xd:portRef=\"$pinName\" xd:route=\"$routeId\" xd:flow=\"$flowId\" xd:region=\"$ssRegion\">"
      foreach connection [find_bd_objs -relation connected_to $ssIntf -thru_hier] {
        set connectedRegion [get_property CONFIG.SLR_ASSIGNMENTS [bd::utils::get_parent $connection]]
        set connectedPort [bd::utils::get_short_name $connection]
        set connectedInst [string trimleft [bd::utils::get_parent $connection] /]
        puts $fp "    <xd:connection xd:instanceRef=\"$connectedInst\" xd:portRef=\"$connectedPort\" xd:region=\"$connectedRegion\"/>"
      }
      puts $fp "  </xd:streamRoute>"
    }
  }
  puts $fp "</xd:streamMap>"
  close $fp
}


