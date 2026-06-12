# (c) Copyright 2023, Advanced Micro Devices, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a 
# copy of this software and associated documentation files (the "Software"), 
# to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, 
# and/or sell copies of the Software, and to permit persons to whom the 
# Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.
############################################################
set core [get_ips]
#set core [ipx::get_cores -from project]
set CompName [get_property NAME $core]
set ipdef [get_property IPDEF $core]

create_bd_design ${CompName}_testbd
set dut [ create_bd_cell -type ip -vlnv $ipdef $CompName ]

  set orig [dict create]
  foreach p [list_property $core CONFIG.*] {
    if {$p != "CONFIG.Component_Name" && [llength [get_property $p $dut]] > 0} {
      dict set orig $p [get_property $p $core]
      dict set orig $p.VALUE_SRC [get_property $p.VALUE_SRC $core]
    }
  }

  set_property -dict $orig $dut

  # Create ports
  set aresetn_0 [ create_bd_port -dir I -type rst aresetn_0 ]
  set aclk_0 [ create_bd_port -dir I -type clk -freq_hz 100000000 aclk_0 ]
  set_property CONFIG.ASSOCIATED_RESET {aresetn_0} $aclk_0
  connect_bd_net -net aclk_0_net [get_bd_ports aclk_0] [get_bd_pins $CompName/aclk_pcie] [get_bd_pins $CompName/aclk_ctrl]
  connect_bd_net -net aresetn_0_net [get_bd_ports aresetn_0] [get_bd_pins $CompName/aresetn_pcie] [get_bd_pins $CompName/aresetn_ctrl]
  make_bd_intf_pins_external  [get_bd_intf_pins $CompName/s_pcie4_cfg_ext]
  
  # Create instances of axi_vip
  set num_pfs [get_property CONFIG.C_NUM_PFS $dut]
  for {set pf 0} {$pf < $num_pfs} {incr pf} {
    set addr_wid [get_property CONFIG.C_PF${pf}_S_AXI_ADDR_WIDTH $dut]
    create_bd_cell -type ip -vlnv xilinx.com:ip:axi_vip:* vip_${pf}
    set_property -dict [ list \
     CONFIG.ADDR_WIDTH $addr_wid \
     CONFIG.DATA_WIDTH {32} \
     CONFIG.HAS_BRESP {1} \
     CONFIG.HAS_RRESP {1} \
     CONFIG.HAS_WSTRB {1} \
     CONFIG.INTERFACE_MODE {MASTER} \
     CONFIG.PROTOCOL {AXI4LITE} \
     CONFIG.READ_WRITE_MODE {READ_WRITE} \
     ] [get_bd_cells vip_${pf}]
    connect_bd_net -net aclk_0_net [get_bd_pins vip_${pf}/aclk]
    connect_bd_net -net aresetn_0_net [get_bd_pins vip_${pf}/aresetn]
    connect_bd_intf_net -intf_net vip_${pf}_net [get_bd_intf_pins vip_${pf}/M_AXI] [get_bd_intf_pins $CompName/s_axi_ctrl_pf${pf}]
  }
  
  # Create address segments
  create_bd_addr_seg -offset 0x00000000 -range 0x00010000 [get_bd_addr_spaces vip_0/Master_AXI] [get_bd_addr_segs $CompName/s_axi_ctrl_pf0/reg0] seg_s_axi_ctrl_pf0
  if {$num_pfs > 1} {create_bd_addr_seg -offset 0x00000000 -range 0x00010000 [get_bd_addr_spaces vip_1/Master_AXI] [get_bd_addr_segs $CompName/s_axi_ctrl_pf1/reg0] seg_s_axi_ctrl_pf1}
  if {$num_pfs > 2} {create_bd_addr_seg -offset 0x00000000 -range 0x00010000 [get_bd_addr_spaces vip_2/Master_AXI] [get_bd_addr_segs $CompName/s_axi_ctrl_pf2/reg0] seg_s_axi_ctrl_pf2}
  if {$num_pfs > 3} {create_bd_addr_seg -offset 0x00000000 -range 0x00010000 [get_bd_addr_spaces vip_3/Master_AXI] [get_bd_addr_segs $CompName/s_axi_ctrl_pf3/reg0] seg_s_axi_ctrl_pf3}

validate_bd_design
save_bd_design
