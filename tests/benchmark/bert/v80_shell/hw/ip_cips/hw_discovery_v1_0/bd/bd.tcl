# (c) Copyright 2022, Advanced Micro Devices, Inc.
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

proc post_config_ip {cellpath otherInfo } {
}

proc pre_propagate {cellpath undefined_params} {
  set props [list \
		C_PF0_NUM_SLOTS_BAR_LAYOUT_TABLE C_PF0_BAR_INDEX C_PF0_HIGH_OFFSET C_PF0_LOW_OFFSET \
	  C_PF0_ENTRY_TYPE_0 C_PF0_ENTRY_TYPE_1 C_PF0_ENTRY_TYPE_2 C_PF0_ENTRY_TYPE_3 C_PF0_ENTRY_TYPE_4 C_PF0_ENTRY_TYPE_5 C_PF0_ENTRY_TYPE_6 C_PF0_ENTRY_TYPE_7 C_PF0_ENTRY_TYPE_8 C_PF0_ENTRY_TYPE_9 C_PF0_ENTRY_TYPE_10 C_PF0_ENTRY_TYPE_11 C_PF0_ENTRY_TYPE_12 C_PF0_ENTRY_TYPE_13 \
	  C_PF0_ENTRY_BAR_0 C_PF0_ENTRY_BAR_1 C_PF0_ENTRY_BAR_2 C_PF0_ENTRY_BAR_3 C_PF0_ENTRY_BAR_4 C_PF0_ENTRY_BAR_5 C_PF0_ENTRY_BAR_6 C_PF0_ENTRY_BAR_7 C_PF0_ENTRY_BAR_8 C_PF0_ENTRY_BAR_9 C_PF0_ENTRY_BAR_10 C_PF0_ENTRY_BAR_11 C_PF0_ENTRY_BAR_12 C_PF0_ENTRY_BAR_13 \
    C_PF0_ENTRY_ADDR_0 C_PF0_ENTRY_ADDR_1 C_PF0_ENTRY_ADDR_2 C_PF0_ENTRY_ADDR_3 C_PF0_ENTRY_ADDR_4 C_PF0_ENTRY_ADDR_5 C_PF0_ENTRY_ADDR_6 C_PF0_ENTRY_ADDR_7 C_PF0_ENTRY_ADDR_8 C_PF0_ENTRY_ADDR_9 C_PF0_ENTRY_ADDR_10 C_PF0_ENTRY_ADDR_11 C_PF0_ENTRY_ADDR_12 C_PF0_ENTRY_ADDR_13 \
    C_PF0_ENTRY_VERSION_TYPE_0 C_PF0_ENTRY_VERSION_TYPE_1 C_PF0_ENTRY_VERSION_TYPE_2 C_PF0_ENTRY_VERSION_TYPE_3 C_PF0_ENTRY_VERSION_TYPE_4 C_PF0_ENTRY_VERSION_TYPE_5 C_PF0_ENTRY_VERSION_TYPE_6 C_PF0_ENTRY_VERSION_TYPE_7 C_PF0_ENTRY_VERSION_TYPE_8 C_PF0_ENTRY_VERSION_TYPE_9 C_PF0_ENTRY_VERSION_TYPE_10 C_PF0_ENTRY_VERSION_TYPE_11 C_PF0_ENTRY_VERSION_TYPE_12 C_PF0_ENTRY_VERSION_TYPE_13 \
    C_PF0_ENTRY_MAJOR_VERSION_0 C_PF0_ENTRY_MAJOR_VERSION_1 C_PF0_ENTRY_MAJOR_VERSION_2 C_PF0_ENTRY_MAJOR_VERSION_3 C_PF0_ENTRY_MAJOR_VERSION_4 C_PF0_ENTRY_MAJOR_VERSION_5 C_PF0_ENTRY_MAJOR_VERSION_6 C_PF0_ENTRY_MAJOR_VERSION_7 C_PF0_ENTRY_MAJOR_VERSION_8 C_PF0_ENTRY_MAJOR_VERSION_9 C_PF0_ENTRY_MAJOR_VERSION_10 C_PF0_ENTRY_MAJOR_VERSION_11 C_PF0_ENTRY_MAJOR_VERSION_12 C_PF0_ENTRY_MAJOR_VERSION_13 \
    C_PF0_ENTRY_MINOR_VERSION_0 C_PF0_ENTRY_MINOR_VERSION_1 C_PF0_ENTRY_MINOR_VERSION_2 C_PF0_ENTRY_MINOR_VERSION_3 C_PF0_ENTRY_MINOR_VERSION_4 C_PF0_ENTRY_MINOR_VERSION_5 C_PF0_ENTRY_MINOR_VERSION_6 C_PF0_ENTRY_MINOR_VERSION_7 C_PF0_ENTRY_MINOR_VERSION_8 C_PF0_ENTRY_MINOR_VERSION_9 C_PF0_ENTRY_MINOR_VERSION_10 C_PF0_ENTRY_MINOR_VERSION_11 C_PF0_ENTRY_MINOR_VERSION_12 C_PF0_ENTRY_MINOR_VERSION_13 \
    C_PF0_ENTRY_RSVD0_0 C_PF0_ENTRY_RSVD0_1 C_PF0_ENTRY_RSVD0_2 C_PF0_ENTRY_RSVD0_3 C_PF0_ENTRY_RSVD0_4 C_PF0_ENTRY_RSVD0_5 C_PF0_ENTRY_RSVD0_6 C_PF0_ENTRY_RSVD0_7 C_PF0_ENTRY_RSVD0_8 C_PF0_ENTRY_RSVD0_9 C_PF0_ENTRY_RSVD0_10 C_PF0_ENTRY_RSVD0_11 C_PF0_ENTRY_RSVD0_12 C_PF0_ENTRY_RSVD0_13 \
		C_PF1_NUM_SLOTS_BAR_LAYOUT_TABLE C_PF1_BAR_INDEX C_PF1_HIGH_OFFSET C_PF1_LOW_OFFSET \
	  C_PF1_ENTRY_TYPE_0 C_PF1_ENTRY_TYPE_1 C_PF1_ENTRY_TYPE_2 C_PF1_ENTRY_TYPE_3 C_PF1_ENTRY_TYPE_4 C_PF1_ENTRY_TYPE_5 C_PF1_ENTRY_TYPE_6 C_PF1_ENTRY_TYPE_7 C_PF1_ENTRY_TYPE_8 C_PF1_ENTRY_TYPE_9 C_PF1_ENTRY_TYPE_10 C_PF1_ENTRY_TYPE_11 C_PF1_ENTRY_TYPE_12 C_PF1_ENTRY_TYPE_13 \
	  C_PF1_ENTRY_BAR_0 C_PF1_ENTRY_BAR_1 C_PF1_ENTRY_BAR_2 C_PF1_ENTRY_BAR_3 C_PF1_ENTRY_BAR_4 C_PF1_ENTRY_BAR_5 C_PF1_ENTRY_BAR_6 C_PF1_ENTRY_BAR_7 C_PF1_ENTRY_BAR_8 C_PF1_ENTRY_BAR_9 C_PF1_ENTRY_BAR_10 C_PF1_ENTRY_BAR_11 C_PF1_ENTRY_BAR_12 C_PF1_ENTRY_BAR_13 \
    C_PF1_ENTRY_ADDR_0 C_PF1_ENTRY_ADDR_1 C_PF1_ENTRY_ADDR_2 C_PF1_ENTRY_ADDR_3 C_PF1_ENTRY_ADDR_4 C_PF1_ENTRY_ADDR_5 C_PF1_ENTRY_ADDR_6 C_PF1_ENTRY_ADDR_7 C_PF1_ENTRY_ADDR_8 C_PF1_ENTRY_ADDR_9 C_PF1_ENTRY_ADDR_10 C_PF1_ENTRY_ADDR_11 C_PF1_ENTRY_ADDR_12 C_PF1_ENTRY_ADDR_13 \
    C_PF1_ENTRY_VERSION_TYPE_0 C_PF1_ENTRY_VERSION_TYPE_1 C_PF1_ENTRY_VERSION_TYPE_2 C_PF1_ENTRY_VERSION_TYPE_3 C_PF1_ENTRY_VERSION_TYPE_4 C_PF1_ENTRY_VERSION_TYPE_5 C_PF1_ENTRY_VERSION_TYPE_6 C_PF1_ENTRY_VERSION_TYPE_7 C_PF1_ENTRY_VERSION_TYPE_8 C_PF1_ENTRY_VERSION_TYPE_9 C_PF1_ENTRY_VERSION_TYPE_10 C_PF1_ENTRY_VERSION_TYPE_11 C_PF1_ENTRY_VERSION_TYPE_12 C_PF1_ENTRY_VERSION_TYPE_13 \
    C_PF1_ENTRY_MAJOR_VERSION_0 C_PF1_ENTRY_MAJOR_VERSION_1 C_PF1_ENTRY_MAJOR_VERSION_2 C_PF1_ENTRY_MAJOR_VERSION_3 C_PF1_ENTRY_MAJOR_VERSION_4 C_PF1_ENTRY_MAJOR_VERSION_5 C_PF1_ENTRY_MAJOR_VERSION_6 C_PF1_ENTRY_MAJOR_VERSION_7 C_PF1_ENTRY_MAJOR_VERSION_8 C_PF1_ENTRY_MAJOR_VERSION_9 C_PF1_ENTRY_MAJOR_VERSION_10 C_PF1_ENTRY_MAJOR_VERSION_11 C_PF1_ENTRY_MAJOR_VERSION_12 C_PF1_ENTRY_MAJOR_VERSION_13 \
    C_PF1_ENTRY_MINOR_VERSION_0 C_PF1_ENTRY_MINOR_VERSION_1 C_PF1_ENTRY_MINOR_VERSION_2 C_PF1_ENTRY_MINOR_VERSION_3 C_PF1_ENTRY_MINOR_VERSION_4 C_PF1_ENTRY_MINOR_VERSION_5 C_PF1_ENTRY_MINOR_VERSION_6 C_PF1_ENTRY_MINOR_VERSION_7 C_PF1_ENTRY_MINOR_VERSION_8 C_PF1_ENTRY_MINOR_VERSION_9 C_PF1_ENTRY_MINOR_VERSION_10 C_PF1_ENTRY_MINOR_VERSION_11 C_PF1_ENTRY_MINOR_VERSION_12 C_PF1_ENTRY_MINOR_VERSION_13 \
    C_PF1_ENTRY_RSVD0_0 C_PF1_ENTRY_RSVD0_1 C_PF1_ENTRY_RSVD0_2 C_PF1_ENTRY_RSVD0_3 C_PF1_ENTRY_RSVD0_4 C_PF1_ENTRY_RSVD0_5 C_PF1_ENTRY_RSVD0_6 C_PF1_ENTRY_RSVD0_7 C_PF1_ENTRY_RSVD0_8 C_PF1_ENTRY_RSVD0_9 C_PF1_ENTRY_RSVD0_10 C_PF1_ENTRY_RSVD0_11 C_PF1_ENTRY_RSVD0_12 C_PF1_ENTRY_RSVD0_13 \
		C_PF2_NUM_SLOTS_BAR_LAYOUT_TABLE C_PF2_BAR_INDEX C_PF2_HIGH_OFFSET C_PF2_LOW_OFFSET \
	  C_PF2_ENTRY_TYPE_0 C_PF2_ENTRY_TYPE_1 C_PF2_ENTRY_TYPE_2 C_PF2_ENTRY_TYPE_3 C_PF2_ENTRY_TYPE_4 C_PF2_ENTRY_TYPE_5 C_PF2_ENTRY_TYPE_6 C_PF2_ENTRY_TYPE_7 C_PF2_ENTRY_TYPE_8 C_PF2_ENTRY_TYPE_9 C_PF2_ENTRY_TYPE_10 C_PF2_ENTRY_TYPE_11 C_PF2_ENTRY_TYPE_12 C_PF2_ENTRY_TYPE_13 \
	  C_PF2_ENTRY_BAR_0 C_PF2_ENTRY_BAR_1 C_PF2_ENTRY_BAR_2 C_PF2_ENTRY_BAR_3 C_PF2_ENTRY_BAR_4 C_PF2_ENTRY_BAR_5 C_PF2_ENTRY_BAR_6 C_PF2_ENTRY_BAR_7 C_PF2_ENTRY_BAR_8 C_PF2_ENTRY_BAR_9 C_PF2_ENTRY_BAR_10 C_PF2_ENTRY_BAR_11 C_PF2_ENTRY_BAR_12 C_PF2_ENTRY_BAR_13 \
    C_PF2_ENTRY_ADDR_0 C_PF2_ENTRY_ADDR_1 C_PF2_ENTRY_ADDR_2 C_PF2_ENTRY_ADDR_3 C_PF2_ENTRY_ADDR_4 C_PF2_ENTRY_ADDR_5 C_PF2_ENTRY_ADDR_6 C_PF2_ENTRY_ADDR_7 C_PF2_ENTRY_ADDR_8 C_PF2_ENTRY_ADDR_9 C_PF2_ENTRY_ADDR_10 C_PF2_ENTRY_ADDR_11 C_PF2_ENTRY_ADDR_12 C_PF2_ENTRY_ADDR_13 \
    C_PF2_ENTRY_VERSION_TYPE_0 C_PF2_ENTRY_VERSION_TYPE_1 C_PF2_ENTRY_VERSION_TYPE_2 C_PF2_ENTRY_VERSION_TYPE_3 C_PF2_ENTRY_VERSION_TYPE_4 C_PF2_ENTRY_VERSION_TYPE_5 C_PF2_ENTRY_VERSION_TYPE_6 C_PF2_ENTRY_VERSION_TYPE_7 C_PF2_ENTRY_VERSION_TYPE_8 C_PF2_ENTRY_VERSION_TYPE_9 C_PF2_ENTRY_VERSION_TYPE_10 C_PF2_ENTRY_VERSION_TYPE_11 C_PF2_ENTRY_VERSION_TYPE_12 C_PF2_ENTRY_VERSION_TYPE_13 \
    C_PF2_ENTRY_MAJOR_VERSION_0 C_PF2_ENTRY_MAJOR_VERSION_1 C_PF2_ENTRY_MAJOR_VERSION_2 C_PF2_ENTRY_MAJOR_VERSION_3 C_PF2_ENTRY_MAJOR_VERSION_4 C_PF2_ENTRY_MAJOR_VERSION_5 C_PF2_ENTRY_MAJOR_VERSION_6 C_PF2_ENTRY_MAJOR_VERSION_7 C_PF2_ENTRY_MAJOR_VERSION_8 C_PF2_ENTRY_MAJOR_VERSION_9 C_PF2_ENTRY_MAJOR_VERSION_10 C_PF2_ENTRY_MAJOR_VERSION_11 C_PF2_ENTRY_MAJOR_VERSION_12 C_PF2_ENTRY_MAJOR_VERSION_13 \
    C_PF2_ENTRY_MINOR_VERSION_0 C_PF2_ENTRY_MINOR_VERSION_1 C_PF2_ENTRY_MINOR_VERSION_2 C_PF2_ENTRY_MINOR_VERSION_3 C_PF2_ENTRY_MINOR_VERSION_4 C_PF2_ENTRY_MINOR_VERSION_5 C_PF2_ENTRY_MINOR_VERSION_6 C_PF2_ENTRY_MINOR_VERSION_7 C_PF2_ENTRY_MINOR_VERSION_8 C_PF2_ENTRY_MINOR_VERSION_9 C_PF2_ENTRY_MINOR_VERSION_10 C_PF2_ENTRY_MINOR_VERSION_11 C_PF2_ENTRY_MINOR_VERSION_12 C_PF2_ENTRY_MINOR_VERSION_13 \
    C_PF2_ENTRY_RSVD0_0 C_PF2_ENTRY_RSVD0_1 C_PF2_ENTRY_RSVD0_2 C_PF2_ENTRY_RSVD0_3 C_PF2_ENTRY_RSVD0_4 C_PF2_ENTRY_RSVD0_5 C_PF2_ENTRY_RSVD0_6 C_PF2_ENTRY_RSVD0_7 C_PF2_ENTRY_RSVD0_8 C_PF2_ENTRY_RSVD0_9 C_PF2_ENTRY_RSVD0_10 C_PF2_ENTRY_RSVD0_11 C_PF2_ENTRY_RSVD0_12 C_PF2_ENTRY_RSVD0_13 \
		C_PF3_NUM_SLOTS_BAR_LAYOUT_TABLE C_PF3_BAR_INDEX C_PF3_HIGH_OFFSET C_PF3_LOW_OFFSET \
	  C_PF3_ENTRY_TYPE_0 C_PF3_ENTRY_TYPE_1 C_PF3_ENTRY_TYPE_2 C_PF3_ENTRY_TYPE_3 C_PF3_ENTRY_TYPE_4 C_PF3_ENTRY_TYPE_5 C_PF3_ENTRY_TYPE_6 C_PF3_ENTRY_TYPE_7 C_PF3_ENTRY_TYPE_8 C_PF3_ENTRY_TYPE_9 C_PF3_ENTRY_TYPE_10 C_PF3_ENTRY_TYPE_11 C_PF3_ENTRY_TYPE_12 C_PF3_ENTRY_TYPE_13 \
	  C_PF3_ENTRY_BAR_0 C_PF3_ENTRY_BAR_1 C_PF3_ENTRY_BAR_2 C_PF3_ENTRY_BAR_3 C_PF3_ENTRY_BAR_4 C_PF3_ENTRY_BAR_5 C_PF3_ENTRY_BAR_6 C_PF3_ENTRY_BAR_7 C_PF3_ENTRY_BAR_8 C_PF3_ENTRY_BAR_9 C_PF3_ENTRY_BAR_10 C_PF3_ENTRY_BAR_11 C_PF3_ENTRY_BAR_12 C_PF3_ENTRY_BAR_13 \
    C_PF3_ENTRY_ADDR_0 C_PF3_ENTRY_ADDR_1 C_PF3_ENTRY_ADDR_2 C_PF3_ENTRY_ADDR_3 C_PF3_ENTRY_ADDR_4 C_PF3_ENTRY_ADDR_5 C_PF3_ENTRY_ADDR_6 C_PF3_ENTRY_ADDR_7 C_PF3_ENTRY_ADDR_8 C_PF3_ENTRY_ADDR_9 C_PF3_ENTRY_ADDR_10 C_PF3_ENTRY_ADDR_11 C_PF3_ENTRY_ADDR_12 C_PF3_ENTRY_ADDR_13 \
    C_PF3_ENTRY_VERSION_TYPE_0 C_PF3_ENTRY_VERSION_TYPE_1 C_PF3_ENTRY_VERSION_TYPE_2 C_PF3_ENTRY_VERSION_TYPE_3 C_PF3_ENTRY_VERSION_TYPE_4 C_PF3_ENTRY_VERSION_TYPE_5 C_PF3_ENTRY_VERSION_TYPE_6 C_PF3_ENTRY_VERSION_TYPE_7 C_PF3_ENTRY_VERSION_TYPE_8 C_PF3_ENTRY_VERSION_TYPE_9 C_PF3_ENTRY_VERSION_TYPE_10 C_PF3_ENTRY_VERSION_TYPE_11 C_PF3_ENTRY_VERSION_TYPE_12 C_PF3_ENTRY_VERSION_TYPE_13 \
    C_PF3_ENTRY_MAJOR_VERSION_0 C_PF3_ENTRY_MAJOR_VERSION_1 C_PF3_ENTRY_MAJOR_VERSION_2 C_PF3_ENTRY_MAJOR_VERSION_3 C_PF3_ENTRY_MAJOR_VERSION_4 C_PF3_ENTRY_MAJOR_VERSION_5 C_PF3_ENTRY_MAJOR_VERSION_6 C_PF3_ENTRY_MAJOR_VERSION_7 C_PF3_ENTRY_MAJOR_VERSION_8 C_PF3_ENTRY_MAJOR_VERSION_9 C_PF3_ENTRY_MAJOR_VERSION_10 C_PF3_ENTRY_MAJOR_VERSION_11 C_PF3_ENTRY_MAJOR_VERSION_12 C_PF3_ENTRY_MAJOR_VERSION_13 \
    C_PF3_ENTRY_MINOR_VERSION_0 C_PF3_ENTRY_MINOR_VERSION_1 C_PF3_ENTRY_MINOR_VERSION_2 C_PF3_ENTRY_MINOR_VERSION_3 C_PF3_ENTRY_MINOR_VERSION_4 C_PF3_ENTRY_MINOR_VERSION_5 C_PF3_ENTRY_MINOR_VERSION_6 C_PF3_ENTRY_MINOR_VERSION_7 C_PF3_ENTRY_MINOR_VERSION_8 C_PF3_ENTRY_MINOR_VERSION_9 C_PF3_ENTRY_MINOR_VERSION_10 C_PF3_ENTRY_MINOR_VERSION_11 C_PF3_ENTRY_MINOR_VERSION_12 C_PF3_ENTRY_MINOR_VERSION_13 \
    C_PF3_ENTRY_RSVD0_0 C_PF3_ENTRY_RSVD0_1 C_PF3_ENTRY_RSVD0_2 C_PF3_ENTRY_RSVD0_3 C_PF3_ENTRY_RSVD0_4 C_PF3_ENTRY_RSVD0_5 C_PF3_ENTRY_RSVD0_6 C_PF3_ENTRY_RSVD0_7 C_PF3_ENTRY_RSVD0_8 C_PF3_ENTRY_RSVD0_9 C_PF3_ENTRY_RSVD0_10 C_PF3_ENTRY_RSVD0_11 C_PF3_ENTRY_RSVD0_12 C_PF3_ENTRY_RSVD0_13]
  set cell [get_bd_cells $cellpath]
	puts "\[VSEC-BAR\] Cell: ${cellpath}"  
  if {[get_property CONFIG.C_MANUAL $cell] == 0} {
    set dflt [dict create]
    foreach p $props {
      dict set dflt CONFIG.${p}.VALUE_SRC DEFAULT
    }
    set_property -dict $dflt $cell
    
    if {[llength [get_property CONFIG.C_INJECT_ENDPOINTS $cell]] > 1} {
      set inject 1
      puts "\[VSEC-BAR\] ${cell} : Injecting PCIE Mapping Info from C_INJECT_ENDPOINTS"
    } elseif {([llength [namespace which vitis::get_pcie_mapping_for]] == 0 || [llength [namespace which vitis::get_endpoints_for_pcie_bar]] == 0)} {
      error "\[VSEC-BAR\] Cell ${cell} is configured for auto configuration, but necessary procedures to auto configure are not present."
      return
    } else {
      set inject 0
      puts "\[VSEC-BAR\] ${cell} is being automatically configured."
    }
  } else {
    puts "\[VSEC-BAR\] ${cell} is manually configured, and is skipping automatic configuration."
    return
  }
  
  set prop_vals [dict create]
  set num_pfs [get_property CONFIG.C_NUM_PFS [get_bd_cells $cellpath]]
	puts "\[VSEC-BAR\] ${cell} : Number of PFs = ${num_pfs}"  
  
  if {$inject == 0} {
    if {[llength [vitis::get_pcie_mapping_info]] > 0} {
      puts "\[VSEC-BAR\] ${cell} : Getting PCIE Mapping Info"
      foreach {pcie_info} [vitis::get_pcie_mapping_info] {
        set pf [dict get $pcie_info physical_function]
        set bar [dict get $pcie_info bar]
      	puts "\[VSEC-BAR\] ${cell} : Physical Function = ${pf}"
      	puts "\[VSEC-BAR\] ${cell} : BAR = ${bar}"  	
        foreach {endpoint} [vitis::get_endpoints_for_pcie_bar $pf $bar "ALL"] {
          puts "\[VSEC-BAR\] ${cell} : Endpoint = ${endpoint}"
          set bar_cell [bd::utils::get_parent [dict get $endpoint intf]]
          if {[string match "xilinx.com:ip:hw_discovery:*" [get_property VLNV $bar_cell]]} {
            set bar_high_addr [format 0x%08X [expr [dict get $endpoint offset] >> 32]]
            set bar_low_addr [format 0x%07X [expr ([dict get $endpoint offset] & 0xFFFFFFFF) / 16]]
            dict set prop_vals CONFIG.C_PF${pf}_BAR_INDEX $bar
            dict set prop_vals CONFIG.C_PF${pf}_HIGH_OFFSET $bar_high_addr
            dict set prop_vals CONFIG.C_PF${pf}_LOW_OFFSET $bar_low_addr
            puts "\[VSEC\] Setting $cell bar reference to [dict get $endpoint intf], @ high $bar_high_addr, low $bar_low_addr"
          }
        }
      }
    } else {
      error "\[VSEC-BAR\] ${cell} : No PCIE Mapping Info found"
    }
  } else {
    set pcie_mapping_info [dict get [get_property CONFIG.C_INJECT_ENDPOINTS $cell] pcie_mapping_info]
    foreach {pcie_info} ${pcie_mapping_info} {
      puts "\[VSEC-BAR\] ${cell} : PCIE Info: $pcie_info"
      set pf [dict get $pcie_info physical_function]
      set bar [dict get $pcie_info bar]
    	puts "\[VSEC-BAR\] ${cell} : Physical Function = ${pf}"
    	puts "\[VSEC-BAR\] ${cell} : BAR = ${bar}"  	
      set endpoints_for_pcie_bar [dict get [get_property CONFIG.C_INJECT_ENDPOINTS $cell] endpoints_for_pcie_bar $pf $bar]
      foreach {endpoint} ${endpoints_for_pcie_bar} {
        puts "\[VSEC-BAR\] ${cell} : Endpoint = ${endpoint}"
        set bar_cell [bd::utils::get_parent [dict get $endpoint intf]]
        if {[llength $bar_cell] > 0} {
          if {[string match "xilinx.com:ip:hw_discovery:*" [get_property VLNV $bar_cell]]} {
            set bar_high_addr [format 0x%08X [expr [dict get ${endpoint} offset] >> 32]]
            set bar_low_addr [format 0x%07X [expr ([dict get ${endpoint} offset] & 0xFFFFFFFF) / 16]]
            dict set prop_vals CONFIG.C_PF${pf}_BAR_INDEX $bar
            dict set prop_vals CONFIG.C_PF${pf}_HIGH_OFFSET $bar_high_addr
            dict set prop_vals CONFIG.C_PF${pf}_LOW_OFFSET $bar_low_addr
            puts "\[VSEC\] Setting $cell bar reference to [dict get ${endpoint} intf], @ high $bar_high_addr, low $bar_low_addr"
          }
        }
      }
    }
  }

  for {set i 0} {$i < $num_pfs} {incr i } {
    set bar_info [list]
    if {$inject == 0} {
  		puts "\[VSEC-BAR\] ${cell} : Getting PCIe Mapping Info for ${cell}/s_axi_ctrl_pf${i}"  	
  	  set bar_info [vitis::get_pcie_mapping_for [get_bd_intf_pins $cell/s_axi_ctrl_pf${i}]]
    } else {
  		puts "\[VSEC-BAR\] ${cell} : Injecting PCIe Mapping Info for ${cell}/s_axi_ctrl_pf${i}"  	
  	  if {[dict exist [get_property CONFIG.C_INJECT_ENDPOINTS $cell] pcie_mapping_for [get_bd_intf_pins $cell/s_axi_ctrl_pf${i}]]} {
    	  set bar_info [dict get [get_property CONFIG.C_INJECT_ENDPOINTS $cell] pcie_mapping_for [get_bd_intf_pins $cell/s_axi_ctrl_pf${i}]]
    	}
  	}
  	
	  if {[llength $bar_info] == 0} {
	    error "\[VSEC-BAR\] ${cell} Could not find a PCIe mapped BAR address"
	    return
	  }
	  set first_bar [lindex $bar_info 0]
	  set pf [dict get $first_bar physical_function]
	  set bar [dict get $first_bar bar]
    set ep_filter [get_property CONFIG.C_PF${pf}_ENDPOINT_NAMES $cell]
	  if {[llength [dict keys $ep_filter]] == 0} {
	    error "\[VSEC-BAR\] ${cell} Unrecognized BAR layout for Physical Function $pf"
	  } else {
	    puts "\[VSEC-BAR\] ${cell} Setting BAR layout for Physical Function $pf"
	  }
	  set index 0
		dict set prop_vals CONFIG.C_PF${pf}_NUM_SLOTS_BAR_LAYOUT_TABLE $index
    if {$inject == 0} {
  	  set endpoints [vitis::get_endpoints_for_pcie_bar $pf $bar]
	  } else {
      set endpoints [dict get [get_property CONFIG.C_INJECT_ENDPOINTS $cell] endpoints_for_pcie_bar $pf $bar]
	  }
	  foreach {pcie_peer} ${endpoints} {
      puts "\[VSEC-BAR\] ${cell} : PCIe Peer = ${pcie_peer}"
	    if {[dict exists $ep_filter [dict get $pcie_peer xrt_endpoint_name]]} {
	      set vlnv_list [split [dict get $pcie_peer reg_abs] ":"]
	      set vlnv_version_list [split [lindex $vlnv_list 3] "."]
	      set ep_info [dict get $ep_filter [dict get $pcie_peer xrt_endpoint_name]]
	      dict unset ep_filter [dict get $pcie_peer xrt_endpoint_name]
	      dict set prop_vals CONFIG.C_PF${pf}_ENTRY_ADDR_${index} [format 0x%012X [dict get $pcie_peer offset]]
	      dict set prop_vals CONFIG.C_PF${pf}_ENTRY_BAR_${index} $bar
	      dict set prop_vals CONFIG.C_PF${pf}_ENTRY_MAJOR_VERSION_${index} [lindex $vlnv_version_list 0]
	      dict set prop_vals CONFIG.C_PF${pf}_ENTRY_MINOR_VERSION_${index} [lindex $vlnv_version_list  1]
	      dict set prop_vals CONFIG.C_PF${pf}_ENTRY_TYPE_${index} [dict get $ep_info type]
	      dict set prop_vals CONFIG.C_PF${pf}_ENTRY_RSVD0_${index} [dict get $ep_info reserve]
	      dict set prop_vals CONFIG.C_PF${pf}_ENTRY_VERSION_TYPE_${index} 0x1
	
	      puts "\[VSEC-BAR\] Adding record to ${cell} for [dict get $pcie_peer xrt_endpoint_name] for endpoint [dict get $pcie_peer intf] at address [format 0x%012X [dict get $pcie_peer offset]]"
	      incr index
	      dict set prop_vals CONFIG.C_PF${pf}_NUM_SLOTS_BAR_LAYOUT_TABLE $index
	    }
	  }
	  set err 0
	  foreach {ep x} ${ep_filter} {
	    puts "\[VSEC-BAR\] Expected to find ${ep} on physical function ${pf} bar ${bar}, but failed to find any such endpoint"
	    set err 1
	  }
	  if {$err == 1} {
	    error "\[VSEC-BAR\] Aborting BAR layout table configuration due to being malformed"
	    return
	  }
	}
	puts "\[VSEC-BAR\] Configuring $cell with $prop_vals"
	set_property -dict $prop_vals $cell
}
