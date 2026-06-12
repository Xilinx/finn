# (c) Copyright 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
#
# This file contains confidential and proprietary information
# of AMD and is protected under U.S. and international copyright
# and other intellectual property laws.
#
# DISCLAIMER
# This disclaimer is not a license and does not grant any
# rights to the materials distributed herewith. Except as
# otherwise provided in a valid license issued to you by
# AMD, and to the maximum extent permitted by applicable
# law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
# WITH ALL FAULTS, AND AMD HEREBY DISCLAIMS ALL WARRANTIES
# AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
# BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
# INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
# (2) AMD shall not be liable (whether in contract or tort,
# including negligence, or under any other theory of
# liability) for any loss or damage of any kind or nature
# related to, arising under or in connection with these
# materials, including for any direct, or any indirect,
# special, incidental, or consequential loss or damage
# (including loss of data, profits, goodwill, or any type of
# loss or damage suffered as a result of any action brought
# by a third party) even if such damage or loss was
# reasonably foreseeable or AMD had been advised of the
# possibility of the same.
#
# CRITICAL APPLICATIONS
# AMD products are not designed or intended to be fail-
# safe, or for use in any application requiring fail-safe
# performance, such as life-support or safety devices or
# systems, Class III medical devices, nuclear facilities,
# applications related to the deployment of airbags, or any
# other applications that could lead to death, personal
# injury, or severe property or environmental damage
# (individually and collectively, "Critical
# Applications"). Customer assumes the sole risk and
# liability of any use of AMD products in Critical
# Applications, subject only to applicable laws and
# regulations governing limitations on product liability.
#
# THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
# PART OF THIS FILE AT ALL TIMES.
############################################################

#==============================================================================#
# Initialization Procedure
#==============================================================================#

proc init { cellPath otherInfo } {
  bd::mark_propagate_overrideable [get_bd_cells $cellPath] [list FREQ_HZ_AXI_ACLK]
}

#==============================================================================#
# Post IP Configuration Procedure
#==============================================================================#

proc post_config_ip { cellPath otherInfo } {
}

#==============================================================================#
# Post Propagate Procedure
#==============================================================================#

proc post_propagate { cellPath otherInfo } {
  set cell    [get_bd_cells $cellPath]
  set clk     [get_bd_pins $cellPath/s_axi_aclk]
  set freq_hz [get_property CONFIG.FREQ_HZ $clk]
  set val_src [string toupper [get_property CONFIG.FREQ_HZ_AXI_ACLK.VALUE_SRC $cell]]
  if {$val_src ne "USER"} {
    if {$freq_hz == ""} {
      ::bd::send_msg -of $cellPath -type error -msg_id 1 -text "AXI Clock Frequency has not propagated"
    } else {
      set_property CONFIG.FREQ_HZ_AXI_ACLK $freq_hz $cell
    }
  }
}
