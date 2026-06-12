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

# Definitional proc to organize widgets for parameters.
proc init_gui { IPINST } {
  ipgui::add_param $IPINST -name "Component_Name"

  #---> Adding Page ---------------------------------------------------------------------------------------------------------------------#

  set General_Config [ipgui::add_page $IPINST -name "General Configuration"]

      #---> Adding Group ----------------------------------------------------------------------------------------------------------------#

      set Class_Group [ipgui::add_group $IPINST -parent $General_Config -name "SMBus Device Class Configuration" ]

          #---> Adding Params -----------------------------------------------------------------------------------------------------------#

          set SMBUS_DEV_CLASS [ipgui::add_param $IPINST -name SMBUS_DEV_CLASS -widget comboBox -parent $Class_Group]
          set_property tooltip "SMBUS_DEV_CLASS: Specify the default SMBus device class that the IP will operate at." $SMBUS_DEV_CLASS

          #---> End Params --------------------------------------------------------------------------------------------------------------#

      #---> End Group -------------------------------------------------------------------------------------------------------------------#

      #---> Adding Group ----------------------------------------------------------------------------------------------------------------#

      set Target_Group [ipgui::add_group $IPINST -parent $General_Config -name "SMBus Target Configuration" ]

          #---> Adding Params -----------------------------------------------------------------------------------------------------------#

          set NUM_TARGET_DEVICES [ipgui::add_param $IPINST -name NUM_TARGET_DEVICES -parent $Target_Group]
          set_property tooltip "NUM_TARGET_DEVICES: Specify the number of devices to be supported when operating as a Target." $NUM_TARGET_DEVICES

          #---> End Params --------------------------------------------------------------------------------------------------------------#

      #---> End Group -------------------------------------------------------------------------------------------------------------------#

      #---> Adding Group ----------------------------------------------------------------------------------------------------------------#

      set Clock_Group [ipgui::add_group $IPINST -parent $General_Config -name "S_AXI Clock Frequency Configuration" ]

          #---> Adding Params -----------------------------------------------------------------------------------------------------------#

          set FREQ_HZ_AXI_ACLK [ipgui::add_param $IPINST -name FREQ_HZ_AXI_ACLK -parent $Clock_Group]
          set_property tooltip "FREQ_HZ_AXI_ACLK: Specify the frequency (in Hz) of the input s_axi_aclk clock." $FREQ_HZ_AXI_ACLK

          #---> End Params --------------------------------------------------------------------------------------------------------------#

      #---> End Group -------------------------------------------------------------------------------------------------------------------#

  #---> End Page ------------------------------------------------------------------------------------------------------------------------#
}

#==========================================================================================================================================#
# Model Parameter Update Procedures
#==========================================================================================================================================#

proc update_MODELPARAM_VALUE.C_MAJOR_VERSION { MODELPARAM_VALUE.C_MAJOR_VERSION PROJECT_PARAM.DEVICE IPINST } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value

  set version [get_property VERSION [current_ipcomp]]
  regexp {^(\d+)\.(\d+)$} $version -> major minor
	set_property value $major ${MODELPARAM_VALUE.C_MAJOR_VERSION}

}

proc update_MODELPARAM_VALUE.C_MINOR_VERSION { MODELPARAM_VALUE.C_MINOR_VERSION PROJECT_PARAM.DEVICE IPINST } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value

  set version [get_property VERSION [current_ipcomp]]
  regexp {^(\d+)\.(\d+)$} $version -> major minor
	set_property value $minor ${MODELPARAM_VALUE.C_MINOR_VERSION}

}

proc update_MODELPARAM_VALUE.C_CORE_REVISION { MODELPARAM_VALUE.C_CORE_REVISION PROJECT_PARAM.DEVICE IPINST } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value

  set revision [get_property CORE_REVISION [current_ipcomp]]
  set_property value $revision ${MODELPARAM_VALUE.C_CORE_REVISION}

}

proc update_MODELPARAM_VALUE.C_FREQ_HZ_AXI_ACLK { MODELPARAM_VALUE.C_FREQ_HZ_AXI_ACLK PARAM_VALUE.FREQ_HZ_AXI_ACLK IPINST } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value

  set_property value [get_property value ${PARAM_VALUE.FREQ_HZ_AXI_ACLK}] ${MODELPARAM_VALUE.C_FREQ_HZ_AXI_ACLK}

}

proc update_MODELPARAM_VALUE.C_SMBUS_DEV_CLASS { MODELPARAM_VALUE.C_SMBUS_DEV_CLASS PARAM_VALUE.SMBUS_DEV_CLASS IPINST } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value

  set_property value [get_property value ${PARAM_VALUE.SMBUS_DEV_CLASS}] ${MODELPARAM_VALUE.C_SMBUS_DEV_CLASS}

}

proc update_MODELPARAM_VALUE.C_NUM_TARGET_DEVICES { MODELPARAM_VALUE.C_NUM_TARGET_DEVICES PARAM_VALUE.NUM_TARGET_DEVICES IPINST } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value

  set_property value [get_property value ${PARAM_VALUE.NUM_TARGET_DEVICES}] ${MODELPARAM_VALUE.C_NUM_TARGET_DEVICES}

}
