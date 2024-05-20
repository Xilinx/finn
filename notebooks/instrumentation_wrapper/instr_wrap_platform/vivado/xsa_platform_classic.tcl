#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: X11
#

## ===================================================================================
## Create Platform Vivado Project
## ===================================================================================
namespace eval _tcl {
  proc get_script_folder {} {
    set script_path [file normalize [info script]]
    set script_folder [file dirname $script_path]
    return $script_folder
  }
}

variable script_folder
set script_folder [_tcl::get_script_folder]

set BOARD_NAME    [lindex $argv 0]
set PLATFORM_NAME [lindex $argv 1]
set PLATFORM_TYPE ${PLATFORM_NAME}_custom
set VER "1.0"
puts "Creating HW Platform project for : \"$PLATFORM_NAME\""
set DEVICE_NAME [lindex $argv 2]
puts "Using : \"$DEVICE_NAME\""
set BOARD_LABEL [lindex $argv 4]
set BOARD_VER [lindex $argv 5]

set AP_CLK_MHZ [lindex $argv 6]
set AP_CLK_2X_MHZ [lindex $argv 7]

set BUILD_DIR build

create_project -f ${PLATFORM_NAME} ${BUILD_DIR}/${PLATFORM_NAME}_vivado -part $DEVICE_NAME

# set board part 
set_property BOARD_PART xilinx.com:${BOARD_LABEL}:part0:${BOARD_VER} [current_project]

set_property preferred_sim_model "tlm" [current_project]

## ===================================================================================
## Create Block Design
## ===================================================================================
source ./dr.bd.tcl

## ===================================================================================
## Create a wrapper for block design. Set the block design as top-level wrapper.
## ===================================================================================
make_wrapper -files [get_files ${BUILD_DIR}/${PLATFORM_NAME}_vivado/${PLATFORM_NAME}.srcs/sources_1/bd/${PLATFORM_NAME}/${PLATFORM_NAME}.bd] -top
add_files -norecurse ${BUILD_DIR}/${PLATFORM_NAME}_vivado/${PLATFORM_NAME}.srcs/sources_1/bd/${PLATFORM_NAME}/hdl/${PLATFORM_NAME}_wrapper.v
update_compile_order -fileset sources_1

## ===================================================================================
## Set output type to hw_export
## ===================================================================================
set_property platform.default_output_type           "sd_card" [current_project]
# Help by explicitly categorizing intended platform
set_property platform.design_intent.server_managed  "false"   [current_project]
set_property platform.design_intent.external_host   "false"   [current_project]
set_property platform.design_intent.embedded        "true"    [current_project]
set_property platform.design_intent.datacenter      "false"   [current_project]
set_property platform.extensible                    "true"    [current_project]

## ===================================================================================
## Wrap up Vivado Platform Project
## ===================================================================================
update_compile_order
assign_bd_address
regenerate_bd_layout
validate_bd_design
import_files

## ===================================================================================
## Generate files necessary to support block design through design flow
## ===================================================================================
generate_target all [get_files ${BUILD_DIR}/${PLATFORM_NAME}_vivado/${PLATFORM_NAME}.srcs/sources_1/bd/${PLATFORM_NAME}/${PLATFORM_NAME}.bd]

variable pre_synth
set pre_synth ""

if { $argc > 1} {
  set pre_synth [lindex $argv 3]
}
#Pre_synth Platform Flow
if {$pre_synth} {
  set_property platform.platform_state "pre_synth" [current_project]
  write_hw_platform -force ${BUILD_DIR}/xsa_platform/${PLATFORM_NAME}.xsa
  validate_hw_platform ${BUILD_DIR}/xsa_platform/${PLATFORM_NAME}.xsa
} else {
  set_property generate_synth_checkpoint true [get_files -norecurse *.bd]
  ## ===================================================================================
  ## Full Synthesis and implementation
  ## ===================================================================================
  launch_runs -jobs 8 synth_1
  wait_on_run synth_1
  puts "Synthesis done!"

  #launch_runs impl_1 -to_step write_bitstream
  launch_runs -jobs 8 impl_1 -to_step write_device_image
  wait_on_run impl_1
  puts "Implementation done!"

  # ===================================================================================
  # Write the XSA for current design for use as a hardware platform
  # ===================================================================================
  open_run impl_1
  write_hw_platform -unified -include_bit -force ${BUILD_DIR}/xsa_platform/${PLATFORM_NAME}.xsa
  validate_hw_platform ${BUILD_DIR}/xsa_platform/${PLATFORM_NAME}.xsa
}

