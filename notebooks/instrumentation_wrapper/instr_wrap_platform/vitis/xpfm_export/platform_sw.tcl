#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: X11
#

set platform_name [lindex $argv 0]
puts "Creating platform: \"$platform_name\""

set xsa [lindex $argv 1]
puts "using xsa: \"$xsa\""

set output_path [lindex $argv 2]
puts "with output path: \"$output_path\""

set OUTPUT platform_repo

set SRC ./src/

platform create -name $platform_name -desc " A platform for demonstration purpose with a Standalone domain" -hw $xsa -out $output_path -no-boot-bsp

## Create the Standalone domain 
domain create -name standalone_domain -os standalone -proc CIPS_0_pspmc_0_psv_cortexa72_0

platform generate
