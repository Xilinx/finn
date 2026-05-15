#############################################################################
# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @brief    Vivado simulation script for add_multi compressor testbench
# @author    Simon Gerber <simon.gerber@amd.com>
#############################################################################

# Template placeholders expanded by run_add_multi_comp_tests.sh:
#   {label}   - Configuration label (e.g. n8_w4_p2)
#   {tb}      - Testbench module name
#   {gen_dir} - Absolute path to gen/<label>/

set label {label}
set tb {tb}
set part {part}
create_project -force add_multi_comp_$label add_multi_comp_$label.vivado -part $part

# Design sources: only the generated compressor
read_verilog -sv {*}[glob {gen_dir}/comp_*.sv]

# Testbench
set simset [current_fileset -simset]
add_files -fileset $simset {gen_dir}/{tb}.sv
set_property top $tb $simset
set_property xsim.simulate.runtime all $simset

if {[catch {launch_simulation} err]} {
    puts "ERROR: Simulation failed: $err"
}
close_sim

quit
