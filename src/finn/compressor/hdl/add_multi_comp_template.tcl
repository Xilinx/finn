# Vivado batch flow for standalone add_multi compressor test.
# Behavioral simulation only — verifies the generated compressor produces correct sums.
#
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

launch_simulation
close_sim

quit
