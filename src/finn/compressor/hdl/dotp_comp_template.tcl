# Create Fresh Project
set label {label}
set src_dir {src_dir}
set tb dotp_comp_{label}_tb
set part {part}
create_project -force dotp_comp_$label dotp_comp_$label.vivado -part $part

# Import Design and Simulation Sources
# Static: mul_comp_map interface
# Expanded: dotp_comp.sv (template with $COMP_MODULE_NAME$ filled in)
# Generated: comp_<sig>.sv (config-specific compressor core)
read_verilog -sv $src_dir/hdl/mul_comp_map.sv $src_dir/gen/$label/dotp_comp.sv {*}[glob $src_dir/gen/$label/comp_*.sv]
set simset [current_fileset -simset]
add_files -fileset $simset $src_dir/gen/$label/$tb.sv
set_property file_type SystemVerilog [get_files -of_objects $simset $src_dir/gen/$label/$tb.sv]
set_property top $tb $simset
set_property xsim.simulate.runtime all $simset

# Run Simulation
launch_simulation
close_sim

quit
