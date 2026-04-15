# Create Fresh Project
set sig {n}x{sa}{na}{sb}{nb}
set top dotp_$sig
set part {part}
create_project -force $top $top.vivado -part $part

# Import Design and Simulation Sources
read_verilog -sv hdl/mul_comp_map.sv gen/comp_$sig.sv gen/$top.sv
set simset [current_fileset -simset]
add_files -fileset $simset gen/${top}_tb.sv
set_property top ${top}_tb $simset
set_property xsim.simulate.runtime all $simset

# Run Simulation
launch_simulation
close_sim

quit
