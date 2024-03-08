create_project -force thresholding thresholding.vivado -part xcvc1902-vsva2197-2MP-e-S
set_property board_part xilinx.com:vck190:part0:2.2 [current_project]

read_verilog hdl/axilite_if.v
read_verilog -sv { hdl/thresholding.sv hdl/thresholding_axi.sv }

set simset [current_fileset -simset]
set_property -name xsim.simulate.log_all_signals -value true -objects $simset
set_property -name xsim.simulate.runtime -value all -objects $simset
add_files -fileset $simset { sim/thresholding_tb.sv sim/thresholding_axi_tb.sv }

foreach top { thresholding_tb thresholding_axi_tb } {
	set_property top $top $simset

	launch_simulation
	close_sim
}
