# ignore the first 2 args, since Vivado HLS also passes -f tclname as args
set config_proj_name [lindex $argv 2]
puts "HLS project: $config_proj_name"
set config_hwsrcdir [lindex $argv 3]
puts "HW source dir: $config_hwsrcdir"
set config_bnnlibdir "$::env(XILINX_RPNN_ROOT)/hls"
puts "BNN library: $config_bnnlibdir"

set config_toplevelfxn "BlackBoxJam"
set config_proj_part "xc7z045ffg900-2"
set config_clkperiod 5

# set up project
open_project $config_proj_name
add_files $config_hwsrcdir/top.cpp -cflags "-std=c++0x -I$config_bnnlibdir"
set_top $config_toplevelfxn
open_solution sol1
set_part $config_proj_part

# use 64-bit AXI MM addresses
config_interface -m_axi_addr64

# syntesize and export
create_clock -period $config_clkperiod -name default
csynth_design
export_design -format ip_catalog
exit 0
