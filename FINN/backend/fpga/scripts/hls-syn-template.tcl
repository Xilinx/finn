puts "HLS project: $config_proj_name"
puts "HW source dir: $config_hwsrcdir"
puts "FINN library: $config_bnnlibdir"
puts "Part: $config_proj_part"
puts "Clock period: $config_clkperiod ns"

# set up project
open_project $config_proj_name
add_files $config_hwsrcdir/docompute.cpp -cflags "-std=c++0x -I$config_bnnlibdir"
add_files $config_bnnlibdir/mlbptop.cpp -cflags "-std=c++0x -I$config_bnnlibdir -I$config_hwsrcdir"
set_top $config_toplevelfxn
open_solution sol1
set_part $config_proj_part
config_compile -name_max_length 300

# use 64-bit AXI MM addresses
config_interface -m_axi_addr64

# syntesize and export
create_clock -period $config_clkperiod -name default
csynth_design
export_design -format ip_catalog
exit 0
