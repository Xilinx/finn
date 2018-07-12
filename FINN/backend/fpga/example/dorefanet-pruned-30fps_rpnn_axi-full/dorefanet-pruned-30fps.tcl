open_project hls-syn

add_files dorefanet-pruned-30fps.cpp -cflags "-std=c++11 -I$::env(RPNN_LIBRARY)"
add_files dorefanet-pruned-30fps-config.h

set_top opencldesign_wrapper

open_solution sol1
set_part {xcku115-flva1517-2-e}
config_interface -m_axi_addr64
create_clock -period 5 -name default
csynth_design
#export_design -format ip_catalog
exit
