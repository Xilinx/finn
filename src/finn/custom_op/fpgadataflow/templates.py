# template for single node execution
docompute_template = """
#include "cnpy.h"
#include "npy2apintstream.hpp"
#include <vector>
#include "bnn-library.h"
// includes for network parameters
$GLOBALS$

// defines for network parameters
$DEFINES$

int main(){

$STREAMDECLARATIONS$

$READNPYDATA$

$DOCOMPUTE$

$DATAOUTSTREAM$

$SAVEASCNPY$

}

"""

# templates for single node ip generation

# cpp file
ipgen_template = """
#include "bnn-library.h"
// includes for network parameters
$GLOBALS$

// defines for network parameters
$DEFINES$

$BLACKBOXFUNCTION$
{
$PRAGMAS$
$DOCOMPUTE$
}
"""

# tcl script
ipgentcl_template = """
set config_proj_name $PROJECTNAME$
puts "HLS project: $config_proj_name"
set config_hwsrcdir "$HWSRCDIR$"
puts "HW source dir: $config_hwsrcdir"
set config_proj_part "$FPGAPART$"

set config_bnnlibdir "$FINNHLSLIBDIR$"

set config_toplevelfxn "$TOPFXN$"
set config_clkperiod $CLKPERIOD$

open_project $config_proj_name
add_files $config_hwsrcdir/top_$TOPFXN$.cpp -cflags "-std=c++0x -I$config_bnnlibdir"

set_top $config_toplevelfxn
open_solution sol1
set_part $config_proj_part

config_interface -m_axi_addr64
config_rtl -auto_prefix

create_clock -period $config_clkperiod -name default
csynth_design
export_design -format ip_catalog
exit 0
"""
