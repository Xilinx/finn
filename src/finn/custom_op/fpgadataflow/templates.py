# flake8: noqa
# Copyright (c) 2020, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# template for single node execution
docompute_template = """
#define AP_INT_MAX_W $AP_INT_MAX_W$
#include "cnpy.h"
#include "npy2apintstream.hpp"
#include <vector>
#include "bnn-library.h"

// includes for network parameters
$GLOBALS$

// defines for network parameters
$DEFINES$

int main(){
$PRAGMAS$

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
#define AP_INT_MAX_W $AP_INT_MAX_W$

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

# verilog wrapper for decoupled mem mode
decoupled_wrapper = """
module $TOPNAME$(
ap_clk,
ap_rst_n,
in0_V_V_TDATA,
in0_V_V_TVALID,
in0_V_V_TREADY,
out_V_V_TDATA,
out_V_V_TVALID,
out_V_V_TREADY
);

input   ap_clk;
input   ap_rst_n;
input  $IN_RANGE$ in0_V_V_TDATA;
input   in0_V_V_TVALID;
output   in0_V_V_TREADY;
output  $OUT_RANGE$ out_V_V_TDATA;
output   out_V_V_TVALID;
input   out_V_V_TREADY;

reg [31:0] config_address = 0;
reg config_ce = 0;
reg config_we = 0;
reg [31:0] config_d0 = 0;
wire [31:0] config_q0;

//multiple wire AXI Streams
wire m_axis_0_afull;
// FIFO count to generate programmable full
wire [5:0] fifo_0_count;
wire m_axis_0_tready;
wire m_axis_0_tvalid;
wire $WEIGHT_RANGE$ m_axis_0_tdata;

wire m_axis_0_tready_q;
wire m_axis_0_tvalid_q;
wire $WEIGHT_RANGE$ m_axis_0_tdata_q;

wire m_axis_0_tready_q2;
wire m_axis_0_tvalid_q2;
wire $WEIGHT_RANGE$ m_axis_0_tdata_q2;

reg m_axis_1_afull = 0;
reg m_axis_1_tready = 1;
wire m_axis_1_tvalid;
wire $WEIGHT_RANGE$ m_axis_1_tdata;

reg m_axis_2_afull = 0;
reg m_axis_2_tready = 1;
wire m_axis_2_tvalid;
wire $WEIGHT_RANGE$ m_axis_2_tdata;

reg m_axis_3_afull = 0;
reg m_axis_3_tready = 1;
wire m_axis_3_tvalid;
wire $WEIGHT_RANGE$ m_axis_3_tdata;

reg m_axis_4_afull = 0;
reg m_axis_4_tready = 1;
wire m_axis_4_tvalid;
wire $WEIGHT_RANGE$ m_axis_4_tdata;

reg m_axis_5_afull = 0;
reg m_axis_5_tready = 1;
wire m_axis_5_tvalid;
wire $WEIGHT_RANGE$ m_axis_5_tdata;

//memstream component

memstream
#(
//parameters to enable/disable axi-mm, set number of streams, set readmemh for
// memory, set per-stream offsets in memory, set per-stream widths
.CONFIG_EN(1),
.NSTREAMS(1),
.MEM_DEPTH($MEM_DEPTH$),
.MEM_WIDTH($WEIGHT_WIDTH$),
.MEM_INIT("./"),
.RAM_STYLE("$RAM_STYLE$"),

//widths per stream
.STRM0_WIDTH($WEIGHT_WIDTH$),
.STRM1_WIDTH($WEIGHT_WIDTH$),
.STRM2_WIDTH($WEIGHT_WIDTH$),
.STRM3_WIDTH($WEIGHT_WIDTH$),
.STRM4_WIDTH($WEIGHT_WIDTH$),
.STRM5_WIDTH($WEIGHT_WIDTH$),

//depths per stream
.STRM0_DEPTH($WSTREAM_DEPTH$),
.STRM1_DEPTH(1),
.STRM2_DEPTH(1),
.STRM3_DEPTH(1),
.STRM4_DEPTH(1),
.STRM5_DEPTH(1),

//offsets for each stream
.STRM0_OFFSET(0),
.STRM1_OFFSET(0),
.STRM2_OFFSET(0),
.STRM3_OFFSET(0),
.STRM4_OFFSET(0),
.STRM5_OFFSET(0)
)
mem
(
.aclk(ap_clk),
.aresetn(ap_rst_n),

//optional configuration interface compatible with ap_memory
.config_address(config_address),
.config_ce(config_ce),
.config_we(config_we),
.config_d0(config_d0),
.config_q0(config_q0),

//multiple output AXI Streams, TDATA width rounded to multiple of 8 bits
.m_axis_0_afull(m_axis_0_afull),
.m_axis_0_tready(m_axis_0_tready),
.m_axis_0_tvalid(m_axis_0_tvalid),
.m_axis_0_tdata(m_axis_0_tdata),

.m_axis_1_afull(m_axis_1_afull),
.m_axis_1_tready(m_axis_1_tready),
.m_axis_1_tvalid(m_axis_1_tvalid),
.m_axis_1_tdata(m_axis_1_tdata),

.m_axis_2_afull(m_axis_2_afull),
.m_axis_2_tready(m_axis_2_tready),
.m_axis_2_tvalid(m_axis_2_tvalid),
.m_axis_2_tdata(m_axis_2_tdata),

.m_axis_3_afull(m_axis_3_afull),
.m_axis_3_tready(m_axis_3_tready),
.m_axis_3_tvalid(m_axis_3_tvalid),
.m_axis_3_tdata(m_axis_3_tdata),

.m_axis_4_afull(m_axis_4_afull),
.m_axis_4_tready(m_axis_4_tready),
.m_axis_4_tvalid(m_axis_4_tvalid),
.m_axis_4_tdata(m_axis_4_tdata),

.m_axis_5_afull(m_axis_5_afull),
.m_axis_5_tready(m_axis_5_tready),
.m_axis_5_tvalid(m_axis_5_tvalid),
.m_axis_5_tdata(m_axis_5_tdata)


);


Q_srl #(
.depth(32),
.width($WEIGHT_WIDTH$)
)
$LAYER_NAME$_w_fifo_1
(
 .clock(ap_clk),
 .reset(!ap_rst_n),
 .i_d(m_axis_0_tdata),
 .i_v(m_axis_0_tvalid),
 .i_r(m_axis_0_tready),
 .o_d(m_axis_0_tdata_q),
 .o_v(m_axis_0_tvalid_q),
 .o_r(m_axis_0_tready_q),
 .count(fifo_0_count)
);


//MVA_Stream_Unit

$LAYER_NAME$
MVA_Stream_U
(
.ap_clk(ap_clk),			//input
.ap_rst_n(ap_rst_n), 			//input
.in0_V_V_TDATA(in0_V_V_TDATA),		//$IN_RANGE$ input
.in0_V_V_TVALID(in0_V_V_TVALID),  	//input
.in0_V_V_TREADY(in0_V_V_TREADY),	//output
.weights_V_V_TDATA(m_axis_0_tdata_q),	//$WEIGHT_RANGE$ input
.weights_V_V_TVALID(m_axis_0_tvalid_q),	//input
.weights_V_V_TREADY(m_axis_0_tready_q),	//output
.out_V_V_TDATA(out_V_V_TDATA),		//$OUT_RANGE$ output
.out_V_V_TVALID(out_V_V_TVALID),	//output
.out_V_V_TREADY(out_V_V_TREADY)		//input
);

// programmable full threshold at 16 elements
assign m_axis_0_afull = (fifo_0_count > 16);

endmodule
"""

ip_package_tcl = """
## IP Info
set Vendor      "xilinx.com"
set Library     "hls"
set IPName      "$TOPNAME$"
set Version     "1.0"
set DisplayName "$TOPNAME$"
set Description "An IP generated by Xilinx FINN"
set Device      "zynq"
set Catalog     "/UserIP"
set RootDir     "$VERILOG_DIR$"

## Variables
set Top "$TOPNAME$"
set VerilogFiles [glob -nocomplain $RootDir/*]


## Enter IP directory
cd [file dir [info script]]

## Generate sub cores
set IPs ""
set IPFiles ""

## Basic info
set core [ipx::create_core $Vendor $Library $IPName $Version]
set_property display_name $DisplayName $core
set_property description $Description $core
set_property taxonomy $Catalog $core
set_property supported_families { \
  artix7 Production \
  artix7l Production \
  kintex7 Production \
  kintex7l Production \
  kintexu Production \
  kintexuplus Production \
  virtex7 Production \
  virtexu Production \
  virtexuplus Production \
  zynq Production \
  zynquplus Production \
  aartix7 Production \
  azynq Production \
  qartix7 Production \
  qkintex7 Production \
  qkintex7l Production \
  qvirtex7 Production \
  qzynq Production \
} $core;

## Add verilog files
if {[llength $VerilogFiles] > 0} {
    # synthesis
    set group [ipx::add_file_group xilinx_verilogsynthesis $core]
    foreach f [concat $VerilogFiles $IPFiles] {
        set current_file [ipx::add_file $f $group]
        if {[file ext $f] == ".dat"} {
            set_property type "mif" $current_file
        }
    }
    set_property model_name $Top $group
    if {$IPs != ""} {
        set_property component_subcores $IPs $group
    }

    # simulation
    set group [ipx::add_file_group xilinx_verilogbehavioralsimulation $core]
    foreach f [concat $VerilogFiles $IPFiles] {
        set current_file [ipx::add_file $f $group]
        if {[file ext $f] == ".dat"} {
            set_property type "mif" $current_file
        }
    }
    set_property model_name $Top $group
    if {$IPs != ""} {
        set_property component_subcores $IPs $group
    }
}

## Import ports
ipx::add_ports_from_hdl \
    -top_level_hdl_file $RootDir/$Top.v \
    -top_module_name $Top \
    $core

## Infer interfaces
ipx::infer_bus_interface ap_clk xilinx.com:signal:clock_rtl:1.0 [ipx::current_core]
ipx::infer_bus_interface ap_rst_n xilinx.com:signal:reset_rtl:1.0 [ipx::current_core]
ipx::infer_bus_interface {in0_V_V_TDATA in0_V_V_TVALID in0_V_V_TREADY} xilinx.com:interface:axis_rtl:1.0 [ipx::current_core]
ipx::infer_bus_interface {out_V_V_TREADY out_V_V_TDATA out_V_V_TVALID} xilinx.com:interface:axis_rtl:1.0 [ipx::current_core]
ipx::associate_bus_interfaces -busif in0_V_V -clock ap_clk [ipx::current_core]
ipx::associate_bus_interfaces -busif out_V_V -clock ap_clk [ipx::current_core]

## Finalize
set_property core_revision 2 [ipx::current_core]
ipx::create_xgui_files [ipx::current_core]
ipx::update_checksums [ipx::current_core]
ipx::save_core [ipx::current_core]
"""
