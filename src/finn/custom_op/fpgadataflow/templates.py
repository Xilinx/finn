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

# tcl script for IP generation
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

config_compile -ignore_long_run_time -disable_unroll_code_size_check
config_interface -m_axi_addr64
config_rtl -auto_prefix
$EXTRA_DIRECTIVES$

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

//depths per stream
.STRM0_DEPTH($WSTREAM_DEPTH$),

//offsets for each stream
.STRM0_OFFSET(0)
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
.m_axis_0_tdata(m_axis_0_tdata)


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
.weights_V_V_TDATA(m_axis_0_tdata),	//$WEIGHT_RANGE$ input
.weights_V_V_TVALID(m_axis_0_tvalid),	//input
.weights_V_V_TREADY(m_axis_0_tready),	//output
.out_V_V_TDATA(out_V_V_TDATA),		//$OUT_RANGE$ output
.out_V_V_TVALID(out_V_V_TVALID),	//output
.out_V_V_TREADY(out_V_V_TREADY)		//input
);

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
  virtexuplusHBM Production \
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

strm_fifo_wrapper = """
module $TOPNAME$(
ap_clk,
ap_rst_n,
count,
in0_V_V_TDATA,
in0_V_V_TVALID,
in0_V_V_TREADY,
out_V_V_TDATA,
out_V_V_TVALID,
out_V_V_TREADY
);

input   ap_clk;
input   ap_rst_n;
output $COUNT_RANGE$ count;
input  $IN_RANGE$ in0_V_V_TDATA;
input   in0_V_V_TVALID;
output   in0_V_V_TREADY;
output  $OUT_RANGE$ out_V_V_TDATA;
output   out_V_V_TVALID;
input   out_V_V_TREADY;

Q_srl #(
.depth($DEPTH$),
.width($WIDTH$)
)
$LAYER_NAME$
(
 .clock(ap_clk),
 .reset(!ap_rst_n),
 .count(count),
 .i_d(in0_V_V_TDATA),
 .i_v(in0_V_V_TVALID),
 .i_r(in0_V_V_TREADY),
 .o_d(out_V_V_TDATA),
 .o_v(out_V_V_TVALID),
 .o_r(out_V_V_TREADY)
);

endmodule
"""

decoupled_thresholding_template = """
template <
    unsigned ImgDim, unsigned NumChannels, unsigned PE,
    typename TSrcI = Identity, typename TDstI = Identity,
    int ActVal=0, typename TT, unsigned int NumSteps,
    typename TI, typename TO>
void Thresholding_Stream_Batch(hls::stream<TI> &in,
                        hls::stream<TO> &out,
                        hls::stream<ap_uint<PE*NumSteps*TT::width>> &weight,
                        int const reps)
{

  // how many different rows each neuron will compute
  // alternatively: number of vertical matrix chunks
  unsigned const NF = NumChannels / PE;

  unsigned nf = 0;
  unsigned tile = 0; // invariant: tile = nf*SF + sf

  ThresholdsActivation<1, PE, NumSteps, TT, TO, ActVal> internal_thr;
  #pragma HLS ARRAY_PARTITION variable=internal_thr.m_thresholds complete dim=0

  // everything merged into a common iteration space (one "big" loop instead
  // of smaller nested loops) to get the pipelinening the way we want
  for (unsigned i = 0; i < reps * ImgDim * ImgDim * NF; i++)
  {
    #pragma HLS PIPELINE II=1

    ap_uint<PE*NumSteps*TT::width> packed_thr;
    packed_thr = weight.read();
    auto const packed_thr_slicer = Slice<TT>()(packed_thr);

    TI inElem;
    inElem = in.read();
    auto outElem = TDstI().template operator()<TO>();

    for (unsigned pe = 0; pe < PE; pe++)
    {
#pragma HLS UNROLL
      for (unsigned nt = 0; nt < NumSteps; nt++)
      {
        internal_thr.m_thresholds[pe][0][nt] = packed_thr_slicer(nt, pe);
      }

      auto const act = TSrcI()(inElem);
      outElem(pe,0,1) = internal_thr.activate(0, pe, act(pe,0));
    }
    out.write(outElem);
    if (++nf == NF)
    {
      nf = 0;
    }
  }
}
"""
