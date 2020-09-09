/*
 Copyright (c) 2020, Xilinx
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name of FINN nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

module memstream
#(
//parameters to enable/disable axi-mm, set number of streams, set readmemh for memory, set per-stream offsets in memory, set per-stream widths
    parameter CONFIG_EN = 1,
    parameter NSTREAMS = 6,//1 up to 6

    parameter MEM_DEPTH = 13824,
    parameter MEM_WIDTH = 32,
    parameter MEM_INIT = "./",
    parameter RAM_STYLE = "auto",

    //widths per stream
	parameter STRM0_WIDTH = 32,
	parameter STRM1_WIDTH = 32,
	parameter STRM2_WIDTH = 32,
	parameter STRM3_WIDTH = 32,
	parameter STRM4_WIDTH = 32,
	parameter STRM5_WIDTH = 32,

	//depths per stream
	parameter STRM0_DEPTH = 2304,
	parameter STRM1_DEPTH = 2304,
	parameter STRM2_DEPTH = 2304,
	parameter STRM3_DEPTH = 2304,
	parameter STRM4_DEPTH = 2304,
	parameter STRM5_DEPTH = 2304,

	//offsets for each stream
	parameter STRM0_OFFSET = 0,
	parameter STRM1_OFFSET = 2304,
	parameter STRM2_OFFSET = 4608,
	parameter STRM3_OFFSET = 6912,
	parameter STRM4_OFFSET = 9216,
	parameter STRM5_OFFSET = 11520
)

(
    input aclk,
    input aresetn,

    //optional configuration interface compatible with ap_memory
	input [31:0] config_address,
	input config_ce,
	input config_we,
	input [31:0] config_d0,
	output [31:0] config_q0,

    //multiple output AXI Streams, TDATA width rounded to multiple of 8 bits
    input m_axis_0_afull,
    input m_axis_0_tready,
    output m_axis_0_tvalid,
    output [((STRM0_WIDTH+7)/8)*8-1:0] m_axis_0_tdata,

    input m_axis_1_afull,
    input m_axis_1_tready,
    output m_axis_1_tvalid,
    output [((STRM1_WIDTH+7)/8)*8-1:0] m_axis_1_tdata,

    input m_axis_2_afull,
    input m_axis_2_tready,
    output m_axis_2_tvalid,
    output [((STRM2_WIDTH+7)/8)*8-1:0] m_axis_2_tdata,

    input m_axis_3_afull,
    input m_axis_3_tready,
    output m_axis_3_tvalid,
    output [((STRM3_WIDTH+7)/8)*8-1:0] m_axis_3_tdata,

    input m_axis_4_afull,
    input m_axis_4_tready,
    output m_axis_4_tvalid,
    output [((STRM4_WIDTH+7)/8)*8-1:0] m_axis_4_tdata,

    input m_axis_5_afull,
    input m_axis_5_tready,
    output m_axis_5_tvalid,
    output [((STRM5_WIDTH+7)/8)*8-1:0] m_axis_5_tdata


);

generate
if(NSTREAMS <= 2) begin: singleblock


memstream_singleblock
#(
    .CONFIG_EN(CONFIG_EN),
    .NSTREAMS(NSTREAMS),
    .MEM_DEPTH(MEM_DEPTH),
    .MEM_WIDTH(MEM_WIDTH),
    .MEM_INIT(MEM_INIT),
    .RAM_STYLE(RAM_STYLE),

    //widths per stream
    .STRM0_WIDTH(STRM0_WIDTH),
    .STRM1_WIDTH(STRM1_WIDTH),

    //depths per stream
    .STRM0_DEPTH(STRM0_DEPTH),
    .STRM1_DEPTH(STRM1_DEPTH),

    //offsets for each stream
    .STRM0_OFFSET(STRM0_OFFSET),
    .STRM1_OFFSET(STRM1_OFFSET)
)
mem
(
    .aclk(aclk),
    .aresetn(aresetn),

    .config_address(config_address),
    .config_ce(config_ce),
    .config_we(config_we),
    .config_d0(config_d0),
    .config_q0(config_q0),

    .m_axis_0_tready(m_axis_0_tready),
    .m_axis_0_tvalid(m_axis_0_tvalid),
    .m_axis_0_tdata(m_axis_0_tdata),

    .m_axis_1_tready(m_axis_1_tready),
    .m_axis_1_tvalid(m_axis_1_tvalid),
    .m_axis_1_tdata(m_axis_1_tdata)
);

assign m_axis_2_tvalid = 0;
assign m_axis_2_tdata = 0;
assign m_axis_3_tvalid = 0;
assign m_axis_3_tdata = 0;
assign m_axis_4_tvalid = 0;
assign m_axis_4_tdata = 0;
assign m_axis_5_tvalid = 0;
assign m_axis_5_tdata = 0;

end else begin: multiblock


memstream_multiblock
#(
    .CONFIG_EN(CONFIG_EN),
    .NSTREAMS(NSTREAMS),
    .MEM_DEPTH(MEM_DEPTH),
    .MEM_WIDTH(MEM_WIDTH),
    .MEM_INIT(MEM_INIT),
    .RAM_STYLE(RAM_STYLE),

    //widths per stream
    .STRM0_WIDTH(STRM0_WIDTH),
    .STRM1_WIDTH(STRM1_WIDTH),
    .STRM2_WIDTH(STRM2_WIDTH),
    .STRM3_WIDTH(STRM3_WIDTH),
    .STRM4_WIDTH(STRM4_WIDTH),
    .STRM5_WIDTH(STRM5_WIDTH),

    //depths per stream
    .STRM0_DEPTH(STRM0_DEPTH),
    .STRM1_DEPTH(STRM1_DEPTH),
    .STRM2_DEPTH(STRM2_DEPTH),
    .STRM3_DEPTH(STRM3_DEPTH),
    .STRM4_DEPTH(STRM4_DEPTH),
    .STRM5_DEPTH(STRM5_DEPTH),

    //offsets for each stream
    .STRM0_OFFSET(STRM0_OFFSET),
    .STRM1_OFFSET(STRM1_OFFSET),
    .STRM2_OFFSET(STRM2_OFFSET),
    .STRM3_OFFSET(STRM3_OFFSET),
    .STRM4_OFFSET(STRM4_OFFSET),
    .STRM5_OFFSET(STRM5_OFFSET)
)
mem
(
    .aclk(aclk),
    .aresetn(aresetn),

    .config_address(config_address),
    .config_ce(config_ce),
    .config_we(config_we),
    .config_d0(config_d0),
    .config_q0(config_q0),

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


end
endgenerate

endmodule
