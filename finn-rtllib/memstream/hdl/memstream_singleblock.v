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

/*
    Implements a lightweight streamer for up to 2 streams in a single block of memory
*/

module memstream_singleblock
#(
    parameter CONFIG_EN = 1,
    parameter NSTREAMS = 2,//1 up to 2

    parameter MEM_DEPTH = 512,
    parameter MEM_WIDTH = 32,
    parameter MEM_INIT = "./",
    parameter RAM_STYLE = "auto",

    //widths per stream
	parameter STRM0_WIDTH = 32,
	parameter STRM1_WIDTH = 32,

	//depths per stream
	parameter STRM0_DEPTH = 256,
	parameter STRM1_DEPTH = 256,

	//offsets for each stream
	parameter STRM0_OFFSET = 0,
	parameter STRM1_OFFSET = 256
)

(
    input aclk,
    input aresetn,

    //optional configuration interface compatible with ap_memory
	input [31:0] config_address,
	input config_ce,
	input config_we,
	input [MEM_WIDTH-1:0] config_d0,
	output [MEM_WIDTH-1:0] config_q0,
    output config_rack,

    //multiple output AXI Streams, TDATA width rounded to multiple of 8 bits
    input m_axis_0_tready,
    output m_axis_0_tvalid,
    output [((STRM0_WIDTH+7)/8)*8-1:0] m_axis_0_tdata,

    input m_axis_1_tready,
    output m_axis_1_tvalid,
    output [((STRM1_WIDTH+7)/8)*8-1:0] m_axis_1_tdata

);


//TODO: check that memory width is equal to the widest stream
//TODO: check that the stream depths and offsets make sense, and that the memory depth is sufficient (or calculate depth here?)
initial begin
    if((NSTREAMS < 1) | (NSTREAMS > 2)) begin
        $display("Invalid setting for NSTREAMS, please set in range [1,2]");
        $finish();
    end
end

//invert reset
wire rst;
assign rst = ~aresetn;

wire strm0_incr_en;
wire strm1_incr_en;

assign strm0_incr_en = m_axis_0_tready | ~m_axis_0_tvalid;
assign strm1_incr_en = m_axis_1_tready | ~m_axis_1_tvalid;

reg rack_shift[1:0];

generate
if(MEM_DEPTH > 1) begin: use_ram

//calculate width of memory address, with a minimum of 1 bit
localparam BLOCKADRWIDTH = $clog2(MEM_DEPTH);

reg [BLOCKADRWIDTH-1:0] strm0_addr = STRM0_OFFSET;
wire strm0_rst;
assign strm0_rst = strm0_incr_en & (strm0_addr == (STRM0_OFFSET + STRM0_DEPTH-1));

//one address counter per stream; more LUTs but keeps routing short and local
always @(posedge aclk) begin
    if(strm0_rst | rst)
        strm0_addr <= STRM0_OFFSET;
    else if(strm0_incr_en)
        strm0_addr <= strm0_addr + 1;
end

if(NSTREAMS == 1) begin: sdp

ramb18_sdp
#(
    .ID(0),
	.DWIDTH(MEM_WIDTH),
	.AWIDTH(BLOCKADRWIDTH),
    .DEPTH(MEM_DEPTH),
	.MEM_INIT(MEM_INIT),
    .RAM_STYLE(RAM_STYLE)
)
ram
(
	.clk(aclk),

    .ena(config_ce),
	.wea(config_we),
	.addra(config_address[BLOCKADRWIDTH-1:0]),
    .wdataa(config_d0),

    .enb(strm0_incr_en | config_ce),
    .enqb(strm0_incr_en | rack_shift[0]),
	.addrb(config_ce ? config_address[BLOCKADRWIDTH-1:0] : strm0_addr),
	.rdqb(m_axis_0_tdata)
);


end else begin: tdp

reg [BLOCKADRWIDTH-1:0] strm1_addr = STRM1_OFFSET;
wire strm1_rst;
assign strm1_rst = strm1_incr_en & (strm1_addr == (STRM1_OFFSET + STRM1_DEPTH-1));

always @(posedge aclk) begin
    if(strm1_rst | rst)
        strm1_addr <= STRM1_OFFSET;
    else if(strm1_incr_en)
        strm1_addr <= strm1_addr + 1;
end

ramb18_wf_dualport
#(
    .ID(0),
	.DWIDTH(MEM_WIDTH),
	.AWIDTH(BLOCKADRWIDTH),
    .DEPTH(MEM_DEPTH),
	.MEM_INIT(MEM_INIT),
    .RAM_STYLE(RAM_STYLE)
)
ram
(
	.clk(aclk),

	.wea(config_we),
    .ena(strm0_incr_en | config_ce),
    .enqa(strm0_incr_en | config_ce_r),
	.addra(config_we ? config_address[BLOCKADRWIDTH-1:0] : strm0_addr),
	.wdataa(config_d0),
	.rdqa(m_axis_0_tdata),

	.web(1'b0),
    .enb(strm1_incr_en),
    .enqb(strm1_incr_en),
	.addrb(strm1_addr),
	.wdatab('d0),
	.rdqb(m_axis_1_tdata)
);

end

end else begin: bypass

reg [MEM_WIDTH-1:0] singleval[0:0];
initial begin
    `ifdef SYNTHESIS
        $readmemh({MEM_INIT,"memblock_synth_0.dat"}, singleval, 0, 0);
    `else
        $readmemh({MEM_INIT,"memblock_sim_0.dat"}, singleval, 0, 0);
    `endif
end

always @(posedge aclk)
    if(config_ce & config_we)
        singleval[0] <= config_d0;

assign m_axis_0_tdata = singleval[0];
assign m_axis_1_tdata = singleval[0];

end
endgenerate

//signal valid after 2 tready cycles after initialization
//then stay valid
reg [1:0] tvalid_pipe0 = 2'd0;
reg [1:0] tvalid_pipe1 = 2'd0;

assign m_axis_0_tvalid = tvalid_pipe0[1];
assign m_axis_1_tvalid = tvalid_pipe1[1];

always @(posedge aclk) begin
    if(rst) begin
        tvalid_pipe0 <= 0;
    end else if(strm0_incr_en) begin
        tvalid_pipe0[0] <= 1;
        tvalid_pipe0[1] <= tvalid_pipe0[0];
    end
end

always @(posedge aclk) begin
    if(rst) begin
        tvalid_pipe1 <= 0;
    end else if(strm1_incr_en) begin
        tvalid_pipe1[0] <= 1;
        tvalid_pipe1[1] <= tvalid_pipe1[0];
    end
end

always @(posedge aclk) begin
    rack_shift[0] <= config_ce & ~config_we;
    rack_shift[1] <= rack_shift[0];
end

assign config_rack = rack_shift[1];
assign config_q0 = m_axis_0_tdata;

endmodule
