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

//calculate number of RAMB18 blocks we need depth-wise
localparam NMEMBLOCKS = (MEM_DEPTH+1023) / 1024; //ceil(MEM_DEPTH/1024)

//calculate width of address for each block
localparam BLOCKADRWIDTH = NMEMBLOCKS > 1 ? 10 : $clog2(MEM_DEPTH);

//determine whether a stream needs to multiplex between memory blocks
localparam STRM0_MUX = ((STRM0_OFFSET/1024) != ((STRM0_OFFSET+STRM0_DEPTH)/1024));
localparam STRM1_MUX = ((STRM1_OFFSET/1024) != ((STRM1_OFFSET+STRM1_DEPTH)/1024));
localparam STRM2_MUX = ((STRM2_OFFSET/1024) != ((STRM2_OFFSET+STRM2_DEPTH)/1024));
localparam STRM3_MUX = ((STRM3_OFFSET/1024) != ((STRM3_OFFSET+STRM3_DEPTH)/1024));
localparam STRM4_MUX = ((STRM4_OFFSET/1024) != ((STRM4_OFFSET+STRM4_DEPTH)/1024));
localparam STRM5_MUX = ((STRM5_OFFSET/1024) != ((STRM5_OFFSET+STRM5_DEPTH)/1024));

//determine what the base block of each stream is
localparam STRM0_BLOCK = (STRM0_OFFSET/1024);
localparam STRM1_BLOCK = (STRM1_OFFSET/1024);
localparam STRM2_BLOCK = (STRM2_OFFSET/1024);
localparam STRM3_BLOCK = (STRM3_OFFSET/1024);
localparam STRM4_BLOCK = (STRM4_OFFSET/1024);
localparam STRM5_BLOCK = (STRM5_OFFSET/1024);

//determine what the end block of each stream is
localparam STRM0_END_BLOCK = ((STRM0_OFFSET+STRM0_DEPTH-1)/1024);
localparam STRM1_END_BLOCK = ((STRM1_OFFSET+STRM1_DEPTH-1)/1024);
localparam STRM2_END_BLOCK = ((STRM2_OFFSET+STRM2_DEPTH-1)/1024);
localparam STRM3_END_BLOCK = ((STRM3_OFFSET+STRM3_DEPTH-1)/1024);
localparam STRM4_END_BLOCK = ((STRM4_OFFSET+STRM4_DEPTH-1)/1024);
localparam STRM5_END_BLOCK = ((STRM5_OFFSET+STRM5_DEPTH-1)/1024);

//determine the number of blocks spanned by each stream
localparam STRM0_NBLOCKS = STRM0_END_BLOCK - STRM0_BLOCK + 1;
localparam STRM1_NBLOCKS = STRM1_END_BLOCK - STRM1_BLOCK + 1;
localparam STRM2_NBLOCKS = STRM2_END_BLOCK - STRM2_BLOCK + 1;
localparam STRM3_NBLOCKS = STRM3_END_BLOCK - STRM3_BLOCK + 1;
localparam STRM4_NBLOCKS = STRM4_END_BLOCK - STRM4_BLOCK + 1;
localparam STRM5_NBLOCKS = STRM5_END_BLOCK - STRM5_BLOCK + 1;

//TODO: check that memory width is equal to the widest stream
//TODO: check that the stream depths and offsets make sense, and that the memory depth is sufficient (or calculate depth here?)
initial begin
    if((NSTREAMS < 1) | (NSTREAMS > 6)) begin
        $display("Invalid setting for NSTREAMS, please set in range [1,6]");
        $finish();
    end
end

//invert reset
wire rst;
assign rst = ~aresetn;

//WARNING: pipeline depth is larger than the number of streams per port so we have in-flight writes that may see not-ready when they get executed
//solution: use prog-full to make sure we have an equal number of free slots in the stream to the read pipeline depth

reg [$clog2(MEM_DEPTH)-1:0] strm0_addr = STRM0_OFFSET;
reg [$clog2(MEM_DEPTH)-1:0] strm1_addr = STRM1_OFFSET;
reg [$clog2(MEM_DEPTH)-1:0] strm2_addr = STRM2_OFFSET;
reg [$clog2(MEM_DEPTH)-1:0] strm3_addr = STRM3_OFFSET;
reg [$clog2(MEM_DEPTH)-1:0] strm4_addr = STRM4_OFFSET;
reg [$clog2(MEM_DEPTH)-1:0] strm5_addr = STRM5_OFFSET;

reg strm0_incr_en;
reg strm1_incr_en;
reg strm2_incr_en;
reg strm3_incr_en;
reg strm4_incr_en;
reg strm5_incr_en;

wire strm0_rst;
wire strm1_rst;
wire strm2_rst;
wire strm3_rst;
wire strm4_rst;
wire strm5_rst;

reg strm0_ready;
reg strm1_ready;
reg strm2_ready;
reg strm3_ready;
reg strm4_ready;
reg strm5_ready;

//arbiter: work on one stream at a time
//multiplex each port between (up to) half of the streams 
reg [1:0] current_stream_porta = 0;
reg [1:0] current_stream_portb = 0;

always @(posedge aclk) begin
    if(rst)
        current_stream_porta <= 0;
    else case(current_stream_porta)
        0: current_stream_porta <= strm2_ready ? 1 : strm4_ready ? 2 : 0;
        1: current_stream_porta <= strm4_ready ? 2 : strm0_ready ? 0 : 1;
        2: current_stream_porta <= strm0_ready ? 0 : strm2_ready ? 1 : 2;
    endcase
    if(rst)
        current_stream_portb <= 0;
    else case(current_stream_portb)
        0: current_stream_portb <= strm3_ready ? 1 : strm5_ready ? 2 : 0;
        1: current_stream_portb <= strm5_ready ? 2 : strm1_ready ? 0 : 1;
        2: current_stream_portb <= strm1_ready ? 0 : strm3_ready ? 1 : 2;
    endcase
end

always @(posedge aclk) begin
    if(rst) begin
        strm0_incr_en <= 0;
        strm1_incr_en <= 0;
        strm2_incr_en <= 0;
        strm3_incr_en <= 0;
        strm4_incr_en <= 0;
        strm5_incr_en <= 0;
    end else begin
        strm0_incr_en <= (current_stream_porta == 0) & strm0_ready;
        strm1_incr_en <= (current_stream_portb == 0) & strm1_ready;
        strm2_incr_en <= (current_stream_porta == 1) & strm2_ready;
        strm3_incr_en <= (current_stream_portb == 1) & strm3_ready;
        strm4_incr_en <= (current_stream_porta == 2) & strm4_ready;
        strm5_incr_en <= (current_stream_portb == 2) & strm5_ready;
    end
end

assign strm0_rst = strm0_incr_en & (strm0_addr == (STRM0_OFFSET + STRM0_DEPTH-1));
assign strm1_rst = strm1_incr_en & (strm1_addr == (STRM1_OFFSET + STRM1_DEPTH-1));
assign strm2_rst = strm2_incr_en & (strm2_addr == (STRM2_OFFSET + STRM2_DEPTH-1));
assign strm3_rst = strm3_incr_en & (strm3_addr == (STRM3_OFFSET + STRM3_DEPTH-1));
assign strm4_rst = strm4_incr_en & (strm4_addr == (STRM4_OFFSET + STRM4_DEPTH-1));
assign strm5_rst = strm5_incr_en & (strm5_addr == (STRM5_OFFSET + STRM5_DEPTH-1));

always @(posedge aclk) begin
    strm0_ready <= ~m_axis_0_afull;
    strm1_ready <= ~m_axis_1_afull & (NSTREAMS >= 2);
    strm2_ready <= ~m_axis_2_afull & (NSTREAMS >= 3);
    strm3_ready <= ~m_axis_3_afull & (NSTREAMS >= 4);
    strm4_ready <= ~m_axis_4_afull & (NSTREAMS >= 5);
    strm5_ready <= ~m_axis_5_afull & (NSTREAMS >= 6);
end

//one address counter per stream; more LUTs but keeps routing short and local
always @(posedge aclk) begin
    if(strm0_rst | rst)
        strm0_addr <= STRM0_OFFSET;
    else if(strm0_incr_en)
        strm0_addr <= strm0_addr + 1;
    if(strm1_rst | rst)
        strm1_addr <= STRM1_OFFSET;
    else if(strm1_incr_en)
        strm1_addr <= strm1_addr + 1;
    if(strm2_rst | rst)
        strm2_addr <= STRM2_OFFSET;
    else if(strm2_incr_en)
        strm2_addr <= strm2_addr + 1;
    if(strm3_rst | rst)
        strm3_addr <= STRM3_OFFSET;
    else if(strm3_incr_en)
        strm3_addr <= strm3_addr + 1;
    if(strm4_rst | rst)
        strm4_addr <= STRM4_OFFSET;
    else if(strm4_incr_en)
        strm4_addr <= strm4_addr + 1;
    if(strm5_rst | rst)
        strm5_addr <= STRM5_OFFSET;
    else if(strm5_incr_en)
        strm5_addr <= strm5_addr + 1;
end

reg [$clog2(MEM_DEPTH)-1:0] addra;
wire [MEM_WIDTH*NMEMBLOCKS-1:0] rdqa;

reg [$clog2(MEM_DEPTH)-1:0] addrb;
wire [MEM_WIDTH*NMEMBLOCKS-1:0] rdqb;

wire [NMEMBLOCKS-1:0] we;

reg [1:0] addr_select_porta;
reg [1:0] addr_select_portb;

//multiplex addresses of various streams into address ports of memory
always @(posedge aclk) begin
    addr_select_porta <= current_stream_porta;
    case(addr_select_porta)
        0: addra <= strm0_addr;
        1: addra <= strm2_addr;
        2: addra <= strm4_addr;
    endcase
    addr_select_portb <= current_stream_portb;
    case(addr_select_portb)
        0: addrb <= strm1_addr;
        1: addrb <= strm3_addr;
        2: addrb <= strm5_addr;
    endcase
end

genvar g;
generate for(g=0; g<NMEMBLOCKS; g=g+1) begin: blockports

assign we[g] = (CONFIG_EN == 1) & config_ce & config_we & (config_address[31:BLOCKADRWIDTH] == g);

ramb18_wf_dualport
#(
    .ID(g),
	.DWIDTH(MEM_WIDTH),
	.AWIDTH(BLOCKADRWIDTH),
	.MEM_INIT(MEM_INIT)
)
ram
(
	.clk(aclk),
	
	.wea(we[g]),
	.addra(we[g] ? config_address[BLOCKADRWIDTH-1:0] : addra[BLOCKADRWIDTH-1:0]),
	.wdataa(config_d0),
	.rdqa(rdqa[(g+1)*MEM_WIDTH-1:g*MEM_WIDTH]),

	.web(1'b0),
	.addrb(addrb[BLOCKADRWIDTH-1:0]),
	.wdatab('d0),
	.rdqb(rdqb[(g+1)*MEM_WIDTH-1:g*MEM_WIDTH])
);

end
endgenerate

integer i;

generate if(NMEMBLOCKS > 1) begin: multiblock

wire [MEM_WIDTH-1:0] rdqmux[5:0];

reg [$clog2(MEM_DEPTH)-BLOCKADRWIDTH-1:0] rdblocka[2:0];
reg [$clog2(MEM_DEPTH)-BLOCKADRWIDTH-1:0] rdblockb[2:0];

always @(posedge aclk) begin
    rdblocka[0] <= addra[$clog2(MEM_DEPTH)-1:BLOCKADRWIDTH];
    rdblockb[0] <= addrb[$clog2(MEM_DEPTH)-1:BLOCKADRWIDTH];
    for(i=0; i<2; i=i+1) begin
		rdblocka[i+1] <= rdblocka[i];
		rdblockb[i+1] <= rdblockb[i];
    end
end

if(NSTREAMS >= 1) begin: en_strm0
	if(STRM0_MUX == 1) begin: mux0
		mux #(STRM0_NBLOCKS, MEM_WIDTH) m(rdqa[(STRM0_BLOCK+STRM0_NBLOCKS)*MEM_WIDTH-1:STRM0_BLOCK*MEM_WIDTH],rdqmux[0],rdblocka[1] - STRM0_BLOCK);
	end else begin: nomux0
		assign rdqmux[0] = rdqa[(STRM0_BLOCK+1)*MEM_WIDTH-1:STRM0_BLOCK*MEM_WIDTH];
	end
	assign m_axis_0_tdata = rdqmux[0][STRM0_WIDTH-1:0];
end

if(NSTREAMS >= 2) begin: en_strm1
	if(STRM1_MUX == 1) begin: mux1
		mux #(STRM1_NBLOCKS, MEM_WIDTH) m(rdqb[(STRM1_BLOCK+STRM1_NBLOCKS)*MEM_WIDTH-1:STRM1_BLOCK*MEM_WIDTH],rdqmux[1],rdblockb[1] - STRM1_BLOCK);
	end else begin: nomux1
		assign rdqmux[1] = rdqb[(STRM1_BLOCK+1)*MEM_WIDTH-1:STRM1_BLOCK*MEM_WIDTH];
	end
	assign m_axis_1_tdata = rdqmux[1][STRM1_WIDTH-1:0];
end

if(NSTREAMS >= 3) begin: en_strm2
	if(STRM2_MUX == 1) begin: mux2
		mux #(STRM2_NBLOCKS, MEM_WIDTH) m(rdqa[(STRM2_BLOCK+STRM2_NBLOCKS)*MEM_WIDTH-1:STRM2_BLOCK*MEM_WIDTH],rdqmux[2],rdblocka[1] - STRM2_BLOCK);
	end else begin: nomux2
		assign rdqmux[2] = rdqa[(STRM2_BLOCK+1)*MEM_WIDTH-1:STRM2_BLOCK*MEM_WIDTH];
	end
	assign m_axis_2_tdata = rdqmux[2][STRM2_WIDTH-1:0];
end

if(NSTREAMS >= 4) begin: en_strm3
	if(STRM3_MUX == 1) begin: mux3
		mux #(STRM3_NBLOCKS, MEM_WIDTH) m(rdqb[(STRM3_BLOCK+STRM3_NBLOCKS)*MEM_WIDTH-1:STRM3_BLOCK*MEM_WIDTH],rdqmux[3],rdblockb[1] - STRM3_BLOCK);
	end else begin: nomux3
		assign rdqmux[3] = rdqb[(STRM3_BLOCK+1)*MEM_WIDTH-1:STRM3_BLOCK*MEM_WIDTH];
	end
	assign m_axis_3_tdata = rdqmux[3][STRM3_WIDTH-1:0];
end

if(NSTREAMS >= 5) begin: en_strm4
	if(STRM4_MUX == 1) begin: mux4
		mux #(STRM4_NBLOCKS, MEM_WIDTH) m(rdqa[(STRM4_BLOCK+STRM4_NBLOCKS)*MEM_WIDTH-1:STRM4_BLOCK*MEM_WIDTH],rdqmux[4],rdblocka[1] - STRM4_BLOCK);
	end else begin: nomux4
		assign rdqmux[4] = rdqa[(STRM4_BLOCK+1)*MEM_WIDTH-1:STRM4_BLOCK*MEM_WIDTH];
	end
	assign m_axis_4_tdata = rdqmux[4][STRM4_WIDTH-1:0];
end

if(NSTREAMS >= 6) begin: en_strm5
	if(STRM5_MUX == 1) begin: mux5
		mux #(STRM5_NBLOCKS, MEM_WIDTH) m(rdqb[(STRM5_BLOCK+STRM5_NBLOCKS)*MEM_WIDTH-1:STRM5_BLOCK*MEM_WIDTH],rdqmux[5],rdblockb[1] - STRM5_BLOCK);
	end else begin: nomux5
		assign rdqmux[5] = rdqb[(STRM5_BLOCK+1)*MEM_WIDTH-1:STRM5_BLOCK*MEM_WIDTH];
	end
	assign m_axis_5_tdata = rdqmux[5][STRM5_WIDTH-1:0];
end

end else begin: singleblock

if(NSTREAMS >= 1) begin: en_strm0_direct
    assign m_axis_0_tdata = rdqa[STRM0_WIDTH-1:0];
end
if(NSTREAMS >= 2) begin: en_strm1_direct
	assign m_axis_1_tdata = rdqb[STRM1_WIDTH-1:0];
end
if(NSTREAMS >= 3) begin: en_strm2_direct
	assign m_axis_2_tdata = rdqa[STRM2_WIDTH-1:0];
end
if(NSTREAMS >= 4) begin: en_strm3_direct
	assign m_axis_3_tdata = rdqb[STRM3_WIDTH-1:0];
end
if(NSTREAMS >= 5) begin: en_strm4_direct
	assign m_axis_4_tdata = rdqa[STRM4_WIDTH-1:0];
end
if(NSTREAMS >= 6) begin: en_strm5_direct
	assign m_axis_5_tdata = rdqb[STRM5_WIDTH-1:0];
end

end
endgenerate

//output to AXI Streams
reg tvalid_pipe0[2:0];
reg tvalid_pipe1[2:0];
reg tvalid_pipe2[2:0];
reg tvalid_pipe3[2:0];
reg tvalid_pipe4[2:0];
reg tvalid_pipe5[2:0];

assign m_axis_0_tvalid = tvalid_pipe0[2];
assign m_axis_1_tvalid = tvalid_pipe1[2];
assign m_axis_2_tvalid = tvalid_pipe2[2];
assign m_axis_3_tvalid = tvalid_pipe3[2];
assign m_axis_4_tvalid = tvalid_pipe4[2];
assign m_axis_5_tvalid = tvalid_pipe5[2];


always @(posedge aclk) begin
    tvalid_pipe0[0] <= strm0_incr_en;
    tvalid_pipe1[0] <= strm1_incr_en;
    tvalid_pipe2[0] <= strm2_incr_en;
    tvalid_pipe3[0] <= strm3_incr_en;
    tvalid_pipe4[0] <= strm4_incr_en;
    tvalid_pipe5[0] <= strm5_incr_en;
    for(i=0; i<2; i=i+1) begin: srl
        tvalid_pipe0[i+1] <= tvalid_pipe0[i];
        tvalid_pipe1[i+1] <= tvalid_pipe1[i];
        tvalid_pipe2[i+1] <= tvalid_pipe2[i];
        tvalid_pipe3[i+1] <= tvalid_pipe3[i];
        tvalid_pipe4[i+1] <= tvalid_pipe4[i];
        tvalid_pipe5[i+1] <= tvalid_pipe5[i];
    end
end

assign config_q0 = 0;

endmodule