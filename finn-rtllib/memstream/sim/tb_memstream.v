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

`timescale 1ns/10ps

module tb_memstream;

//parameters to enable/disable axi-mm, set number of streams, set readmemh for memory, set per-stream offsets in memory, set per-stream widths
parameter CONFIG_EN = 1;
parameter NSTREAMS = 4;//1 up to 6

parameter MEM_DEPTH = 9216;
parameter MEM_WIDTH = 32;
parameter MEM_INIT = "./";
parameter MEM_CHECK = "golden.dat";

//widths per stream
parameter STRM0_WIDTH = 32;
parameter STRM1_WIDTH = 32;
parameter STRM2_WIDTH = 32;
parameter STRM3_WIDTH = 32;
parameter STRM4_WIDTH = 1;
parameter STRM5_WIDTH = 1;

//depths per stream
parameter STRM0_DEPTH = 2304;
parameter STRM1_DEPTH = 2304;
parameter STRM2_DEPTH = 2304;
parameter STRM3_DEPTH = 2304;
parameter STRM4_DEPTH = 1;
parameter STRM5_DEPTH = 1;

//offsets for each stream
parameter STRM0_OFFSET = 0;
parameter STRM1_OFFSET = 2304;
parameter STRM2_OFFSET = 4608;
parameter STRM3_OFFSET = 6912;
parameter STRM4_OFFSET = 0;
parameter STRM5_OFFSET = 0;


reg clk;
reg rst;

reg [31:0] config_address = 0;
reg config_ce = 0;
reg config_we = 0;
reg [31:0] config_d0 = 0;
wire [31:0] config_q0;

//multiple wire AXI Streams
reg m_axis_0_afull;
reg m_axis_0_tready;
wire m_axis_0_tvalid;
wire [STRM0_WIDTH-1:0] m_axis_0_tdata;

reg m_axis_1_afull;
reg m_axis_1_tready;
wire m_axis_1_tvalid;
wire [STRM1_WIDTH-1:0] m_axis_1_tdata;

reg m_axis_2_afull;
reg m_axis_2_tready;
wire m_axis_2_tvalid;
wire [STRM2_WIDTH-1:0] m_axis_2_tdata;

reg m_axis_3_afull;
reg m_axis_3_tready;
wire m_axis_3_tvalid;
wire [STRM3_WIDTH-1:0] m_axis_3_tdata;

reg m_axis_4_afull;
reg m_axis_4_tready;
wire m_axis_4_tvalid;
wire [STRM4_WIDTH-1:0] m_axis_4_tdata;

reg m_axis_5_afull;
reg m_axis_5_tready;
wire m_axis_5_tvalid;
wire [STRM5_WIDTH-1:0] m_axis_5_tdata;

reg [MEM_WIDTH-1:0] golden[MEM_DEPTH-1:0];
integer ptr0, ptr1, ptr2, ptr3, ptr4, ptr5;
integer done = 0;
reg [5:0] rng;

//clock
initial begin
    clk = 0;
    forever #5 clk = ~clk;
end

initial begin
    rst = 1;
	config_ce = 0;
    m_axis_0_afull = 0;
    m_axis_1_afull = 0;
    m_axis_2_afull = 0;
    m_axis_3_afull = 0;
    m_axis_4_afull = 0;
    m_axis_5_afull = 0;
    m_axis_0_tready = 1;
    m_axis_1_tready = 1;
    m_axis_2_tready = 1;
    m_axis_3_tready = 1;
    m_axis_4_tready = 1;
    m_axis_5_tready = 1;
    repeat(100) @(negedge clk);
    rst = 0;
    #100
    fork
	    begin
		    $display("Starting to generate random AFULL");
			while(~done) begin
			    rng = $random;
				m_axis_0_afull = rng[0];
				m_axis_1_afull = rng[1];
				m_axis_2_afull = rng[2];
				m_axis_3_afull = rng[3];
				m_axis_4_afull = rng[4];
				m_axis_5_afull = rng[5];
				@(negedge clk);
			end
		end
	join
end


//DUT
memstream
#(
    CONFIG_EN,
    NSTREAMS,
    MEM_DEPTH,
    MEM_WIDTH,
    MEM_INIT,

    //widths per stream
    STRM0_WIDTH,
    STRM1_WIDTH,
    STRM2_WIDTH,
    STRM3_WIDTH,
    STRM4_WIDTH,
    STRM5_WIDTH,

    //depths per stream
    STRM0_DEPTH,
    STRM1_DEPTH,
    STRM2_DEPTH,
    STRM3_DEPTH,
    STRM4_DEPTH,
    STRM5_DEPTH,

    //offsets for each stream
    STRM0_OFFSET,
    STRM1_OFFSET,
    STRM2_OFFSET,
    STRM3_OFFSET,
    STRM4_OFFSET,
    STRM5_OFFSET
)
dut
(
    clk,
    ~rst,

    //optional AXI-Lite interface
    config_address,
    config_ce,
    config_we,
    config_d0,
    config_q0,

    //multiple output AXI Streams
    m_axis_0_afull,
    m_axis_0_tready,
    m_axis_0_tvalid,
    m_axis_0_tdata,

    m_axis_1_afull,
    m_axis_1_tready,
    m_axis_1_tvalid,
    m_axis_1_tdata,

    m_axis_2_afull,
    m_axis_2_tready,
    m_axis_2_tvalid,
    m_axis_2_tdata,

    m_axis_3_afull,
    m_axis_3_tready,
    m_axis_3_tvalid,
    m_axis_3_tdata,

    m_axis_4_afull,
    m_axis_4_tready,
    m_axis_4_tvalid,
    m_axis_4_tdata,

    m_axis_5_afull,
    m_axis_5_tready,
    m_axis_5_tvalid,
    m_axis_5_tdata


);

//stream checkers
initial begin
    ptr0 = STRM0_OFFSET;
	ptr1 = STRM1_OFFSET;
	ptr2 = STRM2_OFFSET;
	ptr3 = STRM3_OFFSET;
	ptr4 = STRM4_OFFSET;
	ptr5 = STRM5_OFFSET;
    fork
		//check stream 0
	    begin
		    $display("Starting stream 0 checker");
		    while(~done & (NSTREAMS > 0)) begin
				@(negedge clk);
				if(m_axis_0_tvalid) begin
					if(m_axis_0_tdata != golden[ptr0]) begin
						$display("Mismatch on stream 0");
						$stop();
					end
					//increment pointer
					ptr0 = ptr0 + 1;
					//rewind pointer if it's reached end
					if(ptr0 == (STRM0_OFFSET + STRM0_DEPTH))
				    ptr0 = STRM0_OFFSET;
				end
			end
		end
		//check stream 1
	    begin
		    $display("Starting stream 1 checker");
		    while(~done & (NSTREAMS > 1)) begin
				@(negedge clk);
				if(m_axis_1_tvalid) begin
					if(m_axis_1_tdata != golden[ptr1]) begin
						$display("Mismatch on stream 1");
						$stop();
					end
					//increment pointer
					ptr1 = ptr1 + 1;
					//rewind pointer if it's reached end
					if(ptr1 == (STRM1_OFFSET + STRM1_DEPTH))
						ptr1 = STRM1_OFFSET;
				end
			end
		end

		//check stream 2
	    begin
		    $display("Starting stream 2 checker");
		    while(~done & (NSTREAMS > 2)) begin
				@(negedge clk);
				if(m_axis_2_tvalid) begin
					if(m_axis_2_tdata != golden[ptr2]) begin
						$display("Mismatch on stream 2");
						$stop();
					end
					//increment pointer
					ptr2 = ptr2 + 1;
					//rewind pointer if it's reached end
					if(ptr2 == (STRM2_OFFSET + STRM2_DEPTH))
						ptr2 = STRM2_OFFSET;
				end
			end
		end
		//check stream 3
	    begin
		    $display("Starting stream 3 checker");
		    while(~done & (NSTREAMS > 3)) begin
				@(negedge clk);
				if(m_axis_3_tvalid) begin
					if(m_axis_3_tdata != golden[ptr3]) begin
						$display("Mismatch on stream 3");
						$stop();
					end
					//increment pointer
					ptr3 = ptr3 + 1;
					//rewind pointer if it's reached end
					if(ptr3 == (STRM3_OFFSET + STRM3_DEPTH))
						ptr3 = STRM3_OFFSET;
				end
			end
		end
		//check stream 4
	    begin
		    $display("Starting stream 4 checker");
		    while(~done & (NSTREAMS > 4)) begin
				@(negedge clk);
				if(m_axis_4_tvalid) begin
					if(m_axis_4_tdata != golden[ptr4]) begin
						$display("Mismatch on stream 4");
						$stop();
					end
					//increment pointer
					ptr4 = ptr4 + 1;
					//rewind pointer if it's reached end
					if(ptr4 == (STRM4_OFFSET + STRM4_DEPTH))
						ptr4 = STRM4_OFFSET;
				end
			end
		end
		//check stream 5
	    begin
		    $display("Starting stream 5 checker");
		    while(~done & (NSTREAMS > 5)) begin
				@(negedge clk);
				if(m_axis_5_tvalid) begin
					if(m_axis_5_tdata != golden[ptr5]) begin
						$display("Mismatch on stream 5");
						$stop();
					end
					//increment pointer
					ptr5 = ptr5 + 1;
					//rewind pointer if it's reached end
					if(ptr5 == (STRM5_OFFSET + STRM5_DEPTH))
						ptr5 = STRM5_OFFSET;
				end
			end
		end
	join
end

initial begin
    done = 0;
	$readmemh(MEM_CHECK,golden);
//    $dumpfile("wave.vcd");
//    $dumpvars(0,tb_memstream);
    @(negedge rst);
    #10000000
	$display("Test done!");
	done = 1;
	#1000
    $finish();
end

endmodule
