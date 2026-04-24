/**
 * Copyright (c) 2023, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of FINN nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 */

module memstream #(
	int unsigned  SETS,
	int unsigned  DEPTH,
	int unsigned  WIDTH,

	parameter  INIT_FILE = "",
	parameter  RAM_STYLE = "auto",

	localparam int unsigned  ADDR_WIDTH0 = $clog2(SETS*DEPTH),
	localparam int unsigned  ADDR_WIDTH = ADDR_WIDTH0? ADDR_WIDTH0 : 1,
	localparam int unsigned  SET_BITS = SETS > 2? $clog2(SETS) : 1
)(
	input	logic  clk,
	input	logic  rst,

	// Configuration and readback interface - compatible with ap_memory
	input	logic  config_ce,
	input	logic  config_we,
	input	logic [ADDR_WIDTH-1:0]  config_address,
	input	logic [WIDTH-1:0]  config_d0,

	output	logic  config_rack,
	output	logic [WIDTH-1:0]  config_q0,

	// Set selector stream (ignored for SETS = 1)
	input	logic [SET_BITS-1:0]  sidx,
	input	logic  svld,
	output	logic  srdy,

	// Continuous output stream
	input	logic  ordy,
	output	logic  ovld,
	output	logic [WIDTH-1:0]  odat
);

	//-----------------------------------------------------------------------
	// Generation of the Primary Read Address Sequence
	typedef logic [ADDR_WIDTH-1:0]  addr_t;

	uwire addr_t  ptr0;
	uwire  vld0;
	uwire  got0;
	if(SETS < 2) begin : genSingleSet
		// Increment for wrapping DEPTH-1 back to zero
		localparam int unsigned  WRAP_INC = 2**ADDR_WIDTH - DEPTH + 1;

		addr_t  Ptr = 0;
		logic   Lst = DEPTH < 2;
		always_ff @(posedge clk) begin
			if(rst) begin
				Ptr <= 0;
				Lst <= DEPTH < 2;
			end
			else if(got0) begin
				Ptr <= Ptr + (Lst? WRAP_INC : 1);
				Lst <= (DEPTH < 2) || (Ptr == DEPTH-2);
			end
		end
		assign	ptr0 = Ptr;
		assign	vld0 = 1;

	end : genSingleSet
	else begin : genMultiSets

		// Stream Offsets through Lookup Table
		typedef addr_t  offset_table_t[SETS];
		function offset_table_t INIT_OFFSET_TABLE();
			automatic offset_table_t  res;
			foreach(res[i])  res[i] = i*DEPTH;
			return  res;
		endfunction : INIT_OFFSET_TABLE
		localparam offset_table_t  OFFSET_TABLE = INIT_OFFSET_TABLE();

		uwire  lst;
		if(DEPTH < 2)  assign  lst = 1;
		else begin
			logic [$clog2(DEPTH-1):0]  Cnt = 'x; // DEPTH-2, ..., 1, 0, -1 (lst)
			always_ff @(posedge clk) begin
				if(rst)  Cnt <= 'x;
				else begin
					unique if(svld && srdy)  Cnt <= DEPTH-2;
					else   if(got0 && !lst)  Cnt <= Cnt - 1;
					else begin end
				end
			end
			assign	lst = Cnt[$left(Cnt)];
		end

		addr_t  Ptr = 'x;
		logic   Vld =  0;
		assign	srdy = !Vld || (lst && got0);
		always_ff @(posedge clk) begin
			if(rst) begin
				Ptr <= 'x;
				Vld <=  0;
			end
			else begin
				unique if(svld && srdy)  Ptr <= OFFSET_TABLE[sidx];
				else   if(got0 && !lst)  Ptr <= Ptr + 1;
				else begin end
				Vld <= svld || !srdy;
			end
		end
		assign	ptr0 = Ptr;
		assign	vld0 = Vld;

	end : genMultiSets

	//-----------------------------------------------------------------------
	// Memory Pipeline
	//	- free-running pipeline
	//	- credit-based feed of reads for streaming output
	//	- config interface always takes precedence
	localparam int unsigned  FULL_CREDIT = 8;
	typedef logic [WIDTH-1:0]  data_t;

	//- Stage #1: Command Arbitration
	uwire  ogot = ovld && ordy;

	logic  Wr1 = 0;
	logic  Rb1 = 0;
	logic  Rs1 = 0;
	addr_t  Ptr1 = 'x;
	data_t  Dat1 = 'x;
	if(1) begin : blkFeed

		logic signed [$clog2(FULL_CREDIT):0]  Credit = -FULL_CREDIT;

		assign	got0 = vld0 && Credit[$left(Credit)] && !config_ce;
		always_ff @(posedge clk) begin
			if(rst) begin
				Credit <= -FULL_CREDIT;

				Wr1 <= 0;
				Rb1 <= 0;
				Rs1 <= 0;
				Ptr1 <= 'x;
				Dat1 <= 'x;
			end
			else begin
				Credit <= Credit + (ogot == got0? 0 : got0? 1 : -1);

				Wr1 <= config_ce &&  config_we;
				Rb1 <= config_ce && !config_we;
				Rs1 <= got0;
				Ptr1 <= config_ce? config_address : ptr0;
				Dat1 <= config_d0;
			end
		end
	end : blkFeed

	// Stage #2+3: Memory Readout
	logic  Rb3 = 0;
	logic  Rs3 = 0;
	data_t  Dat3 = 'x; /* absorbed register */
	if(1) begin : blkMem
		(* RAM_STYLE = RAM_STYLE *)
		data_t  Mem[SETS*DEPTH];

		// Optional Memory Initialization
		if(INIT_FILE != "")  initial $readmemh(INIT_FILE, Mem);

		// Execute Memory Operation
		logic  Rb2 = 0;
		logic  Rs2 = 0;
		data_t  Dat2 = 'x; /* absorbed register */
		always_ff @(posedge clk) begin // Memory datapath without reset
			if(Wr1)  Mem[Ptr1] <= Dat1;
			Dat2 <= Mem[Ptr1];
			Dat3 <= Dat2;
		end
		always_ff @(posedge clk) begin
			if(rst) begin
				Rb2 <= 0;
				Rs2 <= 0;
				Rb3 <= 0;
				Rs3 <= 0;
			end
			else begin
				Rb2 <= Rb1;
				Rs2 <= Rs1;
				Rb3 <= Rb2;
				Rs3 <= Rs2;
			end
		end
	end : blkMem

	// Complete Readbacks
	assign	config_rack = Rb3;
	assign	config_q0   = Dat3;

	// Stage #4+5: Output Credit Buffer
	if(1) begin : blkStreamOut

		uwire  oload;

		// Stream Output SRL backing Credit Grant for Stream Reads in Flight
		(* SHREG_EXTRACT = "yes" *)
		data_t  OSrl[FULL_CREDIT-1];
		logic [$clog2(FULL_CREDIT):0]  OPtr = -1;
		always_ff @(posedge clk) begin
			if(Rs3)  OSrl <= { Dat3, OSrl[0:FULL_CREDIT-3] };
		end
		always_ff @(posedge clk) begin
			if(rst)  OPtr <= -1;
			else begin
				automatic logic  up = Rs3;
				automatic logic  dn = oload && !OPtr[$left(OPtr)];
				OPtr <= OPtr + (up == dn? 0 : up? 1 : -1);
			end
		end

		// Final Output Register
		(* EXTRACT_ENABLE = "yes" *)
		data_t  ODat = 'x;
		logic   OVld =  0;
		always_ff @(posedge clk) begin
			if(rst) begin
				OVld <= 0;
				ODat <= 'x;
			end
			else if(oload) begin
				OVld <= !OPtr[$left(OPtr)];
				ODat <= OSrl[OPtr];
			end
		end
		assign	oload = ordy || !OVld;

		assign	odat = ODat;
		assign	ovld = OVld;

	end : blkStreamOut

endmodule : memstream
