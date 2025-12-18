/****************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	BRAM with two output registers and streaming interface with backpressure.
 * @author	Shane T. Fleming
 *
 * @description
 *  This module implements a simple BRAM/URAM wrapper with two output
 *  registers to allow Vivado to fuse them into the BRAM for better timing.
 *  The read side features streaming interfaces with proper backpressure
 *  handling via an integrated skid buffer.
 *
 *  The pipeline consists of:
 *    Address -> BRAM -> Dout1 -> Dout2 -> skid buffer -> output
 *
 *  Backpressure is handled by the skid buffer, which can absorb data
 *  during temporary downstream stalls while maintaining data ordering.
 ***************************************************************************/

module elasticmem #(
	int unsigned  WIDTH,
	int unsigned  DEPTH,
	int unsigned  FEED_STAGES = 0,
	parameter     RAM_STYLE = "auto"
)(
	input	logic  clk,
	input	logic  rst,

	// Write port (simple, no handshake)
	input	logic [WIDTH-1:0]                wr_data,
	input	logic [$clog2(DEPTH)-1:0]        wr_addr,
	input	logic                            wr_en,

	// Read request channel (address)
	input	logic [$clog2(DEPTH)-1:0]        rd_addr,
	input	logic                            rd_req_vld,
	output	logic                            rd_req_rdy,

	// Read data channel (downstream)
	output	logic [WIDTH-1:0]                rd_dat,
	output	logic                            rd_dat_vld,
	input	logic                            rd_dat_rdy
);

	//-----------------------------------------------------------------------
	// BRAM and Pipeline Stages
	(* ram_style = RAM_STYLE *)
	logic [WIDTH-1:0]  Mem [DEPTH-1:0];

	// Pipeline registers
	logic [$clog2(DEPTH)-1:0]  AddrReg  = 'x;
	logic                      AddrVld  =  0;
	logic [WIDTH-1:0]          Dout1    = 'x;
	logic                      Dout1Vld =  0;
	logic [WIDTH-1:0]          Dout2    = 'x;
	logic                      Dout2Vld =  0;

	//-----------------------------------------------------------------------
	// Write Port
	always_ff @(posedge clk) begin
		if(wr_en) begin
			Mem[wr_addr] <= wr_data;
		end
	end

	//-----------------------------------------------------------------------
	// Pipeline Control Logic
	logic  skid_irdy;

	uwire stage2_advance = !Dout2Vld || skid_irdy;
	uwire stage1_advance = !Dout1Vld || stage2_advance;
	uwire stage0_advance = !AddrVld  || stage1_advance;
	assign rd_req_rdy = stage0_advance;

	//-----------------------------------------------------------------------
	// Stage 0: Address Register
	always_ff @(posedge clk) begin
		if(rst) begin
			AddrReg <= 'x;
			AddrVld <= 0;
		end
		else if(stage0_advance) begin
			AddrReg <= rd_addr;
			AddrVld <= rd_req_vld;
		end
	end

	//-----------------------------------------------------------------------
	// Stage 1: First Memory Output Register (BRAM output)
	always_ff @(posedge clk) begin
		if(rst) begin
			Dout1    <= 'x;
			Dout1Vld <= 0;
		end
		else if(stage1_advance) begin
			Dout1Vld <= AddrVld;
			Dout1    <= Mem[AddrReg];
		end
	end

	//-----------------------------------------------------------------------
	// Stage 2: Second Output Register (candidate for BRAM fusion)
	always_ff @(posedge clk) begin
		if(rst) begin
			Dout2    <= 'x;
			Dout2Vld <= 0;
		end
		else if(stage2_advance) begin
			Dout2Vld <= Dout1Vld;
			Dout2    <= Dout1;
		end
	end

	//-----------------------------------------------------------------------
	// Skid Buffer for Backpressure Handling
	skid #(
		.DATA_WIDTH (WIDTH),
		.FEED_STAGES(FEED_STAGES)
	) u_skid (
		.clk  (clk),
		.rst  (rst),

		.idat (Dout2),
		.ivld (Dout2Vld),
		.irdy (skid_irdy),

		.odat (rd_dat),
		.ovld (rd_dat_vld),
		.ordy (rd_dat_rdy)
	);

endmodule : elasticmem
