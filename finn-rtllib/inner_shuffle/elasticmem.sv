/****************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Memory with free-running read pipeline and credit-based flow control.
 * @author	Shane T. Fleming
 *
 * @description
 *  This module implements a BRAM/URAM wrapper with a free-running read
 *  pipeline of PIPE_DEPTH stages enabling Vivado to absorb the output
 *  registers into the memory primitive. Flow control is achieved through
 *  a credit-based scheme: a read request is accepted only when credits
 *  are available. Credits are returned as data is consumed from the
 *  output queue, whose depth matches the pipeline depth.
 *
 *  The pipeline consists of:
 *    Addr0 -> Mem -> Dout[PIPE_DEPTH-2] -> ... -> Dout[0] -> queue -> output
 *
 *  PIPE_DEPTH is derived from RAM_STYLE to match the absorbable register
 *  stages of the target memory primitive (AM007):
 *    BRAM:  Addr0 -> Mem -> DOA_REG              = 3
 *    URAM:  Addr0 -> Mem -> OREG -> OREG_ECC     = 4
 ***************************************************************************/

module elasticmem #(
	int unsigned  WIDTH,
	int unsigned  DEPTH,
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

	localparam int unsigned  PIPE_DEPTH = (RAM_STYLE == "ultra")? 4 : 3;
	localparam int unsigned  Q_LATENCY = 3;	// queue pass-through latency
	localparam int unsigned  CREDITS   = PIPE_DEPTH + Q_LATENCY;

	//=======================================================================
	// Memory Array with Immediate Write
	(* RAM_STYLE = RAM_STYLE *)
	logic [WIDTH-1:0]  Mem[DEPTH];
	always_ff @(posedge clk) begin
		if(wr_en)  Mem[wr_addr] <= wr_data;
	end

	//=======================================================================
	// Readout Pipeline

	//- Credit-based Flow Control -------
	logic signed [$clog2(CREDITS):0]  Credit = -CREDITS;	// -CREDITS, ..., -1, 0
	uwire  have_credit = Credit[$left(Credit)];
	uwire  issue  = have_credit && rd_req_vld;
	uwire  settle = rd_dat_vld && rd_dat_rdy;
	always_ff @(posedge clk) begin
		if(rst)  Credit <= -CREDITS;
		else     Credit <= Credit + (issue == settle? 0 : issue? 1 : -1);
	end
	assign	rd_req_rdy = have_credit;

	//- Free-Running Core Pipeline (no enables, no reset)

	// Valid Request Identification
	logic  Vld[PIPE_DEPTH] = '{ default: 0 };
	always_ff @(posedge clk) begin
		if(rst)  Vld <= '{ default: 0 };
		else     Vld <= { Vld[1:PIPE_DEPTH-1], issue };
	end

	// Memory Readout Pipeline
	logic [$clog2(DEPTH)-1:0]  Addr0;
	logic [WIDTH-1:0]  Dout[PIPE_DEPTH-1];
	always_ff @(posedge clk) begin
		Addr0 <= rd_addr;
		Dout  <= { Dout[1:PIPE_DEPTH-2], Mem[Addr0] };
	end

	//- Credit-Enabling Reply Queue -----
	uwire  q_irdy;
	queue #(.DATA_WIDTH(WIDTH), .ELASTICITY(CREDITS)) u_queue (
		.clk, .rst,
		.idat(Dout[0]), .ivld(Vld[0]), .irdy(q_irdy),
		.odat(rd_dat), .ovld(rd_dat_vld), .ordy(rd_dat_rdy)
	);
	always_ff @(posedge clk) begin
		assert(rst || !Vld[0] || q_irdy) else begin
			$error("%m: Overrunning output queue.");
			$stop;
		end
	end

endmodule : elasticmem
