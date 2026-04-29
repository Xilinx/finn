/****************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Two-input elementwise stream operation.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 ***************************************************************************/

module eltwisef #(
	parameter int unsigned PE = 1,  // Number of processing elements
	parameter  OP,	// ADD(a+b), SUB(a-b), SBR(b-a), MUL(a*b)
	shortreal  B_SCALE = 1.0,	// Scale `b` input, must be 1.0 for MUL
	bit  FORCE_BEHAVIORAL = 0
)(
	// Global Control
	input	logic  clk,
	input	logic  rst,

	// Input Streams - PE elements wide
	input	logic [PE-1:0][31:0]  adat,
	input	logic  avld,
	output	logic  ardy,
	input	logic [PE-1:0][31:0]  bdat,
	input	logic  bvld,
	output	logic  brdy,

	// Output Stream - PE elements wide
	output	logic [PE-1:0][31:0]  odat,
	output	logic  ovld,
	input	logic  ordy
);

	typedef logic [31:0]  fp_t;
	typedef logic [PE-1:0][31:0]  fp_vec_t;

	// Input Sidestep Registers - PE-wide
	uwire  take;

	typedef struct {
		fp_vec_t   val;
		logic  rdy;
	} ibuf_t;
	ibuf_t  A = '{ val: 'x, rdy: '1 };
	ibuf_t  B = '{ val: 'x, rdy: '1 };
	always_ff @(posedge clk) begin
		if(rst) begin
			A <= '{ val: 'x, rdy: '1 };
			B <= '{ val: 'x, rdy: '1 };
		end
		else begin
			if(A.rdy)  A.val <= adat;
			A.rdy <= (A.rdy && !avld) || take;
			if(B.rdy)  B.val <= bdat;
			B.rdy <= (B.rdy && !bvld) || take;
		end
	end
	assign	ardy = A.rdy;
	assign	brdy = B.rdy;
	uwire fp_vec_t  a = A.rdy? adat : A.val;
	uwire fp_vec_t  b = B.rdy? bdat : B.val;

	// Credit-based Operation Issue
	localparam int unsigned  CREDIT = 8;
	logic signed [$clog2(CREDIT):0]  Credit = -CREDIT;
	uwire  give = ovld && ordy;
	assign	take = (avld||!ardy) && (bvld||!brdy) && Credit[$left(Credit)];
	always_ff @(posedge clk) begin
		if(rst)  Credit <= -CREDIT;
		else     Credit <= Credit + (give == take? 0 : give? -1 : 1);
	end

	// Free-running Compute Pipeline - PE instances
	uwire fp_vec_t  r;
	uwire [PE-1:0]  rvld_vec;
	uwire  rvld;

	// Generate PE binopf instances
	genvar i;
	generate
		for (i = 0; i < PE; i++) begin : gen_binopf
			binopf #(.OP(OP), .B_SCALE(B_SCALE), .FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)) core (
				.clk, .rst,
				.a(a[i]), .avld(take),
				.b(b[i]), .bload('1),
				.r(r[i]), .rvld(rvld_vec[i])
			);
		end
	endgenerate

	// All PE results should be valid simultaneously
	assign rvld = &rvld_vec;

	// Credit-backing Elastic Output Queue - PE-wide
	uwire  rrdy;
	queue #(.DATA_WIDTH($bits(fp_vec_t)), .ELASTICITY(CREDIT)) obuf (
		.clk, .rst,
		.idat(r), .ivld(rvld), .irdy(rrdy),
		.odat, .ovld, .ordy
	);
	always_ff @(posedge clk) begin
		assert(rrdy || !rvld) else begin
			$error("%m: Result queue overrun.");
			$stop;
		end
	end

endmodule : eltwisef
