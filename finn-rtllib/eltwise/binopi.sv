/****************************************************************************
 * Copyright (C) 2026, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Integer binary operation: a OP b.
 * @author	Shane Fleming <shane.fleming@amd.com>
 ***************************************************************************/

module binopi #(
	parameter  OP,	// ADD(a+b), SUB(a-b), SBR(b-a), MUL(a*b)
	int unsigned  WIDTH,
	bit  SIGNED = 0,

	localparam bit  IS_MUL = (OP == "MUL"),
	localparam int unsigned  O_WIDTH = IS_MUL? 2*WIDTH : WIDTH + 1
)(
	input	logic  clk,
	input	logic  rst,

	input	logic [WIDTH-1:0]  a,
	input	logic  avld,
	input	logic [WIDTH-1:0]  b,
	input	logic  bload,

	output	logic [O_WIDTH-1:0]  r,
	output	logic  rvld
);

	localparam int unsigned  LATENCY = IS_MUL? 3 : 1;

	initial begin
		if(OP != "ADD" && OP != "SUB" && OP != "SBR" && OP != "MUL") begin
			$error("%m: Unsupported integer operation %s", OP);
			$finish;
		end
	end

	//=== Valid Signalling ================================================
	logic [LATENCY-1:0]  Vld = '0;
	always_ff @(posedge clk) begin
		if(rst)  Vld <= '0;
		else     Vld <= { Vld, avld };
	end
	assign	rvld = Vld[$left(Vld)];

	//=== Multiply Pipeline (DSP-inferable) ===============================
	// 3 stages: input regs (AREG), product reg (MREG), output reg (PREG).
	// Vivado retimes into a DSP58 INT MUL.
	if(IS_MUL) begin : genMul
		logic [  WIDTH-1:0]  A1 = 'x;
		logic [  WIDTH-1:0]  B1 = 'x;
		logic [O_WIDTH-1:0]  M  = 'x;
		logic [O_WIDTH-1:0]  P  = 'x;
		always_ff @(posedge clk) begin
			if(rst) begin
				A1 <= 'x;  B1 <= 'x;  M <= 'x;  P <= 'x;
			end
			else begin
				A1 <= a;
				if(bload)  B1 <= b;
				M  <= SIGNED? O_WIDTH'($signed(A1) * $signed(B1)) : A1 * B1;
				P  <= M;
			end
		end
		assign	r = P;
	end : genMul
	else begin : genAddSub
		logic [O_WIDTH-1:0]  P1 = 'x;
		always_ff @(posedge clk) begin
			if(rst)  P1 <= 'x;
			else begin
				if(SIGNED) begin
					unique case(OP)
					"ADD":  P1 <= O_WIDTH'($signed(a) + $signed(b));
					"SUB":  P1 <= O_WIDTH'($signed(a) - $signed(b));
					"SBR":  P1 <= O_WIDTH'($signed(b) - $signed(a));
					endcase
				end
				else begin
					unique case(OP)
					"ADD":  P1 <= a + b;
					"SUB":  P1 <= a - b;
					"SBR":  P1 <= b - a;
					endcase
				end
			end
		end
		assign	r = P1;
	end : genAddSub

endmodule : binopi
