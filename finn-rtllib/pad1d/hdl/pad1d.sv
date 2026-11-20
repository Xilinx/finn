/****************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	One-dimensional padding for folded token streams.
 * @author	Oliver Cassidy <oliver.cassidy@amd.com>
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 *
 * @description
 *	For each sequence, emits PAD_LEFT_TOKENS tokens from PAD_LEFT_DATA,
 *	NUM_TOKENS input tokens, and PAD_RIGHT_TOKENS tokens from
 *	PAD_RIGHT_DATA.
 *
 *	Each token is transferred as NUM_CHANNELS/SIMD folds of SIMD
 *	ELEM_WIDTH-bit elements.  All packed data dimensions are
 *	little-endian.
 ***************************************************************************/

module pad1d #(
	int unsigned  NUM_TOKENS,
	int unsigned  NUM_CHANNELS,
	int unsigned  ELEM_WIDTH,
	int unsigned  SIMD = 1,
	int unsigned  PAD_LEFT_TOKENS = 0,
	int unsigned  PAD_RIGHT_TOKENS = 0,
	bit           EAGER = 1,

	localparam int unsigned  PAD_LEFT_DEPTH  = (PAD_LEFT_TOKENS  > 0)? PAD_LEFT_TOKENS  : 1,
	localparam int unsigned  PAD_RIGHT_DEPTH = (PAD_RIGHT_TOKENS > 0)? PAD_RIGHT_TOKENS : 1,
	parameter logic [PAD_LEFT_DEPTH -1:0][NUM_CHANNELS-1:0][ELEM_WIDTH-1:0]  PAD_LEFT_DATA  = '0,
	parameter logic [PAD_RIGHT_DEPTH-1:0][NUM_CHANNELS-1:0][ELEM_WIDTH-1:0]  PAD_RIGHT_DATA = '0
)(
	// Global Control
	input	logic  clk,
	input	logic  rst,

	// Input Stream
	output	logic  irdy,
	input	logic  ivld,
	input	logic [SIMD-1:0][ELEM_WIDTH-1:0]  idat,

	// Output Stream
	input	logic  ordy,
	output	logic  ovld,
	output	logic [SIMD-1:0][ELEM_WIDTH-1:0]  odat
);
`default_nettype none

	//=== Parameter Validation ==============================================
	initial begin
		automatic bit  fail = 0;

		if(NUM_TOKENS < 1) begin
			$error("%m: NUM_TOKENS must be at least 1.");
			fail = 1;
		end
		if(NUM_CHANNELS < 1) begin
			$error("%m: NUM_CHANNELS must be at least 1.");
			fail = 1;
		end
		if(SIMD < 1) begin
			$error("%m: SIMD must be at least 1.");
			fail = 1;
		end
		if(ELEM_WIDTH < 1) begin
			$error("%m: ELEM_WIDTH must be at least 1.");
			fail = 1;
		end
		if(NUM_CHANNELS % SIMD != 0) begin
			$error("%m: SIMD must divide NUM_CHANNELS.");
			fail = 1;
		end
		if(fail)  $finish;
	end

	if(PAD_LEFT_TOKENS + PAD_RIGHT_TOKENS == 0) begin : genTrivial
		// No Padding: simple passthru wiring
		assign	irdy = ordy;
		assign	ovld = ivld;
		assign	odat = idat;
	end : genTrivial
	else begin : genPad
		// A single row of PAD_LEFT_TOKENS + NUM_TOKENS + PAD_RIGHT_TOKENS tokens
		localparam int unsigned  NUM_POSITIONS = PAD_LEFT_TOKENS + NUM_TOKENS + PAD_RIGHT_TOKENS;
		localparam int unsigned  XBITS = $clog2(NUM_POSITIONS + 1);

		pad #(
			.NUM_CHANNELS(NUM_CHANNELS), .SIMD(SIMD), .ELEM_BITS(ELEM_WIDTH),
			.XCOUNTER_BITS(XBITS), .YCOUNTER_BITS(1),
			.EAGER(EAGER),
			.TOKENS_W(PAD_LEFT_TOKENS), .TOKENS_E(PAD_RIGHT_TOKENS),
			.TOKENS_W_DATA(PAD_LEFT_DATA), .TOKENS_E_DATA(PAD_RIGHT_DATA)
		) core (
			.clk, .rst,

			.xon(XBITS'(PAD_LEFT_TOKENS)),
			.xoff(XBITS'(PAD_LEFT_TOKENS + NUM_TOKENS)),
			.xend(XBITS'(NUM_POSITIONS - 1)),
			.yon(1'd0), .yoff(1'd1), .yend(1'd0),

			.irdy, .ivld, .idat,
			.ordy, .ovld, .odat
		);
	end : genPad

`default_nettype wire
endmodule : pad1d
