/****************************************************************************
 * Copyright (C) 2026, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 * @brief	Integer requantization core with decoupled (streamed) parameters.
 *
 * @description
 *	Decoupled variant of the requant core. Instead of embedding the
 *	single-precision scale/bias as compile-time module parameters (which are
 *	constant-folded into fixed-point in requant.sv), this variant receives the
 *	*already decomposed* fixed-point parameters through a single stream, one
 *	word per compute beat.  Each PE lane carries a packed struct (LSB to MSB):
 *		- SCALE [S_WIDTH]       — signed scale mantissa
 *		- T     [SHIFT_WIDTH]   — shift - SHIFT_MIN (unsigned)
 *		- BIAS  [BIAS_WIDTH]    — signed bias (round constant folded in)
 *
 *	The fixed-point precision considerations documented in requant.sv apply
 *	equally here: the exact integer multiply may produce results differing
 *	by ±1 from a single-precision floating-point evaluation of
 *	round(scale*x+bias) when the scale has a dense (non-power-of-two)
 *	mantissa. The external parameter decomposition feeding the pdat stream
 *	must follow the same fixed-point format as requant.sv::derive_PARAMS.
 ***************************************************************************/

module requant_decoupled #(
	int unsigned  VERSION = 1,  // DSP Version
	int unsigned  K,  // Input Precision
	int unsigned  N,  // Output Precision

	int unsigned  C,       // Channel count
	int unsigned  PE = 1,  // Vector parallelism, must divide C

	int unsigned  SHIFT_MIN,  // Worst-case minimum shift across all channels
	int unsigned  SHIFT_MAX,  // Worst-case maximum shift across all channels

	bit  SIGNED_OUT = 0,  // 0: unsigned clip [0, 2^N-1], 1: signed clip [-2^(N-1), 2^(N-1)-1]

	// Derived multiplier operand widths (must match derive_MUL_WIDTHS)
	localparam int unsigned  S_WIDTH = (K <= (VERSION==3? 24 : 18))? 25 :
	                                    (VERSION==3? 24 : 18),
	localparam int unsigned  X_WIDTH = (K <= (VERSION==3? 24 : 18))? K :
	                                    ((VERSION==1? 25 : 27) < K? (VERSION==1? 25 : 27) : K),
	localparam int unsigned  BIAS_WIDTH  = S_WIDTH + X_WIDTH,
	localparam int unsigned  SHIFT_RANGE = SHIFT_MAX - SHIFT_MIN + 1,
	localparam int unsigned  SHIFT_WIDTH  = (SHIFT_RANGE > 1)? $clog2(SHIFT_RANGE) : 1,
	localparam int unsigned  PARAMS_LANE_WIDTH = S_WIDTH + SHIFT_WIDTH + BIAS_WIDTH
)(
	input	logic  clk,
	input	logic  rst,

	// Input data stream lane-packed
	input	logic signed [PE-1:0][K-1:0]  idat,
	input	logic  ivld,

	// Parameter stream (per lane: { BIAS, T, SCALE })
	input	logic [PE-1:0][PARAMS_LANE_WIDTH-1:0]  pdat,

	// Output data stream
	output	logic [PE-1:0][N-1:0]  odat,
	output	logic  ovld
);
`default_nettype none
	localparam int unsigned  CF = C/PE;  // Channel fold

	// Parameter Constraints Checking
	initial begin
		if(CF*PE != C) begin
			$error("%m: Parallelism PE=%0d does not divide channel count C=%0d.", PE, C);
			$finish;
		end
		if(SHIFT_MAX < SHIFT_MIN) begin
			$error("%m: SHIFT_MAX=%0d smaller than SHIFT_MIN=%0d.", SHIFT_MAX, SHIFT_MIN);
			$finish;
		end
	end

	// Global valid flag forwarding (4 pipeline stages, matches requant.sv)
	logic  Vld[4] = '{ default: 0 };
	always_ff @(posedge clk) begin
		if(rst)  Vld <= '{ default: 0 };
		else     Vld <= { ivld, Vld[0:2] };
	end
	assign	ovld = Vld[3];

	// Instantiate individual compute lanes
	for(genvar  pe = 0; pe < PE; pe++) begin : genPE
		typedef logic [SHIFT_WIDTH-1:0]  shift_t;

		//- Stage #1: sample input + streamed parameters
		logic signed [X_WIDTH-1:0]  X1 = 'x;
		logic signed [S_WIDTH-1:0]  S1 = 'x;
		logic signed [BIAS_WIDTH -1:0]  B1 = 'x;
		shift_t  T1 = 'x;
		always_ff @(posedge clk) begin
			if(rst) begin
				X1 <= 'x;
				S1 <= 'x;
				B1 <= 'x;
				T1 <= 'x;
			end
			else begin
				X1 <= K > X_WIDTH? idat[pe][K-X_WIDTH+:X_WIDTH] : idat[pe];
				S1 <= $signed(pdat[pe][0+:S_WIDTH]);
				T1 <= pdat[pe][S_WIDTH+:SHIFT_WIDTH];
				B1 <= $signed(pdat[pe][S_WIDTH+SHIFT_WIDTH+:BIAS_WIDTH]);
			end
		end

		//- Stage #2: multiply
		logic signed [BIAS_WIDTH-1:0]  M2 = 'x;
		logic signed [BIAS_WIDTH-1:0]  B2 = 'x;
		shift_t  T2 = 'x;
		always_ff @(posedge clk) begin
			if(rst) begin
				M2 <= 'x;
				B2 <= 'x;
				T2 <= 'x;
			end
			else begin
				M2 <= X1 * S1;
				B2 <= B1;
				T2 <= T1;
			end
		end

		//- Stage #3: add bias
		logic signed [BIAS_WIDTH:0]  P3 = 'x;
		shift_t  T3 = 'x;
		always_ff @(posedge clk) begin
			if(rst) begin
				P3 <= 'x;
				T3 <= 'x;
			end
			else begin
				P3 <= M2 + B2;
				T3 <= T2;
			end
		end

		//- Stage #4: window extract, shift, clip (window sized worst-case)
		logic [N-1:0]  R4 = 'x;
		localparam int unsigned  SHIFT_SPAN = SHIFT_MAX - SHIFT_MIN;
		uwire  neg = P3[$left(P3)];
		if(!SIGNED_OUT) begin : blkStage4Unsigned
			uwire [SHIFT_SPAN + N-1:0]  win = P3[SHIFT_MAX+N-1 : SHIFT_MIN];
			uwire [SHIFT_SPAN + N-1:0]  shifted = win >> T3;
			uwire  ovf =
				(($left(P3)    > SHIFT_MAX+N)? |P3[$left(P3)-1:SHIFT_MAX+N] : 0) ||
				((SHIFT_MIN    < SHIFT_MAX  )? |shifted[$left(shifted):N] : 0);
			always_ff @(posedge clk) begin
				if(rst)  R4 <= 'x;
				else begin
					R4 <=
						neg?  0 :
						ovf? '1 :
						shifted[N-1:0];
				end
			end
		end : blkStage4Unsigned
		else begin : blkStage4Signed
			uwire signed [SHIFT_SPAN + N-1:0]  win = P3[SHIFT_MAX+N-1 : SHIFT_MIN];
			uwire signed [SHIFT_SPAN + N-1:0]  shifted = win >>> T3;
			uwire  ovf =
				(($left(P3)    > SHIFT_MAX+N-1)? |(P3[$left(P3)-1:SHIFT_MAX+N-1] ^ {($left(P3)-SHIFT_MAX-N+1){neg}}) : 0) ||
				((SHIFT_MIN    < SHIFT_MAX     )? ~(&shifted[$left(shifted):N-1]) && (|shifted[$left(shifted):N-1]) : 0);
			always_ff @(posedge clk) begin
				if(rst)  R4 <= 'x;
				else     R4 <= ovf? {neg, {(N-1){!neg}}} : shifted[N-1:0];
			end
		end : blkStage4Signed

		assign	odat[pe] = R4;
	end : genPE

`default_nettype wire
endmodule : requant_decoupled
