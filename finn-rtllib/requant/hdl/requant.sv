/****************************************************************************
 * Copyright (C) 2026, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 * @brief	Integer requantization from K to N bits by clip(round(x*scale+bias)).
 *
 * @description
 *	Performs an integer requantization from K to N bits as by computing:
 *		round(x*scale+bias)
 *	The used single-precision floating-point scale and bias parameters are
 *	rotated round-robin through the list of C channel-specific values.
 ***************************************************************************/

module requant #(
	int unsigned  VERSION = 1,  // DSP Version
	int unsigned  K,  // Input Precision
	int unsigned  N,  // Output Precision

	int unsigned  C,       // Channel count
	int unsigned  PE = 1,  // Vector parallelism, must divide C

	shortreal     SCALES[PE][C/PE],
	shortreal     BIASES[PE][C/PE]
)(
	input	logic  clk,
	input	logic  rst,

	input	logic signed [PE-1:0][K-1:0]  idat,
	input	logic  ivld,

	output	logic [PE-1:0][N-1:0]  odat,
	output	logic  ovld
);
	localparam int unsigned  CF = C/PE;  // Channel fold

	// Parameter Constraints Checking
	initial begin
		if(CF*PE != C) begin
			$error("%m: Parallelism PE=%0d does not divide channel count C=%0d.", PE, C);
			$finish;
		end
	end

	// Derive allowable multiplier input widths and implied datapath assignments
	typedef struct {
		int unsigned  s;
		int unsigned  x;
	} mul_widths_t;
	function automatic mul_widths_t derive_MUL_WIDTHS();
		mul_widths_t  res;

		int unsigned  aw;
		int unsigned  bw;
		unique case(VERSION)
		1: begin  aw = 25; bw = 18; end
		2: begin  aw = 27; bw = 18; end
		3: begin  aw = 27; bw = 24; end
		default: begin
			$error("%m: Unsupported DSP VERSION %0d.", VERSION);
			$finish;
		end
		endcase
		if(K <= bw) begin
			res.x = K;
			res.s = 25;
		end
		else begin
			res.x = aw < K? aw : K;
			res.s = bw;
		end

		return  res;

	endfunction : derive_MUL_WIDTHS
	localparam mul_widths_t  MUL_WIDTHS = derive_MUL_WIDTHS();

	// Derive fixed-point parameters for requantization
	typedef struct {
		bit signed   [MUL_WIDTHS.s               -1:0]  scale;
		bit signed   [MUL_WIDTHS.s + MUL_WIDTHS.x  :0]  bias;
		int unsigned  tap;
	} params_t;
	typedef params_t  params_mat_t[PE][CF];
	function automatic params_mat_t derive_PARAMS();
		params_mat_t  res;
		foreach(res[pe, cf]) begin
			params_t  p;

			// Decomposed Scale
			bit [31:0]  s = $shortrealtobits(SCALES[pe][cf]);
			bit [ 7:0]  sc = s[23+:8];
			bit [24:0]  sm = { 1'b1, s[0+:23], 1'b0 }; // * 2^-24
			int         se = int'(sc) - 127;

			// Decomposed Bias
			bit [31:0]  b = $shortrealtobits(BIASES[pe][cf]);
			bit [ 7:0]  bc = b[23+:8];
			bit [23:0]  bm = { 1'b1, b[0+:23] }; // * 2^-23
			int         be = int'(bc) - 127;

			// Tap from Product Datapath
			p.tap = (MUL_WIDTHS.s - 2) - se;
			if(p.tap < 0) begin
				$error("%m: Scale is too large for the output precision.");
				$finish;
			end
			if(MUL_WIDTHS.s + MUL_WIDTHS.x + 1 < p.tap+N) begin
				$error("%m: Scale is too small for the output precision.");
				$finish;
			end

			// Mantissa into Product Datapath
			sm >>= (25 - MUL_WIDTHS.s);
			p.scale = sm[1+:24] + sm[0];
			if(s[31])  p.scale = -p.scale;

			// Align Bias Mantissa into Product Datapath
			if(be > int'(MUL_WIDTHS.x - N)) begin
				$error("%m: Bias overflows the output range.");
				$finish;
			end
			else begin
				// - p.tap identifies the number of fractional bits in the product datapath, bm comes with 23
				// - a bias exponent equal to 23-p.tap requires no shifting of its mantissa for alignment
				// - a bias exponent smaller than 23-p.tap requires a right shift
				// - a bias exponent larger than 23-p.tap requires a left shift
				automatic int  shift = be - (23 - p.tap);
				p.bias =
					(shift < -24)? 0 :
					(shift <   0)? bm >> -shift :
					/* else */     bm <<  shift;

					if(b[31])  p.bias = -p.bias;

					// Rounding
					if(p.tap > 0)       p.bias += 1 << (p.tap-1);
					else if(shift < 0)  p.bias += bm[-shift-1];
			end

			res[pe][cf] = p;
		end
		return  res;
	endfunction : derive_PARAMS
	localparam params_mat_t  PARAMS = derive_PARAMS();
	initial begin
		void'(derive_PARAMS());
	end
	typedef struct {
		int unsigned  min;
		int unsigned  max;
	} minmax_t;
	function automatic minmax_t derive_TAP_MINMAX(input int unsigned  pe);
		minmax_t  res = '{ min: 2**32-1, max: 0 };
		for(int unsigned  cf = 0; cf < CF; cf++) begin
			automatic int unsigned  tap = PARAMS[pe][cf].tap;
			if(tap < res.min)  res.min = tap;
			if(tap > res.max)  res.max = tap;
		end

		return  res;
	endfunction : derive_TAP_MINMAX

	int unsigned  cnl_sel;
	if(CF == 1)  assign  cnl_sel = 0;
	else begin
		logic [$clog2(CF)-1:0]  CnlCnt = 0;
		always_ff @(posedge clk) begin
			if(rst)        CnlCnt <= 0;
			else if(ivld)  CnlCnt <= CnlCnt + (CnlCnt != CF-1? 1 : (2**$clog2(CF)-CF)+1);
		end
		assign	cnl_sel = CnlCnt;
	end

	// Global valid flag forwarding
	logic  Vld[4] = '{ default: 0 };
	always_ff @(posedge clk) begin
		if(rst)  Vld <= '{ default: 0 };
		else     Vld <= { ivld, Vld[0:2] };
	end
	assign	ovld = Vld[3];

	// Instantiate individual compute lanes
	for(genvar  pe = 0; pe < PE; pe++) begin : genPE
		localparam minmax_t  TAP_MINMAX = derive_TAP_MINMAX(pe);
		localparam int unsigned  TAP_RANGE = TAP_MINMAX.max - TAP_MINMAX.min + 1;
		typedef logic [((TAP_RANGE > 1)? $clog2(TAP_RANGE) : 1)-1:0]  tap_t;

		logic signed [             MUL_WIDTHS.x-1:0]  X1 = 'x;
		logic signed [MUL_WIDTHS.s             -1:0]  S1 = 'x;
		logic signed [MUL_WIDTHS.s+MUL_WIDTHS.x-1:0]  B1 = 'x;
		tap_t  T1 = 'x;
		always_ff @(posedge clk) begin
			if(rst) begin
				X1 <= 'x;
				S1 <= 'x;
				B1 <= 'x;
				T1 <= 'x;
			end
			else begin
				automatic params_t  p = PARAMS[pe][cnl_sel];
				X1 <= K > MUL_WIDTHS.x? idat[pe][K-MUL_WIDTHS.x+:MUL_WIDTHS.x] : idat[pe];
				S1 <= p.scale;
				B1 <= p.bias;
				T1 <= p.tap - TAP_MINMAX.min;
			end
		end

		logic signed [MUL_WIDTHS.s+MUL_WIDTHS.x-1:0]  M2 = 'x;
		logic signed [MUL_WIDTHS.s+MUL_WIDTHS.x-1:0]  B2 = 'x;
		tap_t  T2 = 'x;
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

		logic signed [MUL_WIDTHS.s+MUL_WIDTHS.x:0]  P3 = 'x;
		tap_t  T3 = 'x;
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

		logic [N-1:0]  R4 = 'x;
		if(1) begin : blkStage4
			localparam int unsigned  TAP_SPAN = TAP_MINMAX.max - TAP_MINMAX.min;
			uwire [TAP_SPAN + N-1:0]  win = P3[TAP_MINMAX.max+N-1 : TAP_MINMAX.min];
			uwire [TAP_SPAN + N-1:0]  tap = win >> T3;
			uwire  neg = P3[$left(P3)];
			uwire  ovf =
				(($left(P3)      > TAP_MINMAX.max+N)? |P3[$left(P3)-1:TAP_MINMAX.max+N] : 0) ||
				((TAP_MINMAX.min < TAP_MINMAX.max  )? |tap[$left(tap):N] : 0);
			always_ff @(posedge clk) begin
				if(rst) begin
					R4 <= 'x;
				end
				else begin
					R4 <=
						neg?  0 :
						ovf? '1 :
						tap[N-1:0];
				end
			end
		end : blkStage4

		assign	odat[pe] = R4;
	end : genPE

endmodule : requant
