/****************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Time-roled implementation of quick reverse square root around a single DSP.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 *
 * @description
 *	static float Q_rsqrt(float const  x) {
 *		float const  x2 = 0.5f * x;
 *		union { uint32_t  i; float  f; } y = { .f = x };
 *		y.i = 0x5f3759df - ( y.i >> 1 );             // what the fuck?
 *		y.f = y.f * ( 1.5f - ( x2 * y.f * y.f ) );   // 1st iteration
 *	//	y.f = y.f * ( 1.5f - ( x2 * y.f * y.f ) );   // 2nd iteration
 *		return  y.f;
 *	}
 *
 *	Implemented compute schedule on DSP slice:
 *	 - 12-stage schedule with three 4-stage iterations through DSP pipeline
 *	 - 4 jobs can be interleaved when filling the entire DSP pipeline
 *
 *	          │             │             │         1 1 │
 *	Stage:    │ 0     1 2 3 │ 4     5 6 7 │ 8     9 0 1 │
 *	Maturity: |    ITER1    |    ITER2    |    ITER3    |
 *	            ▼             ▼             ▼
 *	A1:   y  ──►A─\───A─A─A───A─\───A─A─A───A─\    // hold or re-feed when interleaving
 *	               *             *             *
 *	BD1: x/2 ──►B─/ \         D─/ \         D─/ \
 *	                 \       /     \       /     \
 *	M2:               M     │       M     │       M
 *	                   \    │        \    │        \
 *	M3:                 M   │         M   │         M
 *	                     \ /           \ /           \
 *	P4:                   P             P             P──►
 *	                                   /
 *	C:   1.5 ─────────────────────────/───────────────────
 *	                      ▲             ▲             ▲
 *	                      │             │             │
 *	                    x/2*y     1.5-x/2*y*y  (1.5-x/2*y*y)*y
 *
 ***************************************************************************/

// Local DSP instantiation wrapper.
module rsqrtf_dspfp32 #(
	bit  FORCE_BEHAVIORAL = 0
)(
	input  logic         clk,
	input  logic         rst,

	input  logic         ena,
	input  logic         bsel,
	input  logic         csel,
	input  logic [31:0]  a,
	input  logic [31:0]  b,
	input  logic [31:0]  c,
	input  logic [31:0]  d,

	input  logic         rvld,
	output logic [31:0]  r
);

	logic  invalid;
	logic  overflow;
	logic  underflow;
	localparam logic [6:0]  MODE_MUL = { 2'b00, 3'b010, 2'b01 };
	localparam logic [6:0]  MODE_SUB = { 2'b01, 3'b110, 2'b01 };

	if(FORCE_BEHAVIORAL) begin : genBehav
		logic [31:0]  A1 = 'x;
		logic [31:0]  B1 = 'x;
		logic [31:0]  D1 = 'x;
		logic         BSel1 = 'x;
		logic         CSel3 = 'x;
		logic [31:0]  M[2:3] = '{ default: 'x };
		logic [31:0]  P4 = 'x;

		always_ff @(posedge clk) begin
			if(ena)  A1 <= a;
			B1 <= b;
			D1 <= d;
			BSel1 <= bsel;
			CSel3 <= csel;
			M <= {
				$shortrealtobits($bitstoshortreal(A1)*$bitstoshortreal(BSel1? B1 : D1)),
				M[2]
			};
			P4 <= CSel3? $shortrealtobits(1.5 - $bitstoshortreal(M[3])) : M[3];
		end

		assign r = P4;

		always_comb begin
			invalid = 0;
			overflow = 0;
			underflow = 0;

			if(&r[30-:8]) begin
				if(|r[0+:23])  invalid = 1;
				else           overflow = 1;
			end
		end
	end : genBehav
	else begin : genDSP
		DSPFP32 #(
			.A_FPTYPE("B32"),
			.A_INPUT("DIRECT"),
			.BCASCSEL("B"),
			.B_D_FPTYPE("B32"),
			.B_INPUT("DIRECT"),
			.PCOUTSEL("FPA"),
			.USE_MULT("MULTIPLY"),
			.IS_CLK_INVERTED(1'b0),
			.IS_FPINMODE_INVERTED(1'b0),
			.IS_FPOPMODE_INVERTED(7'b0000000),
			.IS_RSTA_INVERTED(1'b0),
			.IS_RSTB_INVERTED(1'b0),
			.IS_RSTC_INVERTED(1'b0),
			.IS_RSTD_INVERTED(1'b0),
			.IS_RSTFPA_INVERTED(1'b0),
			.IS_RSTFPINMODE_INVERTED(1'b0),
			.IS_RSTFPMPIPE_INVERTED(1'b0),
			.IS_RSTFPM_INVERTED(1'b0),
			.IS_RSTFPOPMODE_INVERTED(1'b0),
			.ACASCREG(1),
			.AREG(1),
			.FPA_PREG(1),
			.FPBREG(1),
			.FPCREG(0),
			.FPDREG(1),
			.FPMPIPEREG(1),
			.FPM_PREG(1),
			.FPOPMREG(1),
			.INMODEREG(1),
			.RESET_MODE("SYNC")
		) DSPFP32_inst (
			.ACOUT_EXP(), .ACOUT_MAN(), .ACOUT_SIGN(),
			.BCOUT_EXP(), .BCOUT_MAN(), .BCOUT_SIGN(),
			.PCOUT(),
			.FPM_INVALID(), .FPM_OVERFLOW(), .FPM_UNDERFLOW(), .FPM_OUT(),
			.FPA_INVALID(invalid), .FPA_OVERFLOW(overflow), .FPA_UNDERFLOW(underflow), .FPA_OUT(r),
			.ACIN_EXP('x), .ACIN_MAN('x), .ACIN_SIGN('x),
			.BCIN_EXP('x), .BCIN_MAN('x), .BCIN_SIGN('x),
			.PCIN('x),
			.CLK(clk), .FPINMODE(bsel), .FPOPMODE(csel? MODE_SUB : MODE_MUL),
			.A_SIGN(a[31]), .A_EXP(a[30:23]), .A_MAN(a[22:0]),
			.B_SIGN(b[31]), .B_EXP(b[30:23]), .B_MAN(b[22:0]),
			.C(c),
			.D_SIGN(d[31]), .D_EXP(d[30:23]), .D_MAN(d[22:0]),
			.ASYNC_RST('0),
			.CEA1('0), .CEA2(ena),
			.CEB('1), .CEC('0), .CED('1),
			.CEFPA('1), .CEFPINMODE('1), .CEFPM('1), .CEFPMPIPE('1), .CEFPOPMODE('1),
			.RSTA('0), .RSTB('0), .RSTC('0), .RSTD('0),
			.RSTFPA('0), .RSTFPINMODE('0), .RSTFPM('0), .RSTFPMPIPE('0), .RSTFPOPMODE('0)
		);
	end : genDSP

	always_ff @(posedge clk) begin
		if(!rst && rvld) begin
			assert(!invalid) else $warning("%m generated invalid output.");
			assert(!overflow) else $warning("%m generated an overflow.");
			assert(!underflow) else $warning("%m generated an underflow.");
		end
	end

endmodule : rsqrtf_dspfp32

module rsqrtf #(
	int unsigned  SUSTAINABLE_INTERVAL,  // Average II sustained over 12 Cycles
	// Guarantee readiness at II, do not expose delays of arbitrating between iterations:
	//  - by intermittent input delays or
	//  - by revoking readiness.
	bit  STABLE_READINESS = 1,
	bit  FORCE_BEHAVIORAL = 0
)(
	// Global Control
	input	logic  clk,
	input	logic  rst,

	input	logic [31:0]  x,
	input	logic  xvld,
	output	logic  xrdy,

	output	logic [31:0]  r,
	output	logic  rvld
);

	// Isolate input from arbitration between iterations as needed
	uwire [31:0]  xx;
	uwire  xxvld;
	uwire  xxrdy;
	if(STABLE_READINESS && (1 < SUSTAINABLE_INTERVAL) && (SUSTAINABLE_INTERVAL < 9)) begin : genSkid
		queue #(.DATA_WIDTH(32), .ELASTICITY(2)) input_queue (
			.clk, .rst,
			.idat(x), .ivld(xvld), .irdy(xrdy),
			.odat(xx), .ovld(xxvld), .ordy(xxrdy)
		);
	end : genSkid
	else begin : genStraight
		assign	xx = x;
		assign	xxvld = xvld;
		assign	xrdy = xxrdy;
	end : genStraight

	uwire  xsel;  // Feed new input vs. re-feed for interleaving
	uwire [31:0]  afb;
	uwire [31:0]  a = (xsel? 'h5f3759df : afb) - (xsel? xx[31:1] : 0);
	uwire [31:0]  b = { xx[31], xx[30:23]-1, xx[22:0]}; // 0.5*x
	uwire [31:0]  c = $shortrealtobits(1.5);

	case(SUSTAINABLE_INTERVAL)
	0: initial begin
		$error("SUSTAINABLE_INTERVAL must be positive.");
		$finish;
	end
	1: begin : genII1
		localparam int unsigned  DSP_LATENCY = 4;
		localparam int unsigned  LAT = 3*DSP_LATENCY;

		logic  Vld[LAT] = '{ default: 0 };
		logic [31:0]  A[8] = '{ default: 'x };
		uwire [31:0]  p[2];
		always_ff @(posedge clk) begin
			if(rst) begin
				Vld <= '{ default: 0 };
				A <= '{ default: 'x };
			end
			else begin
				Vld <= { xxvld, Vld[0:LAT-2] };
				A <= { a, A[0:6] };
			end
		end
		assign	xsel = 1;
		assign	xxrdy = 1;
		assign	rvld = Vld[LAT-1];

		rsqrtf_dspfp32 DSP0 (
			.clk, .rst,
			.ena('1), .bsel('1), .csel('0),
			.a, .b, .c('x), .d('x),
			.rvld('0), .r(p[0])
		);

		rsqrtf_dspfp32 DSP1 (
			.clk, .rst,
			.ena('1), .bsel('0), .csel('1),
			.a(A[3]), .b('x), .c, .d(p[0]),
			.rvld('0), .r(p[1])
		);

		rsqrtf_dspfp32 DSP2 (
			.clk, .rst,
			.ena('1), .bsel('0), .csel('0),
			.a(A[7]), .b('x), .c('x), .d(p[1]),
			.rvld, .r
		);
	end : genII1
	2: begin : genII2

		logic  Vld[12] = '{ default: 0 };
		always_ff @(posedge clk) begin
			if(rst)  Vld <= '{ default: 0 };
			else     Vld <= { xxrdy && xxvld, Vld[0:10] };
		end

		logic [31:0]  A[8] = '{ default: 'x };
		always_ff @(posedge clk) begin
			if(rst)  A <= '{ default: 'x };
			else     A <= { a, A[0:6] };
		end

		assign	rvld = Vld[11];
		assign	xxrdy = !Vld[7];
		assign	xsel = xxrdy;
		assign	afb = A[7];

		uwire [31:0]  p;  // Second DSP Output
		rsqrtf_dspfp32 DSP0 (
			.clk, .rst,
			.ena('1), .bsel(xsel), .csel('0),
			.a, .b, .c('x), .d(p),
			.rvld, .r
		);

		rsqrtf_dspfp32 DSP1 (
			.clk, .rst,
			.ena('1), .bsel('0), .csel('1),
			.a(A[3]), .b('x), .c, .d(r),
			.rvld('0), .r(p)
		);
	end : genII2
	default: begin : genSharedDSP
		uwire  aload;
		uwire  bsel;
		uwire  csel;

		if(SUSTAINABLE_INTERVAL < 9) begin : genInterleave
			typedef enum logic [1:0] {
				               // bsel/3  csel/1
				IDLE  = 2'b11, //   1       x
				ITER1 = 2'b00, //   0       0
				ITER2 = 2'b01, //   0       1
				ITER3 = 2'b10, //   1       0
				BSEL  = 2'b1x,
				CSEL  = 2'bx1
			} maturity_t;

			maturity_t  Maturity[4] = '{ default: IDLE };
			logic [31:0]  A[4] = '{ default: 'x };
			always_ff @(posedge clk) begin
				if(rst) begin
					Maturity <= '{ default: IDLE };
					A <= '{ default: 'x };
				end
				else begin
					unique casex(Maturity[3])
					ITER1:  Maturity[0] <= ITER2;
					ITER2:  Maturity[0] <= ITER3;
					ITER3,
					IDLE:   Maturity[0] <= xxvld? ITER1 : IDLE;
					endcase
					Maturity[1:3] <= Maturity[0:2];
					A <= { a, A[0:2] };
				end
			end
			assign	bsel = Maturity[3] ==? BSEL;
			assign	csel = Maturity[1] ==? CSEL;
			assign	xsel = bsel;
			assign	xxrdy = bsel;
			assign	rvld = Maturity[3] ==? ITER3;
			assign	aload = 1;
			assign	afb = A[3];
		end : genInterleave
		else if(SUSTAINABLE_INTERVAL < 12) begin : genOverlapped
			logic [3:0]  Cnt  = 8;
			logic [3:0]  RVld = '0;
			uwire  cnt7 = Cnt ==? 4'bx111;
			uwire  cnt8 = Cnt ==? 4'b1xxx;
			always_ff @(posedge clk) begin
				if(rst) begin
					Cnt <= 8;
					RVld <= '0;
				end
				else begin
					Cnt <= Cnt + (!cnt8? 1 : xxvld? 8 : 0);
					RVld <= { cnt7, RVld[3:1] };
				end
			end
			assign	bsel = Cnt[3];
			assign	csel = Cnt[2];
			assign	xsel = 1;
			assign	xxrdy = bsel;
			assign	rvld = RVld[0];
			assign	aload = bsel;
		end : genOverlapped
		else begin : genExclusive
			logic signed [3:0]  Cnt = -1;
			logic  RVld = 0;
			uwire  cnt10 = Cnt ==? 4'b101x;
			always_ff @(posedge clk) begin
				if(rst) begin
					Cnt <= -1;
					RVld <= 0;
				end
				else begin
					Cnt <= Cnt + (cnt10? 'b101 : xxvld || !bsel);
					RVld <= cnt10;
				end
			end
			assign	bsel = &Cnt[3:2];
			assign	csel = Cnt[2];
			assign	xsel = 1;
			assign	xxrdy = bsel;
			assign	rvld = RVld;
			assign	aload = bsel;
		end : genExclusive

		rsqrtf_dspfp32 #(.FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)) DSPFP32_inst (
			.clk, .rst,
			.ena(aload), .bsel, .csel,
			.a, .b, .c, .d(r),
			.rvld, .r
		);
	end : genSharedDSP
	endcase

endmodule : rsqrtf
