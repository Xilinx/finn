/****************************************************************************
 * Copyright (C) 2025-2026, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Quick reverse square root on DSPFP32 slices, time-rolled as needed.
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
 *	The number of standard Newton-Raphson iterations is selected by
 *	ITERATIONS (1 or 2). The 2nd iteration reduces the maximum relative error
 *	from 1.75e-3 to 4.7e-6.
 *
 *	Implemented compute schedule on DSP slice:
 *	 - 4-stage passes through the DSP pipeline, three per iteration
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
 *	A 2nd iteration appends another three passes re-seeding A with the
 *	refined y and B with the retained x/2.
 *
 *	Structures by SUSTAINABLE_INTERVAL (II) for PASSES = 3*ITERATIONS:
 *
 *	  Structure                     DSPs          ITERATIONS=1  ITERATIONS=2
 *	  ─────────────────────────────────────────────────────────────────────
 *	  Unrolled ring chain           PASSES        1             1
 *	  Paired (legacy genII2)        2             2             -
 *	  Ring chain                    ⌈PASSES/II⌉   -             2 (3 DSPs)
 *	                                                            3..5 (2 DSPs)
 *	  Single ring (interleaved)     1             3..8          6..20
 *	  Overlapped                    1             9..11         21..23
 *	  Exclusive                     1             12..          24..
 *
 *	A ring is a DSP executing a contiguous share of the passes of up to four
 *	interleaved jobs, which recirculate through it every four cycles. A ring
 *	executing k passes sustains any II >= k, i.e. it accepts inputs spaced by
 *	at least k cycles without ever exhausting its input skid. Chained
 *	rings are fed in lock step by their predecessor so that a later ring,
 *	executing at most as many passes, always finds the handed-over slot free.
 *
 *	In all structures, a job leaves the DSP pipeline 4*PASSES cycles after its
 *	acceptance. The input skid of interleaving structures adds 2 cycles plus,
 *	for inputs spaced by II, at most 3 (paired: 7) cycles waiting for a slot.
 ***************************************************************************/

// Local DSP instantiation wrapper.
module rsqrtf_dspfp32 #(
	bit  FORCE_BEHAVIORAL = 0
)(
	input  logic         clk,
	input  logic         rst,

	input  logic         ena,   // A register load
	input  logic         bena,  // B register load
	input  logic         bsel,
	input  logic         csel,
	input  logic [31:0]  a,
	input  logic [31:0]  b,
	input  logic [31:0]  c,
	input  logic [31:0]  d,

	input  logic         rvld,
	output logic [31:0]  r
);
`default_nettype none

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
			if(bena)  B1 <= b;
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
			.CEB(bena), .CEC('0), .CED('1),
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

`default_nettype wire
endmodule : rsqrtf_dspfp32


// DSP slice executing passes [FIRST, FIRST+PASSES) of up to four interleaved jobs.
module rsqrtf_ring #(
	int unsigned  FIRST,   // first pass executed in this ring
	int unsigned  PASSES,  // number of passes executed in this ring
	bit  FORCE_BEHAVIORAL = 0
)(
	input	logic  clk,
	input	logic  rst,

	// Job Entry
	input	logic [31:0]  x,  // operand
	input	logic [31:0]  y,  // current approximation  [FIRST > 0]
	input	logic [31:0]  p,  // result of pass FIRST-1  [FIRST > 0]
	input	logic  ivld,
	output	logic  irdy,

	// Job Exit
	output	logic [31:0]  ox,
	output	logic [31:0]  oy,
	output	logic [31:0]  r,  // result of pass FIRST+PASSES-1
	output	logic  ovld
);
`default_nettype none

	localparam logic [31:0]  MAGIC = 'h5f3759df;
	localparam logic [31:0]  C_1_5 = $shortrealtobits(1.5);

	// Passes starting a Newton-Raphson iteration, which need x/2
	function bit HAS_SEED_PASS(input int unsigned  first, input int unsigned  last);
		for(int unsigned  i = first; i <= last; i++) begin
			if(i%3 == 0)  return  1;
		end
		return  0;
	endfunction : HAS_SEED_PASS
	localparam bit  USE_B  = HAS_SEED_PASS(FIRST,   FIRST+PASSES-1);  // at all
	localparam bit  RESEED = HAS_SEED_PASS(FIRST+1, FIRST+PASSES-1);  // within the ring

	//=======================================================================
	// Slot Ring tracking the Pass (relative to FIRST) presented to the DSP
	typedef logic [$clog2(PASSES+1)-1:0]  pass_t;
	localparam pass_t  LAST = PASSES-1;
	localparam pass_t  IDLE = '1;

	pass_t  Pass[4] = '{ default: IDLE };	// Pass[3]: result emerging from DSP
	uwire  free = (PASSES == 1) || (Pass[3] == LAST) || (Pass[3] == IDLE);
	uwire pass_t  pass = free? 0 : Pass[3] + 1;
	always_ff @(posedge clk) begin
		if(rst)  Pass <= '{ default: IDLE };
		else     Pass <= { (free && !ivld)? IDLE : pass, Pass[0:2] };
	end
	assign	irdy = free;
	assign	ovld = Pass[3] == LAST;

	// Pass Operation by (FIRST + pass) % 3:
	//	- 0: x/2 * y           (B selected)
	//	- 1: 1.5 - y * <prev>  (C subtracted 2 cycles later)
	//	- 2:       y * <prev>
	uwire  bsel = (FIRST + pass) % 3 == 0;
	uwire  csel = (FIRST + ((PASSES == 1)? 0 : Pass[1])) % 3 == 1;

	//=======================================================================
	// Operand Selection with Slot Retention of x and y
	logic [31:0]  X[4] = '{ default: 'x };
	logic [31:0]  A[4] = '{ default: 'x };
	uwire [31:0]  xs = free? x : X[3];
	uwire [31:0]  xb = RESEED? xs : x;
	uwire [31:0]  b = USE_B? { xb[31], xb[30:23]-1, xb[22:0] } : 'x;  // x/2
	uwire [31:0]  yfb = (RESEED && bsel)? r : A[3];
	uwire [31:0]  a;
	if(FIRST == 0)  assign  a = (free? MAGIC : yfb) - (free? x[31:1] : 0);
	else            assign  a = free? ((FIRST%3 == 0)? p : y) : yfb;
	uwire [31:0]  d = ((FIRST == 0) || !free)? r : p;
	always_ff @(posedge clk) begin
		if(rst) begin
			X <= '{ default: 'x };
			A <= '{ default: 'x };
		end
		else begin
			X <= { xs, X[0:2] };
			A <= { a, A[0:2] };
		end
	end
	assign	ox = X[3];
	assign	oy = A[3];

	rsqrtf_dspfp32 #(.FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)) dsp (
		.clk, .rst,
		.ena('1), .bena('1), .bsel, .csel,
		.a, .b, .c(C_1_5), .d,
		.rvld(ovld), .r
	);

`default_nettype wire
endmodule : rsqrtf_ring

module rsqrtf #(
	int unsigned  SUSTAINABLE_INTERVAL,  // Minimum spacing of inputs in cycles
	int unsigned  ITERATIONS = 1,        // Newton-Raphson iterations: 1 | 2
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
`default_nettype none

	//=======================================================================
	// Structure Selection
	localparam int unsigned  II = SUSTAINABLE_INTERVAL;
	localparam int unsigned  PASSES = 3*ITERATIONS;

	localparam bit  PAIRED = (ITERATIONS == 1) && (II == 2);  // two DSPs: passes 0+2 and 1
	localparam bit  SEQUENTIAL = II >= 4*PASSES - 3;          // single job in single DSP
	localparam int unsigned  STAGES = II? (PASSES + II - 1) / II : 1;  // ring chain
	localparam int unsigned  STAGE_PASSES = (PASSES + STAGES - 1) / STAGES;

	initial begin
		if((ITERATIONS < 1) || (2 < ITERATIONS)) begin
			$error("%m: ITERATIONS(%0d) must be 1 or 2.", ITERATIONS);
			$finish;
		end
		if(II == 0) begin
			$error("%m: SUSTAINABLE_INTERVAL must be positive.");
			$finish;
		end
	end

	//=======================================================================
	// Isolate input from arbitration between iterations as needed
	uwire [31:0]  xx;
	uwire  xxvld;
	uwire  xxrdy;
	if(STABLE_READINESS && !SEQUENTIAL && (PAIRED || (STAGE_PASSES > 1))) begin : genSkid
		fifo #(.DATA_WIDTH(32), .DEPTH(2)) input_queue (
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

	//=======================================================================
	// Compute
	localparam logic [31:0]  MAGIC = 'h5f3759df;
	uwire [31:0]  b = { xx[31], xx[30:23]-1, xx[22:0]}; // 0.5*x
	uwire [31:0]  c = $shortrealtobits(1.5);

	if(PAIRED) begin : genII2
		uwire  xsel;  // Feed new input vs. re-feed for interleaving
		uwire [31:0]  afb;
		uwire [31:0]  a = (xsel? MAGIC : afb) - (xsel? xx[31:1] : 0);

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
		rsqrtf_dspfp32 #(.FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)) DSP0 (
			.clk, .rst,
			.ena('1), .bena('1), .bsel(xsel), .csel('0),
			.a, .b, .c('x), .d(p),
			.rvld, .r
		);

		rsqrtf_dspfp32 #(.FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)) DSP1 (
			.clk, .rst,
			.ena('1), .bena('1), .bsel('0), .csel('1),
			.a(A[3]), .b('x), .c, .d(r),
			.rvld('0), .r(p)
		);
	end : genII2
	else if(!SEQUENTIAL) begin : genRings
		// Stage Interconnect: [s] feeds ring #s
		uwire [31:0]  sx[STAGES+1];
		uwire [31:0]  sy[STAGES+1];
		uwire [31:0]  sp[STAGES+1];
		uwire  svld[STAGES+1];
		assign	sx[0] = xx;
		assign	sy[0] = 'x;
		assign	sp[0] = 'x;
		assign	svld[0] = xxvld;

		for(genvar  s = 0; s < STAGES; s++) begin : genStages
			localparam int unsigned  FIRST = s * STAGE_PASSES;
			localparam int unsigned  NUM = (PASSES - FIRST < STAGE_PASSES)? PASSES - FIRST : STAGE_PASSES;

			uwire  srdy;
			rsqrtf_ring #(.FIRST(FIRST), .PASSES(NUM), .FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)) ring (
				.clk, .rst,
				.x(sx[s]), .y(sy[s]), .p(sp[s]), .ivld(svld[s]), .irdy(srdy),
				.ox(sx[s+1]), .oy(sy[s+1]), .r(sp[s+1]), .ovld(svld[s+1])
			);

			if(s == 0)  assign  xxrdy = srdy;
			else begin : genHandoverCheck
				always_ff @(posedge clk) begin
					assert(rst || !svld[s] || srdy) else begin
						$error("%m: Ring #%0d occupied on job hand-over.", s);
						$stop;
					end
				end
			end : genHandoverCheck
		end : genStages

		assign	r = sp[STAGES];
		assign	rvld = svld[STAGES];
	end : genRings
	else begin : genSequential
		// Cnt = cycles since job acceptance - 1 while busy; passes every 4 cycles:
		//	- re-seed a Newton-Raphson iteration (Cnt%12 == 11) and
		//	- subtract from 1.5 (Cnt%12 in [4:7]).
		uwire  idle;
		uwire  reseed;
		uwire  csel;
		if(II < 4*PASSES) begin : genOverlapped
			localparam int unsigned  BUSY = 4*PASSES - 4;  // last pass issued
			logic [$clog2(BUSY+1)-1:0]  Cnt = BUSY;   // 0, 1, ..., BUSY(=idle)
			logic [3:0]  RVld = '0;
			uwire  last = Cnt == BUSY-1;
			always_ff @(posedge clk) begin
				if(rst) begin
					Cnt <= BUSY;
					RVld <= '0;
				end
				else begin
					Cnt <= Cnt + (!idle? 1 : xxvld? -BUSY : 0);
					RVld <= { last, RVld[3:1] };
				end
			end
			assign	idle = Cnt == BUSY;
			assign	reseed = Cnt%12 == 11;
			assign	csel = (Cnt%12)/4 == 1;
			assign	rvld = RVld[0];
		end : genOverlapped
		else begin : genExclusive
			localparam int unsigned  BUSY = 4*PASSES - 1;  // last result emerging
			typedef logic [$clog2(BUSY+1)-1:0]  cnt_t;
			localparam cnt_t  IDLE = '1;
			cnt_t  Cnt = IDLE;   // 0, 1, ..., BUSY-1, IDLE
			logic  RVld = 0;
			uwire  last = Cnt == BUSY-1;
			always_ff @(posedge clk) begin
				if(rst) begin
					Cnt <= IDLE;
					RVld <= 0;
				end
				else begin
					Cnt <= Cnt + (last? IDLE-(BUSY-1) : xxvld || !idle);
					RVld <= last;
				end
			end
			assign	idle = Cnt == IDLE;
			assign	reseed = Cnt%12 == 11;
			assign	csel = (Cnt%12)/4 == 1;
			assign	rvld = RVld;
		end : genExclusive
		assign	xxrdy = idle;

		// A and B registers hold y and x/2 between their loads.
		uwire  bsel = idle || reseed;
		uwire  xsel = (ITERATIONS == 1) || idle;
		uwire [31:0]  a = (xsel? MAGIC : r) - (xsel? xx[31:1] : 0);
		rsqrtf_dspfp32 #(.FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)) DSPFP32_inst (
			.clk, .rst,
			.ena(bsel), .bena(idle), .bsel, .csel,
			.a, .b, .c, .d(r),
			.rvld, .r
		);
	end : genSequential

`default_nettype wire
endmodule : rsqrtf
