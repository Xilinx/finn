/****************************************************************************
 * Copyright (C) 2026, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	FP32 piecewise polynomial activation on DSPFP32.
 * @author	Shane Fleming <shane.fleming@amd.com>
 *
 * @description
 *	Supports GELU, SiLU, Sigmoid, and Tanh via `parameter string FUNC`.
 *
 *	Approximated by piecewise degree-D polynomials over segments defined
 *	by FP32 bit-extraction, where D = DEGREE from pwpolyf_pkg.
 *	Evaluated via Horner's method on a chain
 *	of D DSPFP32 instances, each computing FMA: out = C + A*B.
 *
 *	Horner (degree D): y = a_0 + x*(a_1 + x*(... + x*a_D))
 *	  Stage 0: out = a_{D-1} + a_D * x
 *	  Stage j: out = a_{D-1-j} + prev * x   (j = 1 .. D-1)
 *
 *	Clamping for |x| >= 8 (5 octaves):
 *	  GELU/SiLU:  neg -> 0,   pos -> x  (pass-through)
 *	  Sigmoid:    neg -> 0,   pos -> 1.0
 *	  Tanh:       neg -> -1,  pos -> 1.0
 *
 *	Latency: D * DSP_LAT cycles (D DSP stages x 4 cycles each).  II=1.
 ***************************************************************************/

//===----------------------------------------------------------------------===//
// Single DSPFP32 FMA wrapper: r = c + a * b
//===----------------------------------------------------------------------===//
module pwpolyf_dspfp32 (
	input  logic         clk,
	input  logic         rst,

	input  logic [31:0]  a,
	input  logic [31:0]  b,
	input  logic [31:0]  c,

	output logic [31:0]  r,
	input  logic         rvld
);

	// FMA opmode: FPA_OUT = C + A*B
	//  FPOPMODE[6:5] = 00 (no sign flip on C or M)
	//  FPOPMODE[4:2] = 110 (select C for W mux, M for Z mux -- add path)
	//  FPOPMODE[1:0] = 01 (FP mode enable)
	localparam logic [6:0]  MODE_FMA = 7'b00_110_01;

	uwire  invalid;
	uwire  overflow;
	uwire  underflow;

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
		.FPCREG(3),          // C needs 3 pipeline stages to align with M output
		.FPDREG(0),
		.FPMPIPEREG(1),
		.FPM_PREG(1),
		.FPOPMREG(0),
		.INMODEREG(0),
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
		.CLK(clk),
		.FPINMODE('1),       // Select B path (not D)
		.FPOPMODE(MODE_FMA),
		.A_SIGN(a[31]), .A_EXP(a[30:23]), .A_MAN(a[22:0]),
		.B_SIGN(b[31]), .B_EXP(b[30:23]), .B_MAN(b[22:0]),
		.C(c),
		.D_SIGN('x), .D_EXP('x), .D_MAN('x),
		.ASYNC_RST('0),
		.CEA1('0), .CEA2('1),
		.CEB('1), .CEC('1), .CED('0),
		.CEFPA('1), .CEFPINMODE('0), .CEFPM('1), .CEFPMPIPE('1), .CEFPOPMODE('0),
		.RSTA('0), .RSTB('0), .RSTC('0), .RSTD('0),
		.RSTFPA('0), .RSTFPINMODE('0), .RSTFPM('0), .RSTFPMPIPE('0), .RSTFPOPMODE('0)
	);

	// Simulation-time warnings
	always_ff @(posedge clk) begin
		if(!rst && rvld) begin
			assert(!invalid) else $warning("%m generated invalid output.");
			assert(!overflow) else $warning("%m generated an overflow.");
			assert(!underflow) else $warning("%m generated an underflow.");
		end
	end

endmodule : pwpolyf_dspfp32

//===----------------------------------------------------------------------===//
// Full PE-wide streaming activation with piecewise polynomial approximation.
// Degree D derived from DEGREE in pwpolyf_pkg.
//===----------------------------------------------------------------------===//
module pwpolyf #(
	int unsigned  PE = 1,
	string  FUNC = "gelu"
)(
	// Global Control
	input	logic  clk,
	input	logic  rst,

	// Input Stream - PE elements wide
	input	logic [PE-1:0][31:0]  xdat,
	input	logic  xvld,
	output	logic  xrdy,

	// Output Stream - PE elements wide
	output	logic [PE-1:0][31:0]  ydat,
	output	logic  yvld,
	input	logic  yrdy
);

	import pwpolyf_pkg::*;

	localparam int unsigned  NUM_SUBS    = 1 << K;
	localparam int unsigned  DSP_LAT     = 4;
	localparam int unsigned  LATENCY     = DEGREE * DSP_LAT;

	initial begin
		assert(DEGREE >= 1) else begin
			$error("%m: DEGREE must be >= 1.");
			$finish;
		end
		assert(FUNC == "gelu" || FUNC == "silu" || FUNC == "sigmoid" || FUNC == "tanh") else begin
			$error("%m: Unsupported FUNC=\"%s\". Must be gelu|silu|sigmoid|tanh.", FUNC);
			$finish;
		end
	end

	//=== Per-activation configuration =======================================
	localparam func_cfg_t  CFG =
		FUNC == "gelu"    ? GELU :
		FUNC == "silu"    ? SILU :
		FUNC == "sigmoid" ? SIGMOID :
		                    TANH;

	//=== Clamping exponent threshold =========================================
	localparam int unsigned  EXP_CLAMP = 130;  // |x| >= 8.0

	//=== Input Sidestep Register =============================================
	typedef logic [PE-1:0][31:0]  fp_vec_t;

	uwire  take;

	typedef struct {
		fp_vec_t  val;
		logic     rdy;
	} ibuf_t;
	ibuf_t  Ibuf = '{ val: 'x, rdy: '1 };
	always_ff @(posedge clk) begin
		if(rst)
			Ibuf <= '{ val: 'x, rdy: '1 };
		else begin
			if(Ibuf.rdy)  Ibuf.val <= xdat;
			Ibuf.rdy <= (Ibuf.rdy && !xvld) || take;
		end
	end
	assign	xrdy = Ibuf.rdy;
	uwire fp_vec_t  x_cur = Ibuf.rdy? xdat : Ibuf.val;

	//=== Credit-based Operation Issue ========================================
	localparam int unsigned  CREDIT = LATENCY + 3;  // pipeline + sidestep + queue read
	logic signed [$clog2(CREDIT):0]  Credit = -CREDIT;
	uwire  give = yvld && yrdy;
	assign	take = (xvld || !xrdy) && Credit[$left(Credit)];
	always_ff @(posedge clk) begin
		if(rst)  Credit <= -CREDIT;
		else     Credit <= Credit + (give == take? 0 : give? -1 : 1);
	end

	//=== Per-PE Compute Pipeline =============================================
	uwire fp_vec_t  r;
	uwire [PE-1:0]  rvld_vec;
	uwire  rvld;

	for(genvar  pe = 0; pe < PE; pe++) begin : gen_pe
		uwire [31:0]  xi = x_cur[pe];

		//--- Segment selector (combinational) --------------------------------
		uwire         sign = xi[31];
		uwire [7:0]   exp_bits = xi[30:23];
		uwire [K-1:0] sub  = xi[22:23-K];

		// Octave index: exp 125->0, 126->1, 127->2, 128->3, 129->4
		uwire [2:0]  octave = exp_bits - 8'd125;

		// Classify
		uwire  is_near_zero = (exp_bits < 8'd125);
		uwire  is_pos_clamp = !sign && (exp_bits >= EXP_CLAMP);
		uwire  is_neg_clamp =  sign && (exp_bits >= EXP_CLAMP);

		// Segment index for ROM lookup
		uwire [6:0]  seg_idx;
		if(1) begin : blk_seg_idx
			uwire [6:0]  pos_idx = 7'd1 + {1'b0, octave, sub};
			uwire [6:0]  neg_idx = 7'(7'd1 + NUM_SUBS * NUM_OCTAVES) + {1'b0, octave, sub};
			assign	seg_idx = is_near_zero? 7'd0 :
			                  sign? neg_idx : pos_idx;
		end : blk_seg_idx

		//--- Horner chain: DEGREE stages of pwpolyf_dspfp32 ------------------
		// Stage 0: s[0] = coeff[DEGREE-1] + coeff[DEGREE] * x
		// Stage j: s[j] = coeff[DEGREE-1-j] + s[j-1] * x_delayed
		// Total: DEGREE * DSP_LAT cycles

		// Valid pipeline
		logic [LATENCY-1:0]  Vld = '0;
		always_ff @(posedge clk) begin
			if(rst)  Vld <= '0;
			else     Vld <= { Vld[$left(Vld)-1:0], take };
		end
		assign	rvld_vec[pe] = Vld[$left(Vld)];

		// Delay x for DSP B inputs and pass-through clamp
		logic [31:0]  XDly[LATENCY] = '{default: 'x};
		always_ff @(posedge clk) begin
			XDly[0] <= xi;
			for(int i = 1; i < LATENCY; i++)
				XDly[i] <= XDly[i-1];
		end

		// DSP chain
		uwire [31:0]  s[DEGREE];

		for(genvar  j = 0; j < DEGREE; j++) begin : genDSP
			uwire [31:0]  dsp_a = (j == 0)? CFG.coeffs[seg_idx][DEGREE] : s[j-1];
			uwire [31:0]  dsp_b = (j == 0)? xi : XDly[j*DSP_LAT - 1];

			// C input: coeff[DEGREE-1-j] delayed by j*DSP_LAT cycles
			logic [31:0]  dsp_c;
			if(j == 0) begin : genCdir
				assign  dsp_c = CFG.coeffs[seg_idx][DEGREE-1];
			end : genCdir
			else begin : genCdly
				logic [31:0]  CDly[j*DSP_LAT] = '{default: 'x};
				always_ff @(posedge clk) begin
					CDly[0] <= CFG.coeffs[seg_idx][DEGREE-1-j];
					for(int i = 1; i < j*DSP_LAT; i++)
						CDly[i] <= CDly[i-1];
				end
				assign  dsp_c = CDly[j*DSP_LAT - 1];
			end : genCdly

			pwpolyf_dspfp32  dsp (
				.clk, .rst,
				.a(dsp_a), .b(dsp_b), .c(dsp_c),
				.r(s[j]), .rvld(Vld[(j+1)*DSP_LAT - 1])
			);
		end : genDSP

		//--- Clamp mux -------------------------------------------------------
		logic [LATENCY-1:0]  NegClamp = '0;
		logic [LATENCY-1:0]  PosClamp = '0;
		always_ff @(posedge clk) begin
			if(rst) begin
				NegClamp <= '0;
				PosClamp <= '0;
			end
			else begin
				NegClamp <= { NegClamp[$left(NegClamp)-1:0], is_neg_clamp };
				PosClamp <= { PosClamp[$left(PosClamp)-1:0], is_pos_clamp };
			end
		end

		// Output mux
		assign	r[pe] = NegClamp[$left(NegClamp)]? CFG.neg_clamp :
		                 PosClamp[$left(PosClamp)]? (CFG.pos_passthrough? XDly[LATENCY-1] : CFG.pos_clamp) :
		                 s[DEGREE-1];

	end : gen_pe

	// All PE results should be valid simultaneously
	assign	rvld = rvld_vec[0];
	always_ff @(posedge clk) begin
		assert(rvld_vec == {(PE){rvld}}) else begin
			$error("%m: Inconsistent output valid indications.");
			$stop;
		end
	end

	//=== Credit-backing Elastic Output Queue =================================
	uwire  rrdy;
	fifo #(.DATA_WIDTH($bits(fp_vec_t)), .DEPTH(CREDIT)) obuf (
		.clk, .rst,
		.idat(r), .ivld(rvld), .irdy(rrdy),
		.odat(ydat), .ovld(yvld), .ordy(yrdy)
	);
	always_ff @(posedge clk) begin
		assert(rrdy || !rvld) else begin
			$error("%m: Result queue overrun.");
			$stop;
		end
	end

endmodule : pwpolyf
