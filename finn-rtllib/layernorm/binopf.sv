/****************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 * @brief	Computes a OP (B_SCALE * b) with OP in +, -, *.
 ***************************************************************************/

module binopf #(
	parameter  OP,	// ADD(a+b), SUB(a-b), SBR(b-a), MUL(a*b)
	shortreal  B_SCALE = 1.0,	// Scale `b` input, must be 1.0 for MUL
	bit  A_MATCH_OP_DELAY = 0,	// Add delay on `a` input equivalent to this op
	                         	// not available for OP == "MUL" || B_SCALE != 1.0
	bit  FORCE_BEHAVIORAL = 0
)(
	input	logic  clk,
	input	logic  rst,

	input	logic [31:0]  a,
	input	logic  avld,
	input	logic [31:0]  b,
	input	logic  bload,

	output	logic [31:0]  r,
	output	logic  rvld
);

	localparam bit  HAVE_SCALE = B_SCALE != 1.0;
	localparam bit  HAVE_ADD   = (OP != "MUL");
	localparam bit  HAVE_MUL   = HAVE_SCALE || !HAVE_ADD;
	localparam int unsigned  LATENCY = HAVE_SCALE || A_MATCH_OP_DELAY? 4 : 2 + HAVE_MUL;
	initial begin
		if(HAVE_SCALE && !HAVE_ADD) begin
			$error("%m: MUL cannot have B_SCALE=%f != 1.0", B_SCALE);
			$finish;
		end
		if(A_MATCH_OP_DELAY && (!HAVE_ADD || HAVE_SCALE)) begin
			if(!HAVE_ADD)  $error("%m: Input delay cannot combine with MUL.");
			if(HAVE_SCALE) $error("%m: Input delay cannot combine with B_SCALE=%f != 1.0", B_SCALE);
			$finish;
		end
	end

	// Valid Input to Output Signalling
	logic [LATENCY-1:0]  Vld = '0;
	always_ff @(posedge  clk) begin
		if(rst)  Vld <= '0;
		else     Vld <= { Vld, avld };
	end
	assign	rvld = Vld[$left(Vld)];

	// Simulation-time Warnings
	logic [1:0]  inv;
	logic [1:0]  ovf;
	logic [1:0]  unf;
	always_ff @(posedge clk) begin
		automatic logic [1:0]  msk = { HAVE_MUL && Vld[2], HAVE_ADD && rvld };
		assert(!(inv & msk)) else $warning("%m generated invalid output.");
		assert(!(ovf & msk)) else $warning("%m generated an overflow.");
		assert(!(unf & msk)) else $warning("%m generated an underflow.");
	end

	// Operation Specifics
	//  MUL  ADD  |  a        SCALE  b
	// -----------+---------------------
	//   0    1   |  C'´´ OP'        D'  // ´ - A_MATCH_OP_DELAY
	//   1    0   |  B'   *''        A'
	//   1    1   |  C''' OP'  B *'' A'
	uwire [6:0]  opmode;
	uwire [31:0]  aa = !HAVE_MUL? 'x : b;
	uwire [31:0]  bb = !HAVE_MUL? 'x : !HAVE_ADD? a : $shortrealtobits(B_SCALE);
	uwire [31:0]  cc = !HAVE_ADD? 'x : a;
	uwire [31:0]  dd =  HAVE_MUL? 'x : b;
	uwire  en_a = HAVE_MUL? bload : 1'b0;
	uwire  en_b = HAVE_MUL && !HAVE_SCALE;
	uwire  en_c = HAVE_ADD;
	uwire  en_d = HAVE_ADD && !HAVE_MUL? bload : 1'b0;
	case(OP)
	"ADD": assign  opmode = { 5'b00_110, !HAVE_MUL, 1'b1 };	// add(C, D:M)
	"SUB": assign  opmode = { 5'b01_110, !HAVE_MUL, 1'b1 };	// sub(C, D:M)
	"SBR": assign  opmode = { 5'b10_110, !HAVE_MUL, 1'b1 };	// sbr(C, D:M)
	"MUL": assign  opmode = 'x;	// mul(A, B)
	default: initial begin
		$error("%m: Unsupported floating-point operation %s", OP);
		$finish;
	end
	endcase
	uwire [31:0]  m;
	uwire [31:0]  s;
	assign	r = HAVE_ADD? s : m;

	localparam int unsigned  CREGS = HAVE_ADD + 2*(HAVE_SCALE || A_MATCH_OP_DELAY);
	if(FORCE_BEHAVIORAL) begin : genBehav
		logic [31:0]  A1 = 'x;
		logic [31:0]  B1 = 'x;
		logic [31:0]  C[1:3] = '{ default: 'x };
		logic [31:0]  D1 = 'x;
		logic [31:0]  M[2:3] = '{ default: 'x };
		logic [31:0]  P4 = 'x;
		always_ff @(posedge clk) begin
			if(rst) begin
				A1 <= 'x;
				B1 <= 'x;
				C <= '{ default: 'x };
				D1 <= 'x;
				M <= '{ default: 'x };
				P4 <= 'x;
			end
			else begin
				if(en_a)  A1 <= aa;
				if(en_b)  B1 <= bb;
				if(en_c)  C <= { cc, C[1:2] };
				if(en_d)  D1 <= dd;
				M <= { $shortrealtobits($bitstoshortreal(A1)*$bitstoshortreal(HAVE_SCALE? bb : B1)), M[2] };
				P4 <= $shortrealtobits(
					$bitstoshortreal((opmode[6]<<31) ^ C[CREGS]) +
					$bitstoshortreal((opmode[5]<<31) ^ (HAVE_MUL? M[3] : D1))
				);
			end
		end
		assign	m = M[3];
		assign	s = P4;
		always_comb begin
			inv = '0;
			ovf = '0;
			unf = '0;
			if(&s[30-:8]) begin
				if(|s[0+:23])  inv[0] = 1;
				else           ovf[0] = 1;
			end
			if(&m[30-:8]) begin
				if(|m[0+:23])  inv[1] = 1;
				else           ovf[1] = 1;
			end
		end

	end : genBehav
	else begin : genDSP
		DSPFP32 #(
			// Feature Control Attributes: Data Path Selection
			.A_FPTYPE("B32"),    // B16, B32
			.A_INPUT("DIRECT"),  // Selects A input source, "DIRECT" (A port) or "CASCADE" (ACIN port)
			.BCASCSEL("B"),      // Selects B cascade out data (B, D).
			.B_D_FPTYPE("B32"),  // B16, B32
			.B_INPUT("DIRECT"),  // Selects B input source, "DIRECT" (B port) or "CASCADE" (BCIN port)
			.PCOUTSEL("FPA"),    // Select PCOUT output cascade of DSPFP32 (FPA, FPM)
			.USE_MULT(HAVE_MUL? "MULTIPLY" : "NONE"), // Select multiplier usage (DYNAMIC, MULTIPLY, NONE)

			// Programmable Inversion Attributes: Specifies built-in programmable inversion on specific pins
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

			// Register Control Attributes: Pipeline Register Configuration
			.ACASCREG(HAVE_MUL),              // Number of pipeline stages between A/ACIN and ACOUT (0-2)
			.AREG(HAVE_MUL),                  // Pipeline stages for A (0-2)
			.FPA_PREG(HAVE_ADD),              // Pipeline stages for FPA output (0-1)
			.FPBREG(HAVE_MUL && !HAVE_SCALE), // Pipeline stages for B inputs (0-1)
			.FPCREG(CREGS),                   // Pipeline stages for C input (0-3)
			.FPDREG(HAVE_ADD && !HAVE_SCALE), // Pipeline stages for D inputs (0-1)
			.FPMPIPEREG(HAVE_MUL),            // Selects the number of FPMPIPE registers (0-1)
			.FPM_PREG(HAVE_MUL),              // Pipeline stages for FPM output (0-1)
			.FPOPMREG(0),                     // Selects the length of the FPOPMODE pipeline (0-3)
			.INMODEREG(0),                    // Selects the number of FPINMODE registers (0-1)
			.RESET_MODE("SYNC")               // Selection of synchronous or asynchronous reset. (ASYNC, SYNC).
		)
		DSPFP32_inst (
			// Cascade outputs: Cascade Ports
			.ACOUT_EXP(),
			.ACOUT_MAN(),
			.ACOUT_SIGN(),
			.BCOUT_EXP(),
			.BCOUT_MAN(),
			.BCOUT_SIGN(),
			.PCOUT(),

			// Data outputs: Data Ports
			.FPM_INVALID(inv[1]),
			.FPM_OVERFLOW(ovf[1]),
			.FPM_UNDERFLOW(unf[1]),
			.FPM_OUT(m),
			.FPA_INVALID(inv[0]),
			.FPA_OVERFLOW(ovf[0]),
			.FPA_UNDERFLOW(unf[0]),
			.FPA_OUT(s),
			// Cascade inputs: Cascade Ports
			.ACIN_EXP('x),
			.ACIN_MAN('x),
			.ACIN_SIGN('x),
			.BCIN_EXP('x),
			.BCIN_MAN('x),
			.BCIN_SIGN('x),
			.PCIN('x),
			// Control inputs: Control Inputs/Status Bits
			.CLK(clk),
			.FPINMODE('1),
			.FPOPMODE(opmode),
			// Data inputs: Data Ports
			.A_SIGN(aa[31]),
			.A_EXP(aa[30:23]),
			.A_MAN(aa[22:0]),
			.B_SIGN(bb[31]),
			.B_EXP(bb[30:23]),
			.B_MAN(bb[22:0]),
			.C(cc),
			.D_SIGN(dd[31]),
			.D_EXP(dd[30:23]),
			.D_MAN(dd[22:0]),
			// Reset/Clock Enable inputs: Reset/Clock Enable Inputs
			.ASYNC_RST('0),
			.CEA1('0),
			.CEA2(en_a),
			.CEB(en_b),
			.CEC(en_c),
			.CED(en_d),
			.CEFPA(HAVE_ADD),
			.CEFPINMODE('0),
			.CEFPM(HAVE_MUL),
			.CEFPMPIPE(HAVE_MUL),
			.CEFPOPMODE('0),
			.RSTA('0),
			.RSTB('0),
			.RSTC('0),
			.RSTD('0),
			.RSTFPA('0),
			.RSTFPINMODE('0),
			.RSTFPM('0),
			.RSTFPMPIPE('0),
			.RSTFPOPMODE('0)
		);
	end : genDSP

endmodule : binopf
