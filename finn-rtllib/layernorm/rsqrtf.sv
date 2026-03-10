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
 ***************************************************************************/

module rsqrtf #(
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

	logic signed [4:0]  Cnt = -1;	// 10, 9, ..., 0, -1
	logic  Vld = 0;
	always_ff @(posedge clk) begin
		if(rst) begin
			Cnt <= -1;
			Vld <= 0;
		end
		else begin
			Cnt <= Cnt + (xrdy && xvld? 11 : Cnt[4]? 0 : -1);
			Vld <= !Cnt[3:0];
		end
	end
	assign	xrdy = Cnt[4];
	assign	rvld = Vld;

	uwire  bsel = Cnt[3];	// B rather than D to MUL
	uwire  csel = Cnt[2];	// C rather than 0 to ADD
	uwire [31:0]  a = ('h5f3759df - x[31:1]);
	uwire [31:0]  b = { x[31], x[30:23]-1, x[22:0]}; // 0.5*x
	uwire [31:0]  c = $shortrealtobits(1.5);
	uwire [31:0]  d = r;

	logic [1:0]  inv;
	logic [1:0]  ovf;
	logic [1:0]  unf;
	always_ff @(posedge clk) begin
		automatic logic [1:0]  msk = { &Cnt[1:0] && (!xrdy || rvld), Cnt[1:0] == 0 };
		assert(!(inv & msk)) else $warning("%m generated invalid output.");
		assert(!(ovf & msk)) else $warning("%m generated an overflow.");
		assert(!(unf & msk)) else $warning("%m generated an underflow.");
	end

	if(FORCE_BEHAVIORAL) begin : genBehav
		logic [31:0]  A1 = 'x;
		logic [31:0]  B1 = 'x;
		logic [31:0]  D1 = 'x;
		logic  BSel1 = 'x;
		logic  CSel3 = 'x;
		logic [31:0]  M[2:3] = '{ default: 'x };
		logic [31:0]  P4 = 'x;
		always_ff @(posedge clk) begin
			if(rst) begin
			end
			else begin
				if(xrdy)  A1 <= a;
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
		end
		assign	r = P4;
		always_comb begin
			inv = '0;
			ovf = '0;
			unf = '0;
			if(&r[30-:8]) begin
				if(|r[0+:23])  inv[1] = 1;
				else           ovf[1] = 1;
			end
			if(&M[3][30-:8]) begin
				if(|M[3][0+:23])  inv[0] = 1;
				else              ovf[0] = 1;
			end
		end
	end : genBehav
	else begin : genDSP
		DSPFP32 #(
			// Feature Control Attributes: Data Path Selection
			.A_FPTYPE("B32"),      // B16, B32
			.A_INPUT("DIRECT"),    // Selects A input source, "DIRECT" (A port) or "CASCADE" (ACIN port)
			.BCASCSEL("B"),        // Selects B cascade out data (B, D).
			.B_D_FPTYPE("B32"),    // B16, B32
			.B_INPUT("DIRECT"),    // Selects B input source, "DIRECT" (B port) or "CASCADE" (BCIN port)
			.PCOUTSEL("FPA"),      // Select PCOUT output cascade of DSPFP32 (FPA, FPM)
			.USE_MULT("MULTIPLY"), // Select multiplier usage (DYNAMIC, MULTIPLY, NONE)

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
			.ACASCREG(1),                      // Number of pipeline stages between A/ACIN and ACOUT (0-2)
			.AREG(1),                          // Pipeline stages for A (0-2)
			.FPA_PREG(1),                      // Pipeline stages for FPA output (0-1)
			.FPBREG(1),                        // Pipeline stages for B inputs (0-1)
			.FPCREG(0),                        // Pipeline stages for C input (0-3)
			.FPDREG(1),                        // Pipeline stages for D inputs (0-1)
			.FPMPIPEREG(1),                    // Selects the number of FPMPIPE registers (0-1)
			.FPM_PREG(1),                      // Pipeline stages for FPM output (0-1)
			.FPOPMREG(1),                      // Selects the length of the FPOPMODE pipeline (0-3)
			.INMODEREG(1),                     // Selects the number of FPINMODE registers (0-1)
			.RESET_MODE("SYNC")                // Selection of synchronous or asynchronous reset. (ASYNC, SYNC).
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
			.FPM_INVALID(inv[0]),
			.FPM_OVERFLOW(ovf[0]),
			.FPM_UNDERFLOW(unf[0]),
			.FPM_OUT(),
			.FPA_INVALID(inv[1]),
			.FPA_OVERFLOW(ovf[1]),
			.FPA_UNDERFLOW(unf[1]),
			.FPA_OUT(r),
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
			.FPINMODE(bsel),
			.FPOPMODE({
				// (csel? sub : add)(csel? C : 0, M)
				{ 1'b0, csel }, { csel, 2'b10 }, 2'b01
			}),
			// Data inputs: Data Ports
			.A_SIGN(a[31]),
			.A_EXP(a[30:23]),
			.A_MAN(a[22:0]),
			.B_SIGN(b[31]),
			.B_EXP(b[30:23]),
			.B_MAN(b[22:0]),
			.C(c),
			.D_SIGN(d[31]),
			.D_EXP(d[30:23]),
			.D_MAN(d[22:0]),
			// Reset/Clock Enable inputs: Reset/Clock Enable Inputs
			.ASYNC_RST('0),
			.CEA1('0),
			.CEA2(xrdy),
			.CEB('1),
			.CEC('0),
			.CED('1),
			.CEFPA('1),
			.CEFPINMODE('1),
			.CEFPM('1),
			.CEFPMPIPE('1),
			.CEFPOPMODE('1),

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

endmodule : rsqrtf
