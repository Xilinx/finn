/****************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 *
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 * @brief	Fused multiply-add: r = ±a*b ± c in a single DSPFP32.
 *
 * @description
 *	Wraps a single Versal DSPFP32 as a free-running fused multiply-add
 *	unit with a latency of 4 cycles.
 *
 *	Pipeline:
 *	  Stage 1: AREG captures a, FPBREG captures b, FPCREG[1] captures c
 *	  Stage 2: FPMPIPEREG = a × b,  FPCREG[2]
 *	  Stage 3: FPM_PREG = product,   FPCREG[3] = c aligned
 *	  Stage 4: FPA_PREG = c ± product  (output)
 ***************************************************************************/

module fmaf #(
	parameter  OP = "ADD",  // ADD: c+a*b, SUB: c-a*b, SBR: a*b-c
	bit  FORCE_BEHAVIORAL = 0
)(
	input	logic  clk,
	input	logic  rst,

	input	logic [31:0]  a,
	input	logic [31:0]  b,
	input	logic [31:0]  c,
	input	logic  ivld,

	output	logic [31:0]  r,
	output	logic  ovld
);
`default_nettype none
	localparam int unsigned  LATENCY = 4;

	//=== Valid Pipeline ====================================================
	logic [LATENCY-1:0]  Vld = '0;
	always_ff @(posedge clk) begin
		if(rst)  Vld <= '0;
		else     Vld <= { Vld, ivld };
	end
	assign	ovld = Vld[$left(Vld)];

	//=== OPMODE Sign Encoding ==============================================
	uwire [6:0]  opmode;
	case(OP)
	"ADD": assign  opmode = 7'b00_110_01;	// +C + M
	"SBR": assign  opmode = 7'b10_110_01;	// -C + M
	"SUB": assign  opmode = 7'b01_110_01;	// +C - M
	default: initial begin
		$error("%m: Unsupported fused operation %s.", OP);
		$finish;
	end
	endcase

	//=== Simulation-time Warnings ==========================================
	logic  inv;
	logic  ovf;
	logic  unf;

	//=== Datapath ===========================================================
	if(FORCE_BEHAVIORAL) begin : genBehav
		logic [31:0]  A1 = 'x;
		logic [31:0]  B1 = 'x;
		logic [31:0]  C[1:3] = '{ default: 'x };
		logic [31:0]  M[2:3] = '{ default: 'x };
		logic [31:0]  P4 = 'x;
		always_ff @(posedge clk) begin
			A1 <= a;
			B1 <= b;
			C  <= { c, C[1:2] };
			M  <= { $shortrealtobits($bitstoshortreal(A1) * $bitstoshortreal(B1)), M[2] };
			P4 <= $shortrealtobits(
				$bitstoshortreal((opmode[6]<<31) ^ C[3]) +
				$bitstoshortreal((opmode[5]<<31) ^ M[3])
			);
		end
		assign	r = P4;
		always_comb begin
			inv = 0;
			ovf = 0;
			unf = 0;
			if(&P4[30-:8]) begin
				if(|P4[0+:23])  inv = 1;
				else            ovf = 1;
			end
		end

	end : genBehav
	else begin : genDSP
		DSPFP32 #(
			// Feature Control Attributes: Data Path Selection
			.A_FPTYPE("B32"),
			.A_INPUT("DIRECT"),
			.BCASCSEL("B"),
			.B_D_FPTYPE("B32"),
			.B_INPUT("DIRECT"),
			.PCOUTSEL("FPA"),
			.USE_MULT("MULTIPLY"),

			// Programmable Inversion Attributes
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
			.ACASCREG(1),
			.AREG(1),
			.FPA_PREG(1),
			.FPBREG(1),
			.FPCREG(3),
			.FPDREG(0),
			.FPMPIPEREG(1),
			.FPM_PREG(1),
			.FPOPMREG(0),
			.INMODEREG(0),
			.RESET_MODE("SYNC")
		)
		DSPFP32_inst (
			// Cascade outputs
			.ACOUT_EXP(),
			.ACOUT_MAN(),
			.ACOUT_SIGN(),
			.BCOUT_EXP(),
			.BCOUT_MAN(),
			.BCOUT_SIGN(),
			.PCOUT(),

			// Data outputs
			.FPM_INVALID(),
			.FPM_OVERFLOW(),
			.FPM_UNDERFLOW(),
			.FPM_OUT(),
			.FPA_INVALID(inv),
			.FPA_OVERFLOW(ovf),
			.FPA_UNDERFLOW(unf),
			.FPA_OUT(r),

			// Cascade inputs
			.ACIN_EXP('x),
			.ACIN_MAN('x),
			.ACIN_SIGN('x),
			.BCIN_EXP('x),
			.BCIN_MAN('x),
			.BCIN_SIGN('x),
			.PCIN('x),

			// Control inputs
			.CLK(clk),
			.FPINMODE('1),
			.FPOPMODE(opmode),

			// Data inputs
			.A_SIGN(a[31]),
			.A_EXP(a[30:23]),
			.A_MAN(a[22:0]),
			.B_SIGN(b[31]),
			.B_EXP(b[30:23]),
			.B_MAN(b[22:0]),
			.C(c),
			.D_SIGN('0),
			.D_EXP('0),
			.D_MAN('0),

			// Clock Enables — all tied high (free-running)
			.CEA1('1),
			.CEA2('1),
			.CEB('1),
			.CEC('1),
			.CED('0),
			.CEFPA('1),
			.CEFPINMODE('0),
			.CEFPM('1),
			.CEFPMPIPE('1),
			.CEFPOPMODE('0),

			// Resets — inactive
			.ASYNC_RST('0),
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

`default_nettype wire
endmodule : fmaf
