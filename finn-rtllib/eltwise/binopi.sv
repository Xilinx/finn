/****************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Integer binary operation: a OP b.
 * @author	Shane Fleming <shane.fleming@amd.com>
 ***************************************************************************/

module binopi #(
	parameter  OP,	// ADD(a+b), SUB(a-b), SBR(b-a), MUL(a*b)
	int unsigned  WIDTH,
	bit  SIGNED = 0,
	bit  FORCE_BEHAVIORAL = 0,

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

	localparam int unsigned  LATENCY = IS_MUL? 2 : 1;

	// DSP58 INT24 multiply: 27-bit A × 24-bit B (signed).
	// Unsigned operands need a zero MSB, reducing usable width by 1.
	localparam int unsigned  DSP_A_BITS = SIGNED? 27 : 26;
	localparam int unsigned  DSP_B_BITS = SIGNED? 24 : 23;
	initial begin
		if(OP != "ADD" && OP != "SUB" && OP != "SBR" && OP != "MUL") begin
			$error("%m: Unsupported integer operation %s", OP);
			$finish;
		end
		if(IS_MUL && !FORCE_BEHAVIORAL && WIDTH > DSP_B_BITS) begin
			$error("%m: WIDTH=%0d exceeds single DSP58 capacity (%0d bits %s)",
				WIDTH, DSP_B_BITS, SIGNED? "signed" : "unsigned");
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

	//=== Compute =========================================================
	if(IS_MUL) begin : genMul

		if(!FORCE_BEHAVIORAL) begin : genDSP
			// DSP58: AREG=1, BREG=1, MREG=1, PREG=0 → 2-cycle latency
			// OPMODE = 9'b00_000_01_01: P = 0 + 0 + M (multiply pass-through)
			uwire [57:0]  pp;
			DSP58 #(
				.AMULTSEL("A"),
				.A_INPUT("DIRECT"),
				.BMULTSEL("B"),
				.B_INPUT("DIRECT"),
				.DSP_MODE("INT24"),
				.PREADDINSEL("A"),
				.RND('0),
				.USE_MULT("MULTIPLY"),
				.USE_SIMD("ONE58"),
				.USE_WIDEXOR("FALSE"),
				.XORSIMD("XOR24_34_58_116"),

				.AUTORESET_PATDET("NO_RESET"),
				.AUTORESET_PRIORITY("RESET"),
				.MASK('1),
				.PATTERN('0),
				.SEL_MASK("MASK"),
				.SEL_PATTERN("PATTERN"),
				.USE_PATTERN_DETECT("NO_PATDET"),

				.IS_ALUMODE_INVERTED('0),
				.IS_CARRYIN_INVERTED('0),
				.IS_CLK_INVERTED('0),
				.IS_INMODE_INVERTED('0),
				.IS_NEGATE_INVERTED('0),
				.IS_OPMODE_INVERTED('0),
				.IS_RSTALLCARRYIN_INVERTED('0),
				.IS_RSTALUMODE_INVERTED('0),
				.IS_RSTA_INVERTED('0),
				.IS_RSTB_INVERTED('0),
				.IS_RSTCTRL_INVERTED('0),
				.IS_RSTC_INVERTED('0),
				.IS_RSTD_INVERTED('0),
				.IS_RSTINMODE_INVERTED('0),
				.IS_RSTM_INVERTED('0),
				.IS_RSTP_INVERTED('0),

				.ACASCREG(1),
				.ADREG(0),
				.ALUMODEREG(0),
				.AREG(1),
				.BCASCREG(1),
				.BREG(1),
				.CARRYINREG(0),
				.CARRYINSELREG(0),
				.CREG(0),
				.DREG(0),
				.INMODEREG(0),
				.MREG(1),
				.OPMODEREG(0),
				.PREG(0),
				.RESET_MODE("SYNC")
			) dsp (
				.ACOUT(),
				.BCOUT(),
				.CARRYCASCOUT(),
				.MULTSIGNOUT(),
				.PCOUT(),

				.OVERFLOW(),
				.PATTERNBDETECT(),
				.PATTERNDETECT(),
				.UNDERFLOW(),

				.CARRYOUT(),
				.P(pp),
				.XOROUT(),

				.ACIN('x),
				.BCIN('x),
				.CARRYCASCIN('x),
				.MULTSIGNIN('x),
				.PCIN('x),

				.CLK(clk),
				.ALUMODE(4'h0),
				.CARRYINSEL('0),
				.INMODE('0),
				.NEGATE('0),
				.OPMODE(9'b00_000_01_01),

				.A(SIGNED? { {(34-WIDTH){a[WIDTH-1]}}, a } : { {(34-WIDTH){1'b0}}, a }),
				.B(SIGNED? { {(24-WIDTH){b[WIDTH-1]}}, b } : { {(24-WIDTH){1'b0}}, b }),
				.C('x),
				.CARRYIN('0),
				.D('x),

				.ASYNC_RST('0),
				.CEA1('0),
				.CEA2('1),
				.CEAD('0),
				.CEALUMODE('0),
				.CEB1('0),
				.CEB2(bload),
				.CEC('0),
				.CECARRYIN('0),
				.CECTRL('0),
				.CED('0),
				.CEINMODE('0),
				.CEM('1),
				.CEP('0),
				.RSTA(rst),
				.RSTALLCARRYIN('0),
				.RSTALUMODE('0),
				.RSTB(rst),
				.RSTC('0),
				.RSTCTRL('0),
				.RSTD('0),
				.RSTINMODE('0),
				.RSTM(rst),
				.RSTP('0)
			);
			assign	r = pp[O_WIDTH-1:0];
		end : genDSP
		else begin : genBehav
			logic [WIDTH-1:0]  A1 = 'x;
			logic [WIDTH-1:0]  B1 = 'x;
			logic [O_WIDTH-1:0]  P2 = 'x;
			always_ff @(posedge clk) begin
				if(rst) begin
					A1 <= 'x;
					B1 <= 'x;
					P2 <= 'x;
				end
				else begin
					A1 <= a;
					if(bload)  B1 <= b;
					if(SIGNED)  P2 <= O_WIDTH'($signed(A1) * $signed(B1));
					else        P2 <= A1 * B1;
				end
			end
			assign	r = P2;
		end : genBehav

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
