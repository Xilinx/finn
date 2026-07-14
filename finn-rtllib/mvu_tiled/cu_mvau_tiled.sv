/******************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *****************************************************************************/

module cu_mvau_tiled #(
	int unsigned  PE,
	int unsigned  SIMD,
	int unsigned  TH,
	int unsigned  WEIGHT_WIDTH,
	int unsigned  ACTIVATION_WIDTH,
	int unsigned  ACCU_WIDTH,

	bit  SIGNED_ACTIVATIONS = 1,
	localparam int unsigned  WEIGHT_ELEMENTS = PE*SIMD
)(
	input   logic  clk,
	input   logic  rst,
	input   logic  en,

	input   logic  ilast,
	input   logic  ivld,
	input   logic [WEIGHT_ELEMENTS-1:0][WEIGHT_WIDTH-1:0]  w,
	input   logic [SIMD-1:0][ACTIVATION_WIDTH-1:0]  a,

	output  logic  ovld,
	output  logic [PE-1:0][ACCU_WIDTH-1:0]  p
);

	//=== Startup Recovery Watchdog =========================================
	// The DSP slice needs 100ns of recovery time after initial startup before
	// being able to ingest input properly. This watchdog discovers violating
	// stimuli during simulation and produces a corresponding warning.
	if(1) begin : blkRecoveryWatch
		logic  Dirty = 1;
		initial begin
			#100ns;
			Dirty <= 0;
		end

		always_ff @(posedge clk) begin
			assert(!Dirty || rst || !en) else begin
				$warning("%m: Feeding input during DSP startup recovery. Expect functional errors.");
			end
		end
	end : blkRecoveryWatch

	//=== Input Formatting ==================================================
	localparam int unsigned  CHAINLEN = (SIMD+2)/3;
	uwire [26:0]  a_in_i[CHAINLEN];
	uwire [23:0]  b_in_i[PE][CHAINLEN];
	// Array with packed dimension > 256 cannot be handled out-of-the-box with PyVerilator
	uwire [PE-1:0][CHAINLEN-1:0][ACCU_WIDTH-1:0]  pout;

	//--- Valid/Last Pipeline -----------------------------------------------
	localparam int unsigned  DSP_PIPELINE_STAGES = 1;
	logic  L[0:1+DSP_PIPELINE_STAGES] = '{default: 0};
	logic  V[0:1+DSP_PIPELINE_STAGES] = '{default: 0};

	always_ff @(posedge clk) begin
		if(rst) begin
			L <= '{default: 0};
			V <= '{default: 0};
		end
		else if(en) begin
			L[1+DSP_PIPELINE_STAGES] <= ilast;
			L[0:DSP_PIPELINE_STAGES] <= L[1:1+DSP_PIPELINE_STAGES];

			V[1+DSP_PIPELINE_STAGES] <= ivld;
			V[0:DSP_PIPELINE_STAGES] <= V[1:1+DSP_PIPELINE_STAGES];
		end
	end

	uwire  last = L[0];
	uwire  vld  = V[0];

	//--- Activation Padding ------------------------------------------------
	localparam int unsigned  PAD_BITS_ACT = 9 - ACTIVATION_WIDTH;
	for(genvar  i = 0; i < CHAINLEN; i++) begin : genActSIMD
		localparam int unsigned  LANES_OCCUPIED = i == CHAINLEN-1? SIMD - 3*i : 3;

		for(genvar  j = 0; j < LANES_OCCUPIED; j++) begin : genAin
			assign  a_in_i[i][9*j +: 9] =
				SIGNED_ACTIVATIONS?
					PAD_BITS_ACT == 0? a[3*i+j] : { {PAD_BITS_ACT{a[3*i+j][ACTIVATION_WIDTH-1]}}, a[3*i+j] } :
					PAD_BITS_ACT == 0? a[3*i+j] : { {PAD_BITS_ACT{1'b0}}, a[3*i+j] };
		end : genAin
		for(genvar  j = LANES_OCCUPIED; j < 3; j++) begin : genAinZero
			assign  a_in_i[i][9*j +: 9] = 9'd0;
		end : genAinZero
	end : genActSIMD

	//--- Weight Padding ----------------------------------------------------
	localparam int unsigned  PAD_BITS_WEIGHT = 8 - WEIGHT_WIDTH;

	for(genvar  i = 0; i < PE; i++) begin : genWeightPE
		for(genvar  j = 0; j < CHAINLEN; j++) begin : genWeightSIMD
			localparam int unsigned  LANES_OCCUPIED = j == CHAINLEN-1? SIMD - 3*j : 3;

			for(genvar  k = 0; k < LANES_OCCUPIED; k++) begin : genBin
				assign  b_in_i[i][j][8*k +: 8] =
					PAD_BITS_WEIGHT == 0? w[SIMD*i+3*j+k] : { {PAD_BITS_WEIGHT{w[SIMD*i+3*j+k][WEIGHT_WIDTH-1]}}, w[SIMD*i+3*j+k] };
			end : genBin
			for(genvar  k = LANES_OCCUPIED; k < 3; k++) begin : genBinZero
				assign  b_in_i[i][j][8*k +: 8] = 8'd0;
			end : genBinZero
		end : genWeightSIMD
	end : genWeightPE

	//=== DSP Instances =====================================================
	for(genvar  i = 0; i < PE; i++) begin : genPE
		for(genvar  j = 0; j < CHAINLEN; j++) begin : genChain
			localparam int unsigned  INTERNAL_REGS = 1;
			localparam bit  PREG = 1;

			DSP58 #(
				// Feature Control Attributes: Data Path Selection
				.AMULTSEL("A"),
				.A_INPUT("DIRECT"),
				.BMULTSEL("B"),
				.B_INPUT("DIRECT"),
				.DSP_MODE("INT8"),
				.PREADDINSEL("A"),
				.RND(58'h000000000000000),
				.USE_MULT("MULTIPLY"),
				.USE_SIMD("ONE58"),
				.USE_WIDEXOR("FALSE"),
				.XORSIMD("XOR24_34_58_116"),
				// Pattern Detector Attributes
				.AUTORESET_PATDET("NO_RESET"),
				.AUTORESET_PRIORITY("RESET"),
				.MASK(58'h0ffffffffffffff),
				.PATTERN(58'h000000000000000),
				.SEL_MASK("MASK"),
				.SEL_PATTERN("PATTERN"),
				.USE_PATTERN_DETECT("NO_PATDET"),
				// Programmable Inversion Attributes
				.IS_ALUMODE_INVERTED(4'b0000),
				.IS_CARRYIN_INVERTED(1'b0),
				.IS_CLK_INVERTED(1'b0),
				.IS_INMODE_INVERTED(5'b00000),
				.IS_NEGATE_INVERTED(3'b000),
				.IS_OPMODE_INVERTED({
					2'b00,  // W: 0 (unused, accumulation is external)
					3'b000, // Z: 0 (unused)
					2'b01,  // Y: M (multiply)
					2'b01   // X: M (multiply)
				}), // Static OPMODE='0 inverted to select P = M (multiply-only)
				.IS_RSTALLCARRYIN_INVERTED(1'b0),
				.IS_RSTALUMODE_INVERTED(1'b0),
				.IS_RSTA_INVERTED(1'b0),
				.IS_RSTB_INVERTED(1'b0),
				.IS_RSTCTRL_INVERTED(1'b0),
				.IS_RSTC_INVERTED(1'b0),
				.IS_RSTD_INVERTED(1'b0),
				.IS_RSTINMODE_INVERTED(1'b0),
				.IS_RSTM_INVERTED(1'b0),
				.IS_RSTP_INVERTED(1'b0),
				// Register Control Attributes
				.ACASCREG(INTERNAL_REGS),
				.ADREG(0),
				.ALUMODEREG(0),
				.AREG(INTERNAL_REGS),
				.BCASCREG(INTERNAL_REGS),
				.BREG(INTERNAL_REGS),
				.CARRYINREG(0),
				.CARRYINSELREG(0),
				.CREG(0),
				.DREG(0),
				.INMODEREG(1),
				.MREG(1),
				.OPMODEREG(0),  // No register needed: OPMODE is static
				.PREG(PREG),
				.RESET_MODE("SYNC")
			)
			DSP58_inst (
				// Cascade outputs
				.ACOUT(),
				.BCOUT(),
				.CARRYCASCOUT(),
				.MULTSIGNOUT(),
				.PCOUT(),
				// Control outputs
				.OVERFLOW(),
				.PATTERNBDETECT(),
				.PATTERNDETECT(),
				.UNDERFLOW(),
				// Data outputs
				.CARRYOUT(),
				.P(pout[i][j]),
				.XOROUT(),
				// Cascade inputs
				.ACIN('x),
				.BCIN('x),
				.CARRYCASCIN('x),
				.MULTSIGNIN('x),
				.PCIN('x),
				// Control inputs
				.ALUMODE(4'h0),
				.CARRYINSEL('0),
				.CLK(clk),
				.INMODE({
					INTERNAL_REGS == 2? 1'b0 : 1'b1,
					2'b00,
					1'b0,
					INTERNAL_REGS == 2? 1'b0 : 1'b1
				}),
				.NEGATE('0),
				.OPMODE('0),  // Static (inverted to X=Y=M, W=Z=0)
				// Data inputs
				.A({ 7'bx, a_in_i[j] }),
				.B(b_in_i[i][j]),
				.C('x),
				.CARRYIN('0),
				.D('x),
				// Reset/Clock Enable inputs
				.ASYNC_RST('0),
				.CEA1(en),
				.CEA2(INTERNAL_REGS == 2? en : '0),
				.CEAD('0),
				.CEALUMODE('0),
				.CEB1(en),
				.CEB2(INTERNAL_REGS == 2? en : '0),
				.CEC('0),
				.CECARRYIN('0),
				.CECTRL('0),
				.CED('0),
				.CEINMODE(en),
				.CEM(en),
				.CEP(PREG && en),
				.RSTA('0),
				.RSTALLCARRYIN('0),
				.RSTALUMODE('0),
				.RSTB('0),
				.RSTC('0),
				.RSTCTRL('0),
				.RSTD('0),
				.RSTINMODE(rst),
				.RSTM('0),
				.RSTP('0)
			);
		end : genChain
	end : genPE

	//=== Accumulation ======================================================
	acc_stage #(.CHAINLEN(CHAINLEN), .PE(PE), .ACCU_WIDTH(ACCU_WIDTH), .TH(TH)) inst_acc_stage (
		.clk(clk),
		.rst(rst),
		.en(en),
		.idat(pout),
		.ival(vld),
		.ilast(last),
		.odat(p),
		.oval(ovld)
	);

endmodule : cu_mvau_tiled
