/******************************************************************************
 * Copyright (C) 2024, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * @brief	Matrix/Vector Vector Unit (MVU/VVU) core compute kernel utilizing DSP58.
 *****************************************************************************/

module mvu_vvu_8sx9_dsp58 #(
	bit IS_MVU,
    int unsigned PE,
    int unsigned SIMD,
    int unsigned ACTIVATION_WIDTH,
    int unsigned WEIGHT_WIDTH,
	int unsigned ACCU_WIDTH,
    bit SIGNED_ACTIVATIONS = 0,
    int unsigned SEGMENTLEN = 0, // Default to 0 (which implies a single segment)
	bit FORCE_BEHAVIORAL = 0,

	localparam int unsigned ACTIVATION_ELEMENTS = (IS_MVU ? 1 : PE) * SIMD,
	localparam int unsigned WEIGHT_ELEMENTS = PE*SIMD
  )
  (
    // Global Control
	input   logic clk,
    input   logic rst,
    input   logic en,

	// Input
    input   logic last,
    input   logic zero, // ignore current inputs and force this partial product to zero
    input   logic [WEIGHT_ELEMENTS-1:0][WEIGHT_WIDTH-1:0] w, // weights
	input   logic [ACTIVATION_ELEMENTS-1:0][ACTIVATION_WIDTH-1:0] a, // activations

	// Ouput
	output  logic vld,
    output  logic [PE-1:0][ACCU_WIDTH-1:0] p
  );
	// for verilator always use behavioral code
	localparam bit  BEHAVIORAL =
`ifdef VERILATOR
		1 ||
`endif
		FORCE_BEHAVIORAL;

//-------------------- Declare global signals --------------------\\
	localparam int unsigned CHAINLEN = (SIMD+2)/3;
	localparam int unsigned SEGLEN = SEGMENTLEN == 0 ? CHAINLEN : SEGMENTLEN; // Additional constant to default a SEGMENTLEN of '0' to the DSP-chain length
	localparam int unsigned PE_ACTIVATION = IS_MVU ? 1 : PE;
	uwire [26:0] a_in_i [PE_ACTIVATION * CHAINLEN];
	uwire [23:0] b_in_i [PE][CHAINLEN];
	uwire [PE-1:0][CHAINLEN-1:0][57:0] pcout; // Array with packed dimension > 256 (with a loop-carried dependency) cannot be handled out-of-the-box with PyVerilator

//-------------------- Shift register for opmode select signal --------------------\\
	localparam int unsigned MAX_PIPELINE_STAGES = (CHAINLEN + SEGLEN-1)/SEGLEN; // >=1 (== number of pipeline registers + 1 (A/B inputs always have 1 register))
	logic L [0:1+MAX_PIPELINE_STAGES] = '{default: 0}; // After MAX_PIPELINE_STAGES (== number of pipeline stages for input data), we have 3 additional cycles latency (A/B reg, Mreg, Preg).
	// Thus, we add +2 (since OPMODE is buffered by 1 cycle in the DSP fabric)

	always_ff @(posedge clk) begin
		if(rst)     L <= '{default: 0};
		else if(en) begin
			L[1+MAX_PIPELINE_STAGES] <= last;
			L[0:MAX_PIPELINE_STAGES] <= L[1:1+MAX_PIPELINE_STAGES];
		end
	end
	assign vld = L[0];

//-------------------- Shift register for ZERO flag --------------------\\
	logic Z [0:MAX_PIPELINE_STAGES-2] = '{default:0}; // We need MAX_PIPELINE_STAGES-1 pipeline stages (note: INMODE is buffered inside DSP fabric)

	if (MAX_PIPELINE_STAGES > 1) begin : genZreg
		always_ff @(posedge clk) begin
			if (rst)      Z <= '{default: 0};
			else if(en) begin
				Z[0] <= zero;
				if (MAX_PIPELINE_STAGES > 2)  Z[1:MAX_PIPELINE_STAGES-2] <= Z[0:MAX_PIPELINE_STAGES-3];
			end
		end
	end;

//-------------------- Buffer for input activations --------------------\\
	localparam int unsigned PAD_BITS_ACT = 9 - ACTIVATION_WIDTH;
	for (genvar k=0; k<PE_ACTIVATION; k++) begin : genActPE
		for (genvar i=0; i<CHAINLEN; i++) begin : genActSIMD
			localparam int TOTAL_PREGS = i/SEGLEN;
			localparam int EXTERNAL_PREGS = TOTAL_PREGS>1 ? TOTAL_PREGS-1 : 0;
			localparam int LANES_OCCUPIED = i == CHAINLEN-1 ? SIMD - 3*i : 3;

			if (EXTERNAL_PREGS > 0) begin : genExternalPregAct
				logic [0:EXTERNAL_PREGS-1][LANES_OCCUPIED-1:0][ACTIVATION_WIDTH-1:0] A = '{ default : 0};
				always_ff @(posedge clk) begin
					if (rst)     A <= '{default: 0};
					else if(en) begin
						A[EXTERNAL_PREGS-1] <=
// synthesis translate_off
							zero ? '1 :
// synthesis translate_on
							a[SIMD*k + 3*i +: LANES_OCCUPIED];
						if (EXTERNAL_PREGS > 1)   A[0:EXTERNAL_PREGS-2] <= A[1:EXTERNAL_PREGS-1];
					end
				end
				for (genvar j=0; j<LANES_OCCUPIED; j++) begin : genAin
				assign a_in_i[CHAINLEN*k+i][9*j +: 9] = SIGNED_ACTIVATIONS ? PAD_BITS_ACT == 0 ? A[0][j] : { {PAD_BITS_ACT{A[0][j][ACTIVATION_WIDTH-1]}}, A[0][j] }
													  : PAD_BITS_ACT == 0 ? A[0][j] : { {PAD_BITS_ACT{1'b0}}, A[0][j] } ;
				end : genAin
				for (genvar j=LANES_OCCUPIED; j<3; j++) begin : genAinZero
					assign a_in_i[CHAINLEN*k+i][9*j +: 9] = 9'b0;
				end : genAinZero
			end : genExternalPregAct
			else begin : genInpDSPAct
				for (genvar j=0; j<LANES_OCCUPIED; j++) begin : genAin
					assign a_in_i[CHAINLEN*k+i][9*j +: 9] =
// synthesis translate_off
						zero ? '1 :
// synthesis translate_on
						SIGNED_ACTIVATIONS ? PAD_BITS_ACT == 0 ? a[SIMD*k+3*i+j] : { {PAD_BITS_ACT{a[SIMD*k+3*i+j][ACTIVATION_WIDTH-1]}}, a[SIMD*k+3*i+j] }
													: PAD_BITS_ACT == 0 ? a[SIMD*k+3*i+j] : { {PAD_BITS_ACT{1'b0}}, a[SIMD*k+3*i+j] } ;
				end : genAin
				for (genvar j=LANES_OCCUPIED; j<3; j++) begin : genAinZero
					assign a_in_i[CHAINLEN*k+i][9*j +: 9] = 9'b0;
				end : genAinZero
			end : genInpDSPAct
		end : genActSIMD
	end : genActPE

//-------------------- Buffer for weights --------------------\\
	localparam int unsigned PAD_BITS_WEIGHT = 8 - WEIGHT_WIDTH;

	for (genvar i=0; i<PE; i++) begin : genWeightPE
		for (genvar j=0; j<CHAINLEN; j++) begin : genWeightSIMD
			localparam int TOTAL_PREGS = j/SEGLEN;
			localparam int EXTERNAL_PREGS = TOTAL_PREGS>1 ? TOTAL_PREGS-1 : 0;
			localparam int LANES_OCCUPIED = j == CHAINLEN-1 ? SIMD - 3*j : 3;

			if (EXTERNAL_PREGS > 0) begin : genExternalPregWeight
				logic [0:PE-1][0:EXTERNAL_PREGS-1][LANES_OCCUPIED-1:0][WEIGHT_WIDTH-1:0] B = '{ default : 0};
				always_ff @(posedge clk) begin
					if (rst)    B <= '{default: 0};
					else if (en) begin
						B[i][EXTERNAL_PREGS-1] <=
// synthesis translate_off
							zero ? '1 :
// synthesis translate_on
							//w[i][3*j +: LANES_OCCUPIED];
							w[SIMD*i+3*j +: LANES_OCCUPIED];
						if (EXTERNAL_PREGS > 1) B[i][0:EXTERNAL_PREGS-2] <= B[i][1:EXTERNAL_PREGS-1];
					end
				end
				for (genvar k = 0 ; k < LANES_OCCUPIED ; k++) begin : genBin
					assign b_in_i[i][j][8*k +: 8] = PAD_BITS_WEIGHT == 0 ? B[i][0][k] : { {PAD_BITS_WEIGHT{B[i][0][k][WEIGHT_WIDTH-1]}}, B[i][0][k] };
				end : genBin
				for (genvar k=LANES_OCCUPIED; k<3; k++) begin : genBinZero
					assign b_in_i[i][j][8*k +: 8] = 8'b0;
				end : genBinZero
			end : genExternalPregWeight
			else begin : genInpDSPWeight
				for (genvar k = 0; k < LANES_OCCUPIED; k++) begin : genBin
					assign b_in_i[i][j][8*k +: 8] =
// synthesis translate_off
						zero ? '1 :
// synthesis translate_on
						PAD_BITS_WEIGHT == 0 ? w[SIMD*i+3*j+k] : { {PAD_BITS_WEIGHT{w[SIMD*i+3*j+k][WEIGHT_WIDTH-1]}}, w[SIMD*i+3*j+k] };
				end : genBin
				for (genvar k=LANES_OCCUPIED; k<3; k++) begin : genBinZero
					assign b_in_i[i][j][8*k +: 8] = 8'b0;
				end : genBinZero
			end : genInpDSPWeight
		end : genWeightSIMD
	end : genWeightPE

//-------------------- Instantiate PE x CHAINLEN DSPs --------------------\\
	for (genvar i=0; i<PE; i++) begin : genDSPPE
		for (genvar j=0; j<CHAINLEN; j++) begin : genDSPChain
			localparam int TOTAL_PREGS = j/SEGLEN;
			localparam int INTERNAL_PREGS = TOTAL_PREGS>0 ? 2 : 1; // 1 : 0
			localparam bit PREG = (j+1)%SEGLEN==0 || j == CHAINLEN-1;
			localparam bit FIRST = j == 0;
			localparam bit LAST = j == CHAINLEN-1;
			uwire [57:0] pp;

			if (LAST) begin : genPOUT
				assign p[i] = pp[ACCU_WIDTH-1:0];
			end

			// Note: Since the product B * AD is computed,
			//       rst can be only applied to AD and zero only to B
			//       with the same effect as zeroing both.
			if(BEHAVIORAL) begin : genBehav
				// Stage #1: Input A/B
				logic signed [33:0] Areg [INTERNAL_PREGS];
				always_ff @(posedge clk) begin
					if (rst)	Areg <= '{ default : 0};
					else if (en) begin
						Areg[0] <= { 7'bx, a_in_i[(IS_MVU ? 0 : CHAINLEN*i) + j] };
						if (INTERNAL_PREGS == 2) Areg[1] <= Areg[0];
					end
				end
				logic signed [23:0] Breg [INTERNAL_PREGS];
				always_ff @(posedge clk) begin
					if (rst)	Breg <= '{ default : 0};
					else if (en) begin
						Breg[0] <= b_in_i[i][j];
						if (INTERNAL_PREGS == 2) Breg[1] <= Breg[0];
					end
				end

				// Stage #2: Multiply-Accumulate
				logic signed [57:0] Mreg;
				logic InmodeZero = 0;
				always_ff @(posedge clk) begin
					if (rst)		InmodeZero <= 0;
					else if (en)	InmodeZero <= ( TOTAL_PREGS > 0 ? Z[TOTAL_PREGS-1] : zero );
				end
				always_ff @(posedge clk) begin
					if (rst)	Mreg <= 0;
					else if (en) begin
						automatic logic signed [57:0] m = 0;
						for (int k = 0; k < 3; k++) begin
							m = m + (InmodeZero ? 0 : $signed(Areg[INTERNAL_PREGS-1][9*k +: 9]) * $signed(Breg[INTERNAL_PREGS-1][8*k +: 8]));
						end
						Mreg <= m;
					end
				end

				// Stage #3: Accumulate
				logic signed [57:0] Preg;
				logic Opmode = 0;
				if (FIRST && !LAST) begin : genFirst
					if (PREG) begin : genPregBehav
						always_ff @(posedge clk) begin
							if (rst)		Preg <= 0;
							else if (en)	Preg <= Mreg;
						end
					end
					else	assign Preg = Mreg;
				end
				else if (FIRST && LAST) begin : genSingle
					always_ff @(posedge clk) begin
						if (rst)		Opmode <= 0;
						else if (en)	Opmode <= L[1];
					end
					always_ff @(posedge clk) begin
						if (rst) 		Preg <= 0;
						else if (en)	Preg <= (Opmode ? 0 : Preg) + Mreg;
					end
				end
				else if (!FIRST && LAST) begin : genLast
					always_ff @(posedge clk) begin
						if (rst)		Opmode <= 0;
						else if (en)	Opmode <= L[1];
					end
					always_ff @(posedge clk) begin
						if (rst) 		Preg <= 0;
						else if (en)	Preg <= (Opmode ? 0 : Preg) + Mreg + pcout[i][j-1];
					end
				end
				else begin : genMid
					if (PREG) begin : genPregBehav
						always_ff @(posedge clk) begin
							if (rst)		Preg <= 0;
							else if (en)	Preg <= Mreg + pcout[i][j-1];
						end
					end
					else	assign Preg = Mreg + pcout[i][j-1];
				end
				assign pp = Preg;
				assign pcout[i][j] = Preg;
			end : genBehav
`ifndef VERILATOR
			else begin: genDSP
				DSP58 #(
					// Feature Control Attributes: Data Path Selection
					.AMULTSEL("A"),                     // Selects A input to multiplier (A, AD)
					.A_INPUT("DIRECT"),                 // Selects A input source, "DIRECT" (A port) or "CASCADE" (ACIN port)
					.BMULTSEL("B"),                     // Selects B input to multiplier (AD, B)
					.B_INPUT("DIRECT"),                 // Selects B input source, "DIRECT" (B port) or "CASCADE" (BCIN port)
					.DSP_MODE("INT8"),                  // Configures DSP to a particular mode of operation. Set to INT24 for
														// legacy mode.
					.PREADDINSEL("A"),                  // Selects input to pre-adder (A, B)
					.RND(58'h000000000000000),          // Rounding Constant
					.USE_MULT("MULTIPLY"),              // Select multiplier usage (DYNAMIC, MULTIPLY, NONE)
					.USE_SIMD("ONE58"),                 // SIMD selection (FOUR12, ONE58, TWO24)
					.USE_WIDEXOR("FALSE"),              // Use the Wide XOR function (FALSE, TRUE)
					.XORSIMD("XOR24_34_58_116"),        // Mode of operation for the Wide XOR (XOR12_22, XOR24_34_58_116)
					// Pattern Detector Attributes: Pattern Detection Configuration
					.AUTORESET_PATDET("NO_RESET"),      // NO_RESET, RESET_MATCH, RESET_NOT_MATCH
					.AUTORESET_PRIORITY("RESET"),       // Priority of AUTORESET vs. CEP (CEP, RESET).
					.MASK(58'h0ffffffffffffff),         // 58-bit mask value for pattern detect (1=ignore)
					.PATTERN(58'h000000000000000),      // 58-bit pattern match for pattern detect
					.SEL_MASK("MASK"),                  // C, MASK, ROUNDING_MODE1, ROUNDING_MODE2
					.SEL_PATTERN("PATTERN"),            // Select pattern value (C, PATTERN)
					.USE_PATTERN_DETECT("NO_PATDET"),   // Enable pattern detect (NO_PATDET, PATDET)
					// Programmable Inversion Attributes: Specifies built-in programmable inversion on specific pins
					.IS_ALUMODE_INVERTED(4'b0000),      // Optional inversion for ALUMODE
					.IS_CARRYIN_INVERTED(1'b0),         // Optional inversion for CARRYIN
					.IS_CLK_INVERTED(1'b0),             // Optional inversion for CLK
					.IS_INMODE_INVERTED(5'b00000),      // Optional inversion for INMODE
					.IS_NEGATE_INVERTED(3'b000),        // Optional inversion for NEGATE
					.IS_OPMODE_INVERTED({ LAST ? 2'b01 : 2'b00 , // W: LAST ? (L[1] ? 0 : P) : 0
										FIRST ? 3'b000 : 3'b001, // Z: FIRST ? 0 : PCIN
										2'b01, // Y : M
										2'b01  // X: M
					}), // Optional inversion for OPMODE
					.IS_RSTALLCARRYIN_INVERTED(1'b0),   // Optional inversion for RSTALLCARRYIN
					.IS_RSTALUMODE_INVERTED(1'b0),      // Optional inversion for RSTALUMODE
					.IS_RSTA_INVERTED(1'b0),            // Optional inversion for RSTA
					.IS_RSTB_INVERTED(1'b0),            // Optional inversion for RSTB
					.IS_RSTCTRL_INVERTED(1'b0),         // Optional inversion for STCONJUGATE_A
					.IS_RSTC_INVERTED(1'b0),            // Optional inversion for RSTC
					.IS_RSTD_INVERTED(1'b0),            // Optional inversion for RSTD
					.IS_RSTINMODE_INVERTED(1'b0),       // Optional inversion for RSTINMODE
					.IS_RSTM_INVERTED(1'b0),            // Optional inversion for RSTM
					.IS_RSTP_INVERTED(1'b0),            // Optional inversion for RSTP
					// Register Control Attributes: Pipeline Register Configuration
					.ACASCREG(INTERNAL_PREGS),          // Number of pipeline stages between A/ACIN and ACOUT (0-2)
					.ADREG(0),                          // Pipeline stages for pre-adder (0-1)
					.ALUMODEREG(0),                     // Pipeline stages for ALUMODE (0-1)
					.AREG(INTERNAL_PREGS),              // Pipeline stages for A (0-2)
					.BCASCREG(INTERNAL_PREGS),          // Number of pipeline stages between B/BCIN and BCOUT (0-2)
					.BREG(INTERNAL_PREGS),              // Pipeline stages for B (0-2)
					.CARRYINREG(0),                     // Pipeline stages for CARRYIN (0-1)
					.CARRYINSELREG(0),                  // Pipeline stages for CARRYINSEL (0-1)
					.CREG(0),                           // Pipeline stages for C (0-1)
					.DREG(0),                           // Pipeline stages for D (0-1)
					.INMODEREG(1),                      // Pipeline stages for INMODE (0-1)
					.MREG(1),                           // Multiplier pipeline stages (0-1)
					.OPMODEREG(1),                      // Pipeline stages for OPMODE (0-1)
					.PREG(PREG),                        // Number of pipeline stages for P (0-1)
					.RESET_MODE("SYNC")                 // Selection of synchronous or asynchronous reset. (ASYNC, SYNC).
				)
				DSP58_inst (
					// Cascade outputs: Cascade Ports
					.ACOUT(),                           // 34-bit output: A port cascade
					.BCOUT(),                           // 24-bit output: B cascade
					.CARRYCASCOUT(),                    // 1-bit output: Cascade carry
					.MULTSIGNOUT(),                     // 1-bit output: Multiplier sign cascade
					.PCOUT(pcout[i][j]),                // 58-bit output: Cascade output
					// Control outputs: Control Inputs/Status Bits
					.OVERFLOW(),                        // 1-bit output: Overflow in add/acc
					.PATTERNBDETECT(),                  // 1-bit output: Pattern bar detect
					.PATTERNDETECT(),                   // 1-bit output: Pattern detect
					.UNDERFLOW(),                       // 1-bit output: Underflow in add/acc
					// Data outputs: Data Ports
					.CARRYOUT(),                        // 4-bit output: Carry
					.P(pp),                             // 58-bit output: Primary data
					.XOROUT(),                          // 8-bit output: XOR data
					// Cascade inputs: Cascade Ports
					.ACIN('x),                          // 34-bit input: A cascade data
					.BCIN('x),                          // 24-bit input: B cascade
					.CARRYCASCIN('x),                   // 1-bit input: Cascade carry
					.MULTSIGNIN('x),                    // 1-bit input: Multiplier sign cascade
					.PCIN(FIRST ? 'x : pcout[i][j-1]),  // 58-bit input: P cascade
					// Control inputs: Control Inputs/Status Bits
					.ALUMODE(4'h0),                     // 4-bit input: ALU control
					.CARRYINSEL('0),                    // 3-bit input: Carry select
					.CLK(clk),                          // 1-bit input: Clock
					.INMODE({
							INTERNAL_PREGS==2 ? 1'b0 : 1'b1,
							2'b00,
							TOTAL_PREGS > 0 ? Z[TOTAL_PREGS-1] : zero,
							INTERNAL_PREGS==2 ? 1'b0 : 1'b1
					}),                                 // 5-bit input: INMODE control
					.NEGATE('0),                        // 3-bit input: Negates the input of the multiplier
					.OPMODE({
							LAST ? {1'b0, L[1]} : 2'b00,
							7'b000_0000
					}), // 9-bit input: Operation mode
					// Data inputs: Data Ports
					.A({ 7'bx, a_in_i[(IS_MVU ? 0 : CHAINLEN*i) + j] }),            // 34-bit input: A data
					.B(b_in_i[i][j]),                   // 24-bit input: B data
					.C('x),                             // 58-bit input: C data
					.CARRYIN('0),                       // 1-bit input: Carry-in
					.D('x),                             // 27-bit input: D data
					// Reset/Clock Enable inputs: Reset/Clock Enable Inputs
					.ASYNC_RST('0),                     // 1-bit input: Asynchronous reset for all registers.
					.CEA1(en),                          // 1-bit input: Clock enable for 1st stage AREG
					.CEA2(INTERNAL_PREGS==2 ? en : '0), // 1-bit input: Clock enable for 2nd stage AREG
					.CEAD('0),                          // 1-bit input: Clock enable for ADREG
					.CEALUMODE('0),                     // 1-bit input: Clock enable for ALUMODE
					.CEB1(en),                          // 1-bit input: Clock enable for 1st stage BREG
					.CEB2(INTERNAL_PREGS==2 ? en : '0), // 1-bit input: Clock enable for 2nd stage BREG
					.CEC('0),                           // 1-bit input: Clock enable for CREG
					.CECARRYIN('0),                     // 1-bit input: Clock enable for CARRYINREG
					.CECTRL(en),                        // 1-bit input: Clock enable for OPMODEREG and CARRYINSELREG
					.CED('0),                           // 1-bit input: Clock enable for DREG
					.CEINMODE(en),                      // 1-bit input: Clock enable for INMODEREG
					.CEM(en),                           // 1-bit input: Clock enable for MREG
					.CEP(PREG && en),                   // 1-bit input: Clock enable for PREG
					.RSTA(rst),                         // 1-bit input: Reset for AREG
					.RSTALLCARRYIN('0),                 // 1-bit input: Reset for CARRYINREG
					.RSTALUMODE('0),                    // 1-bit input: Reset for ALUMODEREG
					.RSTB(rst),                         // 1-bit input: Reset for BREG
					.RSTC('0),                          // 1-bit input: Reset for CREG
					.RSTCTRL(rst),                      // 1-bit input: Reset for OPMODEREG and CARRYINSELREG
					.RSTD('0),                          // 1-bit input: Reset for DREG and ADREG
					.RSTINMODE(rst),                    // 1-bit input: Reset for INMODE register
					.RSTM(rst),                         // 1-bit input: Reset for MREG
					.RSTP(PREG && rst)                  // 1-bit input: Reset for PREG
				);
			end : genDSP
`endif
		end : genDSPChain
	end : genDSPPE

endmodule : mvu_vvu_8sx9_dsp58
