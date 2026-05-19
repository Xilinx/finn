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
 *****************************************************************************/

module cu_mvau_tiled #(
    int unsigned PE,
    int unsigned SIMD,
    int unsigned TH,
	int unsigned WEIGHT_WIDTH,
	int unsigned ACTIVATION_WIDTH,
	int unsigned ACCU_WIDTH,

    bit SIGNED_ACTIVATIONS = 1,
	localparam int unsigned WEIGHT_ELEMENTS = PE*SIMD
  )  (
    // Global Control
	input   logic clk,
    input   logic rst,
    input   logic en,

	// Input
    input   logic ilast,
    input   logic ivld,
    input   logic [WEIGHT_ELEMENTS-1:0][WEIGHT_WIDTH-1:0] w, // weights
	input   logic [SIMD-1:0][ACTIVATION_WIDTH-1:0] a, // activations

	// Ouput
	output  logic ovld,
    output  logic [PE-1:0][ACCU_WIDTH-1:0] p
  );

	//-----------------------------------------------------------------------
	// Startup Recovery Watchdog
	//  The DSP slice needs 100ns of recovery time after initial startup before
	//  being able to ingest input properly. This watchdog discovers violating
	//  stimuli during simulation and produces a corresponding warning.
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

//-------------------- Declare global signals --------------------\\
	localparam int unsigned CHAINLEN = (SIMD+2)/3;
	uwire [26:0] a_in_i [CHAINLEN];
	uwire [23:0] b_in_i [PE][CHAINLEN];
	uwire [PE-1:0][CHAINLEN-1:0][ACCU_WIDTH-1:0] pout; // Array with packed dimension > 256 (with a loop-carried dependency) cannot be handled out-of-the-box with PyVerilator

//-------------------- Shift register for last and valid signals --------------------\\
	localparam int unsigned DSP_PIPELINE_STAGES = 1;
	logic L [0:1+DSP_PIPELINE_STAGES] = '{default: 0};
    logic V [0:1+DSP_PIPELINE_STAGES] = '{default: 0};

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

    logic last;
    logic vld;
    assign last = L[0];
    assign vld = V[0];

//-------------------- Buffer for input activations --------------------\\
	localparam int unsigned PAD_BITS_ACT = 9 - ACTIVATION_WIDTH;
    for (genvar i=0; i<CHAINLEN; i++) begin : genActSIMD
        localparam int LANES_OCCUPIED = i == CHAINLEN-1 ? SIMD - 3*i : 3;

        for (genvar j=0; j<LANES_OCCUPIED; j++) begin : genAin
            assign a_in_i[i][9*j +: 9] =
                SIGNED_ACTIVATIONS ? PAD_BITS_ACT == 0 ? a[3*i+j] : { {PAD_BITS_ACT{a[3*i+j][ACTIVATION_WIDTH-1]}}, a[3*i+j] }
                                            : PAD_BITS_ACT == 0 ? a[3*i+j] : { {PAD_BITS_ACT{1'b0}}, a[3*i+j] } ;
        end : genAin
        for (genvar j=LANES_OCCUPIED; j<3; j++) begin : genAinZero
            assign a_in_i[i][9*j +: 9] = 9'b0;
        end : genAinZero
    end : genActSIMD

//-------------------- Buffer for weights --------------------\\
	localparam int unsigned PAD_BITS_WEIGHT = 8 - WEIGHT_WIDTH;

	for (genvar i=0; i<PE; i++) begin : genWeightPE
		for (genvar j=0; j<CHAINLEN; j++) begin : genWeightSIMD
			localparam int LANES_OCCUPIED = j == CHAINLEN-1 ? SIMD - 3*j : 3;


            for (genvar k = 0; k < LANES_OCCUPIED; k++) begin : genBin
                assign b_in_i[i][j][8*k +: 8] =
                    PAD_BITS_WEIGHT == 0 ? w[SIMD*i+3*j+k] : { {PAD_BITS_WEIGHT{w[SIMD*i+3*j+k][WEIGHT_WIDTH-1]}}, w[SIMD*i+3*j+k] };
            end : genBin
            for (genvar k=LANES_OCCUPIED; k<3; k++) begin : genBinZero
                assign b_in_i[i][j][8*k +: 8] = 8'b0;
            end : genBinZero
		end : genWeightSIMD
	end : genWeightPE

//-------------------- Instantiate PE x CHAINLEN DSPs --------------------\\
	for (genvar i=0; i<PE; i++) begin
		for (genvar j=0; j<CHAINLEN; j++) begin
			localparam int INTERNAL_REGS = 1; // 1 : 0
			localparam bit PREG = 1;

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
                .IS_OPMODE_INVERTED({ 2'b00 , // W: LAST ? (L[1] ? 0 : P) : 0
                                    3'b000, // Z: FIRST ? 0 : PCIN
                                    2'b01, // Y: M
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
                .ACASCREG(INTERNAL_REGS),          // Number of pipeline stages between A/ACIN and ACOUT (0-2)
                .ADREG(0),                          // Pipeline stages for pre-adder (0-1)
                .ALUMODEREG(0),                     // Pipeline stages for ALUMODE (0-1)
                .AREG(INTERNAL_REGS),              // Pipeline stages for A (0-2)
                .BCASCREG(INTERNAL_REGS),          // Number of pipeline stages between B/BCIN and BCOUT (0-2)
                .BREG(INTERNAL_REGS),              // Pipeline stages for B (0-2)
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
                .PCOUT(),                // 58-bit output: Cascade output
                // Control outputs: Control Inputs/Status Bits
                .OVERFLOW(),                        // 1-bit output: Overflow in add/acc
                .PATTERNBDETECT(),                  // 1-bit output: Pattern bar detect
                .PATTERNDETECT(),                   // 1-bit output: Pattern detect
                .UNDERFLOW(),                       // 1-bit output: Underflow in add/acc
                // Data outputs: Data Ports
                .CARRYOUT(),                        // 4-bit output: Carry
                .P(pout[i][j]),                     // 58-bit output: Primary data
                .XOROUT(),                          // 8-bit output: XOR data
                // Cascade inputs: Cascade Ports
                .ACIN('x),                          // 34-bit input: A cascade data
                .BCIN('x),                          // 24-bit input: B cascade
                .CARRYCASCIN('x),                   // 1-bit input: Cascade carry
                .MULTSIGNIN('x),                    // 1-bit input: Multiplier sign cascade
                .PCIN('x),  // 58-bit input: P cascade
                // Control inputs: Control Inputs/Status Bits
                .ALUMODE(4'h0),                     // 4-bit input: ALU control
                .CARRYINSEL('0),                    // 3-bit input: Carry select
                .CLK(clk),                          // 1-bit input: Clock
                .INMODE({
                        INTERNAL_REGS==2 ? 1'b0 : 1'b1,
                        2'b00,
                        1'b0,
                        INTERNAL_REGS==2 ? 1'b0 : 1'b1
                }),                                 // 5-bit input: INMODE control
                .NEGATE('0),                        // 3-bit input: Negates the input of the multiplier
                .OPMODE('0),                        // 9-bit input: Operation mode
                // Data inputs: Data Ports
                .A({ 7'bx, a_in_i[j] }),            // 34-bit input: A data
                .B(b_in_i[i][j]),                   // 24-bit input: B data
                .C('x),                             // 58-bit input: C data
                .CARRYIN('0),                       // 1-bit input: Carry-in
                .D('x),                             // 27-bit input: D data
                // Reset/Clock Enable inputs: Reset/Clock Enable Inputs
                .ASYNC_RST('0),                     // 1-bit input: Asynchronous reset for all registers.
                .CEA1(en),                          // 1-bit input: Clock enable for 1st stage AREG
                .CEA2(INTERNAL_REGS==2 ? en : '0),  // 1-bit input: Clock enable for 2nd stage AREG
                .CEAD('0),                          // 1-bit input: Clock enable for ADREG
                .CEALUMODE('0),                     // 1-bit input: Clock enable for ALUMODE
                .CEB1(en),                          // 1-bit input: Clock enable for 1st stage BREG
                .CEB2(INTERNAL_REGS==2 ? en : '0),  // 1-bit input: Clock enable for 2nd stage BREG
                .CEC('0),                           // 1-bit input: Clock enable for CREG
                .CECARRYIN('0),                     // 1-bit input: Clock enable for CARRYINREG
                .CECTRL(en),                        // 1-bit input: Clock enable for OPMODEREG and CARRYINSELREG
                .CED('0),                           // 1-bit input: Clock enable for DREG
                .CEINMODE(en),                      // 1-bit input: Clock enable for INMODEREG
                .CEM(en),                           // 1-bit input: Clock enable for MREG
                .CEP(PREG && en),                   // 1-bit input: Clock enable for PREG
                .RSTA('0),                          // 1-bit input: Reset for AREG
                .RSTALLCARRYIN('0),                 // 1-bit input: Reset for CARRYINREG
                .RSTALUMODE('0),                    // 1-bit input: Reset for ALUMODEREG
                .RSTB('0),                          // 1-bit input: Reset for BREG
                .RSTC('0),                          // 1-bit input: Reset for CREG
                .RSTCTRL(rst),                      // 1-bit input: Reset for OPMODEREG and CARRYINSELREG
                .RSTD('0),                          // 1-bit input: Reset for DREG and ADREG
                .RSTINMODE(rst),                    // 1-bit input: Reset for INMODE register
                .RSTM('0),                          // 1-bit input: Reset for MREG
                .RSTP('0)                           // 1-bit input: Reset for PREG
            );
        end
    end

//-------------------- Instantiate accumulation --------------------\\

    acc_stage #(
        .CHAINLEN(CHAINLEN),
        .PE(PE),
        .ACCU_WIDTH(ACCU_WIDTH),
        .TH(TH)
    ) inst_acc_stage (
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
