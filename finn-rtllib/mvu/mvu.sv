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
 * @brief	Matrix Vector Unit (MVU) core compute kernel utilizing DSP48.
 *****************************************************************************/

module mvu #(
	int unsigned  VERSION = 1,	// Allowed versions - 1: DSP48E1, 2: DSP48E2, 3: DSP58

	int unsigned  PE,
	int unsigned  SIMD,
	int unsigned  WEIGHT_WIDTH,
	int unsigned  ACTIVATION_WIDTH,
	int unsigned  ACCU_WIDTH,

	bit  SIGNED_ACTIVATIONS = 0,
	bit  NARROW_WEIGHTS   = 0,	// Weights in (-W:W) rather than [-W:W) with W = 2**(WEIGHT_WIDTH-1)
	bit  FORCE_BEHAVIORAL = 0
)(
	// Global Control
	input	logic  clk,
	input	logic  rst,
	input	logic  en,

	// Input
	input	logic  last,
	input	logic  zero,	// ignore current inputs and force this partial product to zero
	input	logic signed [PE-1:0][SIMD-1:0][WEIGHT_WIDTH    -1:0]  w,	// signed weights, WEIGHT_WIDTH >= 2
	input	logic                [SIMD-1:0][ACTIVATION_WIDTH-1:0]  a,	// unsigned activations (override by SIGNED_ACTIVATIONS), ACTIVATION_WIDTH >= 2

	// Ouput
	output	logic  vld,
	output	logic signed [PE-1:0][ACCU_WIDTH-1:0]  p
);

	import  mvu_pkg::*;

	// For Verilator: always use behavioral code
	localparam bit  BEHAVIORAL =
`ifdef VERILATOR
		1 ||
`endif
		FORCE_BEHAVIORAL;

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
			assert(!Dirty || rst || !en || zero) else begin
				$warning("%m: Feeding input during DSP startup recovery. Expect functional errors.");
			end
		end
	end : blkRecoveryWatch

	//-----------------------------------------------------------------------
	// Determine version-specific constraints
	typedef enum { DSP48E1 = 1, DSP48E2 = 2, DSP58 = 3 }  dsp_version_e;
	localparam int unsigned  A_WIDTH = 25 + 2*(VERSION > 1);     	// Width of A datapath
	localparam int unsigned  B_WIDTH = 18 + 6*(VERSION > 2);     	// Width of B datapath
	localparam int unsigned  P_WIDTH = VERSION == DSP58? 58 : 48;	// Width of P datapath

	initial begin
		if(WEIGHT_WIDTH < 2) begin
			$error("%m: Requested WEIGHT_WIDTH=%0d below the minimum width of 2 bits.", WEIGHT_WIDTH);
			$finish;
		end
		if(ACTIVATION_WIDTH < 2) begin
			$error("%m: Requested ACTIVATION_WIDTH=%0d below the minimum width of 2 bits.", WEIGHT_WIDTH);
			$finish;
		end
		if(WEIGHT_WIDTH > A_WIDTH) begin
			$error("%m: Requested ACTIVATION_WIDTH=%0d beyond the size of the A datapath of %0d bits.", ACTIVATION_WIDTH, A_WIDTH);
			$finish;
		end
	end

	localparam int unsigned  MIN_LANE_WIDTH = WEIGHT_WIDTH + ACTIVATION_WIDTH - 1;
	// number of lanes: for only 1 lane, NARROW_WEIGHTS makes no difference
	localparam int unsigned  NUM_LANES = A_WIDTH == WEIGHT_WIDTH? 1 : 1 + (A_WIDTH - !NARROW_WEIGHTS - WEIGHT_WIDTH) / MIN_LANE_WIDTH;
	/**
	 * Lane Slicing
	 *	Assumptions:
	 *	 - Internal lane widths differ, at most, by a single bit.
	 *	 - The minimum lane width is WEIGHT_WIDTH + ACTIVATION_WIDTH - 1 bits
	 *	   so as to confine cross-lane overflows to {-1,0,1}.
	 *	 - The rightmost lane (#0) has the maximum internal width.
	 *	 - The leftmost lane (#NUM_LANES-1) extends into the wide DSP accumulation path and
	 *	   is typically constrained by ACCU_WIDTH rather than the next lane. If so,
	 *	   it doesn't have an external high extension.
	 *	 - The one but leftmost lane (#NUM_LANES-2) has the minimum internal width and, hence,
	 *	   the maximum external high extension.
	 */
	typedef int unsigned  lane_offset_v[NUM_LANES:0];
	function lane_offset_v sliceLanes();
		automatic lane_offset_v  res;

		// Determine number of bits beyond accommodating minimum lane width
		automatic int  bit_slack = A_WIDTH;
		// protect sign if not narrow, leftmost weight entry, minimum for rest of lanes
		bit_slack -= !NARROW_WEIGHTS + WEIGHT_WIDTH + (NUM_LANES - 1) * MIN_LANE_WIDTH;
		if(bit_slack < 0) begin
			localparam  dsp_version_e  VER = dsp_version_e'(VERSION);
			$error(
				"%m: Cannot accommodate %0d-bit %snarrow weights on %s.",
				WEIGHT_WIDTH, NARROW_WEIGHTS? "" : "non-", VER.name
			);
			$finish;
		end

		// Distribute slack bits preferring right lanes
		res[0] = 0;
		for(int unsigned  i = 1; i < NUM_LANES; i++) begin
			automatic int unsigned  extra = (bit_slack + (NUM_LANES-1-i)) / (NUM_LANES-i);
			res[i] = res[i-1] + MIN_LANE_WIDTH + extra;
			bit_slack -= extra;
		end

		// Last lane bounded by the smaller of ACCU_WIDTH or P datapath
		res[NUM_LANES] = res[NUM_LANES-1] + ACCU_WIDTH;
		if(res[NUM_LANES] > P_WIDTH)  res[NUM_LANES] = P_WIDTH;

		return  res;
	endfunction : sliceLanes
	localparam lane_offset_v  OFFSETS = sliceLanes();

	function int unsigned sum_width(input int unsigned  n, input int unsigned  w);
		return  w <= 16? $clog2(1 + n*(2**w - 1)) : w + $clog2(n);
	endfunction : sum_width
	function int unsigned lo_width(input int unsigned  i);
		return  OFFSETS[i+1] - OFFSETS[i];
	endfunction : lo_width
	/**
	 * Determine number of bits needed to accommodate the high part
	 *	1. For ACCU_WIDTH > lw:
	 *		- the number of bits is determined by the lowest possible value for the high part
	 *		- every accumulator from a lower lane can contribute to the high part of lane i
	 *		- the contribution from the k-th preceding lane (wrt i) is given by (-2**(d_k - 1)), where d_k = (ACCU_WIDTH - sum_{j=(i-k)}^i lw_j)
	 *		- additionally every SIMD element can contribute an additional (-1)
	 *		- the required number of bits is 1 + clog2([sum_{k=0}^i 2**(d_k - 1)] + SIMD)
	 *		- to avoid numerical overflows for large values of (ACCU_WIDTH-lw), use a simplified version
	 *		- use clog2(2**d_0) = d_0 which can be used if 2**d_0 >= [sum_{k=0}^i 2**(d_k - 1)] + SIMD
	 *		- this is equivalent to 2**(d_0-1) >= [sum_{k=1}^i 2**(d_k - 1)] + SIMD
	 *		- since the difference between the individual d values is at least MIN_LANE_WIDTH >= 3, it can be simplified:
	 *		- [sum_{k=1}^i 2**(d_k - 1)] <= 2 * 2**(d_1 - 1) = 2**d_1 <= 2**(d0-3)
	 *		- therefore 2**(d_0-1) - 2**(d0-3) = 3*2**(d_0-3) >= SIMD is a sufficient condition for the simplification
	 *		- applying clog2 requires strictly smaller and gives the final condition: clog2(SIMD) < clog2(3) + (d_0-3) = 2 + (d_0-3) = d_0 - 1
	 *	2. For ACCU_WIDTH <= lw:
	 *		- only the (-1) contributions from each SIMD element need to be considered, which require 1 + clog2(SIMD) bits
	 *	3. Special case: ACCU_WIDTH <= lw and i == NUM_LANES-1:
	 *		- no need to track the overflow as there is no next lane
	 */
	function int unsigned hi_width(input int unsigned  i);
		automatic int unsigned res;
		automatic int unsigned lw[NUM_LANES-1:0];
		foreach(lw[k]) begin
			lw[k] = lo_width(k);
		end
		if(ACCU_WIDTH <= lw[i]) res = i == NUM_LANES-1? 0 : 1 + $clog2(SIMD);
		else begin
			res = 1;
			if($clog2(SIMD) < ACCU_WIDTH-lw[i]-1) res += ACCU_WIDTH-lw[i];
			else begin
				automatic int unsigned total_ov = 0;
				automatic int unsigned d = 0;
				for(int k = i; k >= 0; k--) begin
					d += lw[k];
					total_ov += 2**(ACCU_WIDTH-d-1);
				end
				res += $clog2(total_ov + SIMD);
			end
		end

		return res;
	endfunction : hi_width
	localparam int unsigned  LO_WIDTH_MAX = lo_width(NUM_LANES-1);
	localparam int unsigned  HI_WIDTH_MAX = hi_width(NUM_LANES < 2? 0 : NUM_LANES-2);

	// Pipeline for last indicator flag
	// Depth: 3 cycles for DSP + external SIMD reduction
	localparam int unsigned  PIPELINE_DEPTH = 3 + $clog2(SIMD) + (SIMD == 1) + 1;
/* verilator lint_off LITENDIAN */
	logic [1:PIPELINE_DEPTH] L = '0;
/* verilator lint_on LITENDIAN */
	always_ff @(posedge clk) begin
		if(rst)      L <= '0;
		else if(en)  L <= { last, L[1:PIPELINE_DEPTH-1] };
	end
	assign	vld = L[PIPELINE_DEPTH];

	// Stages #1 - #3: DSP Lanes + cross-lane canaries duplicated with SIMD parallelism
	localparam int unsigned  PIPE_COUNT = 1 + (PE-1)/NUM_LANES;
	for(genvar  c = 0; c < PIPE_COUNT; c++) begin : genPipes

		localparam int unsigned  PE_BEG = NUM_LANES*c;
		localparam int unsigned  PE_END = PE < NUM_LANES*(c+1)? PE : NUM_LANES*(c+1);
		localparam int unsigned  PE_REM = NUM_LANES*(c+1) - PE_END;

		uwire        [P_WIDTH-1:0]  p3[SIMD];
		uwire signed [        1:0]  h3[SIMD][NUM_LANES];
		for(genvar  s = 0; s < SIMD; s++) begin : genSIMD

			// Input Lane Assembly
			uwire [B_WIDTH-1:0]  bb = { {(B_WIDTH-ACTIVATION_WIDTH){SIGNED_ACTIVATIONS && a[s][ACTIVATION_WIDTH-1]}}, a[s] };
			logic [A_WIDTH-1:0]  aa;
			logic [A_WIDTH-1:0]  dd;
			logic [1:0]  xx[(NUM_LANES-1):1];
			if(1) begin : blkVectorize
				uwire signed [(WEIGHT_WIDTH-1):0]  ww[PE_END - PE_BEG];
				for(genvar  pe = 0; pe < PE_END - PE_BEG; pe++) begin
					assign	ww[pe] = $signed(w[PE_BEG + pe][s]);
					if(pe > 0) begin
						if(BEHAVIORAL)  assign  xx[pe + PE_REM] = zero? 0 : ww[pe] * a[s];
`ifndef VERILATOR
						else begin
							LUT6_2 #(.INIT(64'h0000_6AC0_0000_8888)) lut_x (
								.O6(xx[pe + PE_REM][1]),
								.O5(xx[pe + PE_REM][0]),
								.I5(1'b1),
								.I4(zero),
								.I3(ww[pe][1]),
								.I2(a[s][1]),
								.I1(ww[pe][0]),
								.I0(a[s][0])
							);
						end
`endif
					end
				end
				always_comb begin
					dd = '0;
					aa = '0;
					for(int unsigned  pe = 0; pe < PE_END - PE_BEG; pe++) begin
						automatic int unsigned  ofs = OFFSETS[pe + PE_REM];
						dd[ofs+:WEIGHT_WIDTH-1] = ww[pe][0+:WEIGHT_WIDTH-1];

						// The sign of the weights are generally put on the subtracted A port.
						// However, when coinciding with the actual sign bit position of the
						// multiplier input path, it also goes onto the D input. This prevents
						// sign extensions that may happen when a DSP primitive is auto-promoted
						// to a newer generation.
						if(ofs+WEIGHT_WIDTH-1 == A_WIDTH-1)  dd[ofs+WEIGHT_WIDTH-1] = ww[pe][WEIGHT_WIDTH-1];
						else                                 aa[ofs+WEIGHT_WIDTH-1] = ww[pe][WEIGHT_WIDTH-1];
					end
				end
				if(NARROW_WEIGHTS) begin : genNarrowCheck
					always_ff @(posedge clk iff en && !rst) begin
						foreach(ww[pe]) begin
							assert(zero || (ww[pe] !== -2**(WEIGHT_WIDTH-1))) else begin
								$warning("%m: Weight of %0x violates NARROW_WEIGHTS commitment.", ww[pe]);
							end
						end
					end
				end

			end : blkVectorize

			uwire [P_WIDTH-1:0]  pp;

			// Note: Since the product B * AD is computed,
			//       rst can be only applied to AD and zero only to B
			//       with the same effect as zeroing both.
			if(BEHAVIORAL) begin : genBehav

				// Stage #1: Input Refine
				logic signed [B_WIDTH-1:0]  B1  = 0;
				always_ff @(posedge clk) begin
					if(zero)     B1  <= 0;
					else if(en)  B1  <= bb;
				end

				logic signed [A_WIDTH-1:0]  AD1 = 0;
				always_ff @(posedge clk) begin
					if(rst)      AD1 <= 0;
					else if(en)  AD1 <= dd - aa;
				end

				// Stage #2: Multiply
				logic signed [A_WIDTH+B_WIDTH-1:0]  M2 = 0;
				always_ff @(posedge clk) begin
					if(rst)      M2 <= 0;
					else if(en)  M2 <=
// synthesis translate off
						(B1 === '0) || (AD1 === '0)? 0 :
// synthesis translate on
						B1 * AD1;
				end

				// Stage #3: Accumulate
				logic signed [P_WIDTH-1:0]  P3 = 0;
				always_ff @(posedge clk) begin
					if(rst)      P3 <= 0;
					else if(en)  P3 <= M2 + (L[3]? 0 : P3);
				end

				assign	pp = P3;
			end : genBehav
`ifndef VERILATOR
			else begin : genDSP
				localparam logic [6:0]  OPMODE_INVERSION = 7'b010_01_01;
				uwire [6:0]  opmode = { { 1'b0, L[2], 1'b0 }, 4'b00_00 };
				case(VERSION)
				DSP48E1: DSP48E1 #(
					// Feature Control Attributes: Data Path Selection
					.A_INPUT("DIRECT"),		// Selects A input source, "DIRECT" (A port) or "CASCADE" (ACIN port)
					.B_INPUT("DIRECT"),		// Selects B input source, "DIRECT" (B port) or "CASCADE" (BCIN port)
					.USE_DPORT("TRUE"),		// Select D port usage (TRUE or FALSE)
					.USE_MULT("MULTIPLY"),	// Select multiplier usage ("MULTIPLY", "DYNAMIC", or "NONE")
					.USE_SIMD("ONE48"),		// SIMD selection ("ONE48", "TWO24", "FOUR12")

					// Pattern Detector Attributes: Pattern Detection Configuration
					.AUTORESET_PATDET("NO_RESET"),		// "NO_RESET", "RESET_MATCH", "RESET_NOT_MATCH"
					.MASK('1),							// 48-bit mask value for pattern detect (1=ignore)
					.PATTERN('0),						// 48-bit pattern match for pattern detect
					.SEL_MASK("MASK"),					// "C", "MASK", "ROUNDING_MODE1", "ROUNDING_MODE2"
					.SEL_PATTERN("PATTERN"),			// Select pattern value ("PATTERN" or "C")
					.USE_PATTERN_DETECT("NO_PATDET"),	// Enable pattern detect ("PATDET" or "NO_PATDET")

					// Register Control Attributes: Pipeline Register Configuration
					.ACASCREG(0),		// Number of pipeline stages between A/ACIN and ACOUT (0, 1 or 2)
					.ADREG(1),			// Number of pipeline stages for pre-adder (0 or 1)
					.ALUMODEREG(0),		// Number of pipeline stages for ALUMODE (0 or 1)
					.AREG(0),			// Number of pipeline stages for A (0, 1 or 2)
					.BCASCREG(1),		// Number of pipeline stages between B/BCIN and BCOUT (0, 1 or 2)
					.BREG(1),			// Number of pipeline stages for B (0, 1 or 2)
					.CARRYINREG(0),		// Number of pipeline stages for CARRYIN (0 or 1)
					.CARRYINSELREG(0),	// Number of pipeline stages for CARRYINSEL (0 or 1)
					.CREG(0),			// Number of pipeline stages for C (0 or 1)
					.DREG(0),			// Number of pipeline stages for D (0 or 1)
					.INMODEREG(0),		// Number of pipeline stages for INMODE (0 or 1)
					.MREG(1),			// Number of multiplier pipeline stages (0 or 1)
					.OPMODEREG(1),		// Number of pipeline stages for OPMODE (0 or 1)
					.PREG(1)			// Number of pipeline stages for P (0 or 1)
				) dsp (
					// Cascade: 30-bit (each) output: Cascade Ports
					.ACOUT(),			// 30-bit output: A port cascade output
					.BCOUT(),			// 18-bit output: B port cascade output
					.CARRYCASCOUT(),	// 1-bit output: Cascade carry output
					.MULTSIGNOUT(),		// 1-bit output: Multiplier sign cascade output
					.PCOUT(),			// 48-bit output: Cascade output

					// Control: 1-bit (each) output: Control Inputs/Status Bits
					.OVERFLOW(),		 // 1-bit output: Overflow in add/acc output
					.PATTERNBDETECT(),	 // 1-bit output: Pattern bar detect output
					.PATTERNDETECT(),	 // 1-bit output: Pattern detect output
					.UNDERFLOW(),		 // 1-bit output: Underflow in add/acc output

					// Data: 4-bit (each) output: Data Ports
					.CARRYOUT(),	// 4-bit output: Carry output
					.P(pp),			// 48-bit output: Primary data output

					// Cascade: 30-bit (each) input: Cascade Ports
					.ACIN('x),			 // 30-bit input: A cascade data input
					.BCIN('x),			 // 18-bit input: B cascade input
					.CARRYCASCIN('x),	 // 1-bit input: Cascade carry input
					.MULTSIGNIN('x),	 // 1-bit input: Multiplier sign input
					.PCIN('x),			 // 48-bit input: P cascade input

					// Control: 4-bit (each) input: Control Inputs/Status Bits
					.CLK(clk),				// 1-bit input: Clock input
					.ALUMODE('0),			// 4-bit input: ALU control input
					.CARRYINSEL('0),		// 3-bit input: Carry select input
					.INMODE(5'b01100),		// 5-bit input: INMODE control input
					.OPMODE(opmode ^ OPMODE_INVERSION), // 7-bit input: Operation mode input

					// Data: 30-bit (each) input: Data Ports
					.A({5'b0, aa}),	// 30-bit input: A data input
					.B(bb),			// 18-bit input: B data input
					.C('x),			// 48-bit input: C data input
					.CARRYIN('0),	// 1-bit input: Carry input signal
					.D(dd),			// 25-bit input: D data input

					// Reset/Clock Enable: 1-bit (each) input: Reset/Clock Enable Inputs
					.CEA1('0),			// 1-bit input: Clock enable input for 1st stage AREG
					.CEA2('0),			// 1-bit input: Clock enable input for 2nd stage AREG
					.CEAD(en),			// 1-bit input: Clock enable input for ADREG
					.CEALUMODE('0),		// 1-bit input: Clock enable input for ALUMODERE
					.CEB1('0),			// 1-bit input: Clock enable input for 1st stage BREG
					.CEB2(en),			// 1-bit input: Clock enable input for 2nd stage BREG
					.CEC('0),			// 1-bit input: Clock enable input for CREG
					.CECARRYIN('0),		// 1-bit input: Clock enable input for CARRYINREG
					.CECTRL(en),		// 1-bit input: Clock enable input for OPMODEREG and CARRYINSELREG
					.CED('0),			// 1-bit input: Clock enable input for DREG
					.CEINMODE('0),		// 1-bit input: Clock enable input for INMODEREG
					.CEM(en),			// 1-bit input: Clock enable input for MREG
					.CEP(en),			// 1-bit input: Clock enable input for PREG
					.RSTA('0),			// 1-bit input: Reset input for AREG
					.RSTB(				// 1-bit input: Reset for BREG
// synthesis translate_off
						rst ||
// synthesis translate_on
						zero
					),
					.RSTC('0),			// 1-bit input: Reset for CREG
					.RSTD(				// 1-bit input: Reset for DREG and ADREG
// synthesis translate_off
						zero ||
// synthesis translate_on
						rst
					),
					.RSTALLCARRYIN('0),	// 1-bit input: Reset for CARRYINREG
					.RSTALUMODE('0),	// 1-bit input: Reset for ALUMODEREG
					.RSTCTRL('0),		// 1-bit input: Reset for OPMODEREG and CARRYINSELREG
					.RSTINMODE('0),		// 1-bit input: Reset for INMODE register
					.RSTM(rst),			// 1-bit input: Reset for MREG
					.RSTP(rst)			// 1-bit input: Reset for PREG
				);
				DSP48E2: DSP48E2 #(
					// Feature Control Attributes: Data Path Selection
					.AMULTSEL("AD"),	// Selects A input to multiplier (A, AD)
					.A_INPUT("DIRECT"),	// Selects A input source, "DIRECT" (A port) or "CASCADE" (ACIN port)
					.BMULTSEL("B"),		// Selects B input to multiplier (AD, B)
					.B_INPUT("DIRECT"),	// Selects B input source, "DIRECT" (B port) or "CASCADE" (BCIN port)
					.PREADDINSEL("A"),                 // Selects input to pre-adder (A, B)
					.RND('0),                          // Rounding Constant
					.USE_MULT("MULTIPLY"),             // Select multiplier usage (DYNAMIC, MULTIPLY, NONE)
					.USE_SIMD("ONE48"),                // SIMD selection (FOUR12, ONE58, TWO24)
					.USE_WIDEXOR("FALSE"),             // Use the Wide XOR function (FALSE, TRUE)
					.XORSIMD("XOR24_48_96"),       // Mode of operation for the Wide XOR (XOR12_22, XOR24_34_58_116)

					// Pattern Detector Attributes: Pattern Detection Configuration
					.AUTORESET_PATDET("NO_RESET"),     // NO_RESET, RESET_MATCH, RESET_NOT_MATCH
					.AUTORESET_PRIORITY("RESET"),      // Priority of AUTORESET vs. CEP (CEP, RESET).
					.MASK('1),                         // 58-bit mask value for pattern detect (1=ignore)
					.PATTERN('0),                      // 58-bit pattern match for pattern detect
					.SEL_MASK("MASK"),                 // C, MASK, ROUNDING_MODE1, ROUNDING_MODE2
					.SEL_PATTERN("PATTERN"),           // Select pattern value (C, PATTERN)
					.USE_PATTERN_DETECT("NO_PATDET"),  // Enable pattern detect (NO_PATDET, PATDET)

					// Programmable Inversion Attributes: Specifies built-in programmable inversion on specific pins
					.IS_ALUMODE_INVERTED('0),							// Optional inversion for ALUMODE
					.IS_CARRYIN_INVERTED('0),							// Optional inversion for CARRYIN
					.IS_CLK_INVERTED('0),								// Optional inversion for CLK
					.IS_INMODE_INVERTED('0),							// Optional inversion for INMODE
					.IS_OPMODE_INVERTED({ 2'b00, OPMODE_INVERSION}),	// Optional inversion for OPMODE
					.IS_RSTALLCARRYIN_INVERTED('0),						// Optional inversion for RSTALLCARRYIN
					.IS_RSTALUMODE_INVERTED('0),						// Optional inversion for RSTALUMODE
					.IS_RSTA_INVERTED('0),								// Optional inversion for RSTA
					.IS_RSTB_INVERTED('0),								// Optional inversion for RSTB
					.IS_RSTCTRL_INVERTED('0),							// Optional inversion for STCONJUGATE_A
					.IS_RSTC_INVERTED('0),								// Optional inversion for RSTC
					.IS_RSTD_INVERTED('0),								// Optional inversion for RSTD
					.IS_RSTINMODE_INVERTED('0),							// Optional inversion for RSTINMODE
					.IS_RSTM_INVERTED('0),								// Optional inversion for RSTM
					.IS_RSTP_INVERTED('0),								// Optional inversion for RSTP

					// Register Control Attributes: Pipeline Register Configuration
					.ACASCREG(0),                      // Number of pipeline stages between A/ACIN and ACOUT (0-2)
					.ADREG(1),                         // Pipeline stages for pre-adder (0-1)
					.ALUMODEREG(0),                    // Pipeline stages for ALUMODE (0-1)
					.AREG(0),                          // Pipeline stages for A (0-2)
					.BCASCREG(1),                      // Number of pipeline stages between B/BCIN and BCOUT (0-2)
					.BREG(1),                          // Pipeline stages for B (0-2)
					.CARRYINREG(0),                    // Pipeline stages for CARRYIN (0-1)
					.CARRYINSELREG(0),                 // Pipeline stages for CARRYINSEL (0-1)
					.CREG(0),                          // Pipeline stages for C (0-1)
					.DREG(0),                          // Pipeline stages for D (0-1)
					.INMODEREG(0),                     // Pipeline stages for INMODE (0-1)
					.MREG(1),                          // Multiplier pipeline stages (0-1)
					.OPMODEREG(1),                     // Pipeline stages for OPMODE (0-1)
					.PREG(1)                          // Number of pipeline stages for P (0-1)
				) dsp (
					// Cascade outputs: Cascade Ports
					.ACOUT(),			// 34-bit output: A port cascade
					.BCOUT(),			// 24-bit output: B cascade
					.CARRYCASCOUT(),	// 1-bit output: Cascade carry
					.MULTSIGNOUT(),		// 1-bit output: Multiplier sign cascade
					.PCOUT(),			// 58-bit output: Cascade output

					// Control outputs: Control Inputs/Status Bits
					.OVERFLOW(),		// 1-bit output: Overflow in add/acc
					.PATTERNBDETECT(),	// 1-bit output: Pattern bar detect
					.PATTERNDETECT(),	// 1-bit output: Pattern detect
					.UNDERFLOW(),		// 1-bit output: Underflow in add/acc

					// Data outputs: Data Ports
					.CARRYOUT(),		// 4-bit output: Carry
					.P(pp),				// 58-bit output: Primary data
					.XOROUT(),			// 8-bit output: XOR data

					// Cascade inputs: Cascade Ports
					.ACIN('x),			// 34-bit input: A cascade data
					.BCIN('x),			// 24-bit input: B cascade
					.CARRYCASCIN('x),	// 1-bit input: Cascade carry
					.MULTSIGNIN('x),	// 1-bit input: Multiplier sign cascade
					.PCIN('x),			// 58-bit input: P cascade

					// Control inputs: Control Inputs/Status Bits
					.CLK(clk),					// 1-bit input: Clock
					.ALUMODE(4'h0),				// 4-bit input: ALU control
					.CARRYINSEL('0),			// 3-bit input: Carry select
					.INMODE(5'b01100),			// 5-bit input: INMODE control
					.OPMODE({ 2'b00, opmode }),	// 9-bit input: Operation mode

					// Data inputs: Data Ports
					.A({3'b0, aa}),	// 30-bit input: A data
					.B(bb),			// 18-bit input: B data
					.C('x),			// 48-bit input: C data
					.CARRYIN('0),	// 1-bit input: Carry-in
					.D(dd),			// 27-bit input: D data

					// Reset/Clock Enable inputs: Reset/Clock Enable Inputs
					.CEA1('0),			// 1-bit input: Clock enable for 1st stage AREG
					.CEA2('0),			// 1-bit input: Clock enable for 2nd stage AREG
					.CEAD(en),			// 1-bit input: Clock enable for ADREG
					.CEALUMODE('0),		// 1-bit input: Clock enable for ALUMODE
					.CEB1('0),			// 1-bit input: Clock enable for 1st stage BREG
					.CEB2(en),			// 1-bit input: Clock enable for 2nd stage BREG
					.CEC('0),			// 1-bit input: Clock enable for CREG
					.CECARRYIN('0),		// 1-bit input: Clock enable for CARRYINREG
					.CECTRL(en),		// 1-bit input: Clock enable for OPMODEREG and CARRYINSELREG
					.CED('0),			// 1-bit input: Clock enable for DREG
					.CEINMODE('0),		// 1-bit input: Clock enable for INMODEREG
					.CEM(en),			// 1-bit input: Clock enable for MREG
					.CEP(en),			// 1-bit input: Clock enable for PREG
					.RSTA('0),			// 1-bit input: Reset for AREG
					.RSTB(				// 1-bit input: Reset for BREG
// synthesis translate_off
						rst ||
// synthesis translate_on
						zero
					),
					.RSTC('0),			// 1-bit input: Reset for CREG
					.RSTD(				// 1-bit input: Reset for DREG and ADREG
// synthesis translate_off
						zero ||
// synthesis translate_on
						rst
					),
					.RSTALLCARRYIN('0),	// 1-bit input: Reset for CARRYINREG
					.RSTALUMODE('0),	// 1-bit input: Reset for ALUMODEREG
					.RSTCTRL('0),		// 1-bit input: Reset for OPMODEREG and CARRYINSELREG
					.RSTINMODE('0),		// 1-bit input: Reset for INMODE register
					.RSTM(rst),			// 1-bit input: Reset for MREG
					.RSTP(rst)			// 1-bit input: Reset for PREG
				);
				DSP58: DSP58 #(
					// Feature Control Attributes: Data Path Selection
					.AMULTSEL("AD"),		// Selects A input to multiplier (A, AD)
					.A_INPUT("DIRECT"),		// Selects A input source, "DIRECT" (A port) or "CASCADE" (ACIN port)
					.BMULTSEL("B"),			// Selects B input to multiplier (AD, B)
					.B_INPUT("DIRECT"),		// Selects B input source, "DIRECT" (B port) or "CASCADE" (BCIN port)
					.DSP_MODE("INT24"),
					.PREADDINSEL("A"),			// Selects input to pre-adder (A, B)
					.RND('0),					// Rounding Constant
					.USE_MULT("MULTIPLY"),		// Select multiplier usage (DYNAMIC, MULTIPLY, NONE)
					.USE_SIMD("ONE58"),			// SIMD selection (FOUR12, ONE58, TWO24)
					.USE_WIDEXOR("FALSE"),		// Use the Wide XOR function (FALSE, TRUE)
					.XORSIMD("XOR24_34_58_116"),// Mode of operation for the Wide XOR (XOR12_22, XOR24_34_58_116)

					// Pattern Detector Attributes: Pattern Detection Configuration
					.AUTORESET_PATDET("NO_RESET"),		// NO_RESET, RESET_MATCH, RESET_NOT_MATCH
					.AUTORESET_PRIORITY("RESET"),		// Priority of AUTORESET vs. CEP (CEP, RESET).
					.MASK('1),							// 58-bit mask value for pattern detect (1=ignore)
					.PATTERN('0),						// 58-bit pattern match for pattern detect
					.SEL_MASK("MASK"),					// C, MASK, ROUNDING_MODE1, ROUNDING_MODE2
					.SEL_PATTERN("PATTERN"),			// Select pattern value (C, PATTERN)
					.USE_PATTERN_DETECT("NO_PATDET"),	// Enable pattern detect (NO_PATDET, PATDET)

					// Programmable Inversion Attributes: Specifies built-in programmable inversion on specific pins
					.IS_ALUMODE_INVERTED('0),							// Optional inversion for ALUMODE
					.IS_CARRYIN_INVERTED('0),							// Optional inversion for CARRYIN
					.IS_CLK_INVERTED('0),								// Optional inversion for CLK
					.IS_INMODE_INVERTED('0),							// Optional inversion for INMODE
					.IS_NEGATE_INVERTED('0),							// Optional inversion for NEGATE
					.IS_OPMODE_INVERTED({ 2'b00, OPMODE_INVERSION}),	// Optional inversion for OPMODE
					.IS_RSTALLCARRYIN_INVERTED('0),						// Optional inversion for RSTALLCARRYIN
					.IS_RSTALUMODE_INVERTED('0),						// Optional inversion for RSTALUMODE
					.IS_RSTA_INVERTED('0),								// Optional inversion for RSTA
					.IS_RSTB_INVERTED('0),								// Optional inversion for RSTB
					.IS_RSTCTRL_INVERTED('0),							// Optional inversion for STCONJUGATE_A
					.IS_RSTC_INVERTED('0),								// Optional inversion for RSTC
					.IS_RSTD_INVERTED('0),								// Optional inversion for RSTD
					.IS_RSTINMODE_INVERTED('0),							// Optional inversion for RSTINMODE
					.IS_RSTM_INVERTED('0),								// Optional inversion for RSTM
					.IS_RSTP_INVERTED('0),								// Optional inversion for RSTP

					// Register Control Attributes: Pipeline Register Configuration
					.ACASCREG(0),		// Number of pipeline stages between A/ACIN and ACOUT (0-2)
					.ADREG(1),			// Pipeline stages for pre-adder (0-1)
					.ALUMODEREG(0),		// Pipeline stages for ALUMODE (0-1)
					.AREG(0),			// Pipeline stages for A (0-2)
					.BCASCREG(1),		// Number of pipeline stages between B/BCIN and BCOUT (0-2)
					.BREG(1),			// Pipeline stages for B (0-2)
					.CARRYINREG(0),		// Pipeline stages for CARRYIN (0-1)
					.CARRYINSELREG(0),	// Pipeline stages for CARRYINSEL (0-1)
					.CREG(0),			// Pipeline stages for C (0-1)
					.DREG(0),			// Pipeline stages for D (0-1)
					.INMODEREG(0),		// Pipeline stages for INMODE (0-1)
					.MREG(1),			// Multiplier pipeline stages (0-1)
					.OPMODEREG(1),		// Pipeline stages for OPMODE (0-1)
					.PREG(1),			// Number of pipeline stages for P (0-1)
					.RESET_MODE("SYNC")	// Selection of synchronous or asynchronous reset. (ASYNC, SYNC)
				) dsp (
					// Cascade outputs: Cascade Ports
					.ACOUT(),			// 34-bit output: A port cascade
					.BCOUT(),			// 24-bit output: B cascade
					.CARRYCASCOUT(),	// 1-bit output: Cascade carry
					.MULTSIGNOUT(),		// 1-bit output: Multiplier sign cascade
					.PCOUT(),			// 58-bit output: Cascade output

					// Control outputs: Control Inputs/Status Bits
					.OVERFLOW(),		// 1-bit output: Overflow in add/acc
					.PATTERNBDETECT(),	// 1-bit output: Pattern bar detect
					.PATTERNDETECT(),	// 1-bit output: Pattern detect
					.UNDERFLOW(),		// 1-bit output: Underflow in add/acc

					// Data outputs: Data Ports
					.CARRYOUT(),		// 4-bit output: Carry
					.P(pp),				// 58-bit output: Primary data
					.XOROUT(),			// 8-bit output: XOR data

					// Cascade inputs: Cascade Ports
					.ACIN('x),			// 34-bit input: A cascade data
					.BCIN('x),			// 24-bit input: B cascade
					.CARRYCASCIN('x),	// 1-bit input: Cascade carry
					.MULTSIGNIN('x),	// 1-bit input: Multiplier sign cascade
					.PCIN('x),			// 58-bit input: P cascade

					// Control inputs: Control Inputs/Status Bits
					.CLK(clk),					// 1-bit input: Clock
					.ALUMODE(4'h0),				// 4-bit input: ALU control
					.CARRYINSEL('0),			// 3-bit input: Carry select
					.INMODE(5'b01100),			// 5-bit input: INMODE control
					.NEGATE('0),				// 3-bit input: Negates the input of the multiplier
					.OPMODE({ 2'b00, opmode }),	// 9-bit input: Operation mode

					// Data inputs: Data Ports
					.A({7'b0, aa}),				// 34-bit input: A data
					.B(bb),						// 24-bit input: B data
					.C('x),						// 58-bit input: C data
					.CARRYIN('0),				// 1-bit input: Carry-in
					.D(dd),						// 27-bit input: D data

					// Reset/Clock Enable inputs: Reset/Clock Enable Inputs
					.ASYNC_RST('0),		// 1-bit input: Asynchronous reset for all registers
					.CEA1('0),			// 1-bit input: Clock enable for 1st stage AREG
					.CEA2('0),			// 1-bit input: Clock enable for 2nd stage AREG
					.CEAD(en),			// 1-bit input: Clock enable for ADREG
					.CEALUMODE('0),		// 1-bit input: Clock enable for ALUMODE
					.CEB1('0),			// 1-bit input: Clock enable for 1st stage BREG
					.CEB2(en),			// 1-bit input: Clock enable for 2nd stage BREG
					.CEC('0),			// 1-bit input: Clock enable for CREG
					.CECARRYIN('0),		// 1-bit input: Clock enable for CARRYINREG
					.CECTRL(en),		// 1-bit input: Clock enable for OPMODEREG and CARRYINSELREG
					.CED('0),			// 1-bit input: Clock enable for DREG
					.CEINMODE('0),		// 1-bit input: Clock enable for INMODEREG
					.CEM(en),			// 1-bit input: Clock enable for MREG
					.CEP(en),			// 1-bit input: Clock enable for PREG
					.RSTA('0),			// 1-bit input: Reset for AREG
					.RSTB(				// 1-bit input: Reset for BREG
// synthesis translate_off
						rst ||
// synthesis translate_on
						zero
					),
					.RSTC('0),			// 1-bit input: Reset for CREG
					.RSTD(				// 1-bit input: Reset for DREG and ADREG
// synthesis translate_off
						zero ||
// synthesis translate_on
						rst
					),
					.RSTALLCARRYIN('0),	// 1-bit input: Reset for CARRYINREG
					.RSTALUMODE('0),	// 1-bit input: Reset for ALUMODEREG
					.RSTCTRL('0),		// 1-bit input: Reset for OPMODEREG and CARRYINSELREG
					.RSTINMODE('0),		// 1-bit input: Reset for INMODE register
					.RSTM(rst),			// 1-bit input: Reset for MREG
					.RSTP(rst)			// 1-bit input: Reset for PREG
				);
				default: initial begin
					$error("Unknown DSP version.");
					$finish;
				end
				endcase
			end : genDSP
`endif

			// External Canary Pipeline
			logic [1:0]  X1[NUM_LANES-1:1] = '{ default: 0 };
			logic [1:0]  X2[NUM_LANES-1:1] = '{ default: 0 };
			logic [1:0]  X3[NUM_LANES-1:1] = '{ default: 0 };
			always_ff @(posedge clk) begin
				if(rst) begin
					X1 <= '{ default: 0 };
					X2 <= '{ default: 0 };
					X3 <= '{ default: 0 };
				end
				else if(en) begin
					X1 <= xx;
					X2 <= X1;
					foreach(X3[i]) begin
						X3[i] <= X2[i] + (L[3]? 2'h0 : pp[OFFSETS[i]+:2]);
					end
				end
			end

			// Derive actual cross-lane overflows
			for(genvar  i = 0; i < NUM_LANES-1; i++) begin
				assign	h3[s][i] = pp[OFFSETS[i+1]+:2] - X3[i+1];
			end
			// Overflow out of high lane
			logic  PZ = 0;
			always_ff @(posedge clk) begin
				if(rst)      PZ <= 0;
				else if(en)  PZ <= L[3]? 0 : pp[$left(pp)];
			end
			assign	h3[s][NUM_LANES-1] =
				( PZ && !pp[$left(pp)-:2])? +1 :
				(!PZ && &pp[$left(pp)-:2])? -1 : 0;

			assign	p3[s] = pp;

		end : genSIMD

		// Stage #4: Potentially Multiple Cycles of Cross-SIMD Reduction
		// - binary reduction trees with SIMD leaf nodes for both the core lane outputs and the spill accumulation
		// - balanced tree construction with all fully occupied levels pipelined

		// Count leaves reachable from each node
		uwire signed [HI_WIDTH_MAX                 -1:0]  hi4[NUM_LANES];
		uwire        [sum_width(SIMD, LO_WIDTH_MAX)-1:0]  lo4[NUM_LANES];
		for(genvar  i = 0; i < NUM_LANES; i++) begin : genLanes

			// Conclusive high part accumulation
			if(i < PE_REM)  assign  hi4[i] = 0;
			else begin : genHi
				localparam int unsigned  HI_WIDTH = hi_width(i);
				if(HI_WIDTH == 0)  assign  hi4[i] = 0;
				else begin
					localparam int unsigned  SUM_WIDTH = sumwidth(SIMD, 2, -1, 1);

					uwire [1:0]  arg[SIMD];
					uwire [SUM_WIDTH-1:0]  sum;
					add_multi #(.N(SIMD), .DEPTH(PIPELINE_DEPTH-5), .ARG_WIDTH(2), .ARG_LO(-1), .ARG_HI(1)) reduce (
						.clk, .rst, .en,
						.arg, .sum
					);
					for(genvar  s = 0; s < SIMD; s++)  assign  arg[s] = h3[s][i];

					// High Sideband Accumulation
					logic signed [HI_WIDTH-1:0]  Hi4 = 0;
					always_ff @(posedge clk) begin
						if(rst)  Hi4 <= 0;
						else if(en) begin
							automatic logic signed [HI_WIDTH:0]  h = $signed(L[PIPELINE_DEPTH-1]? {(HI_WIDTH){1'b0}} : Hi4) + $signed(sum);
							assert(h[HI_WIDTH] === h[HI_WIDTH-1]) else begin
								$error("%m [%0d:%0d]: Accumulation overflow for ACCU_WIDTH=%0d", c, i, ACCU_WIDTH);
							end
							Hi4 <= h[HI_WIDTH-1:0];
						end
					end
					assign	hi4[i] = Hi4;
				end
			end : genHi

			// Conclusive low part accumulation (all unsigned arithmetic)
			if(i < PE_REM)  assign  lo4[i] = '0;
			else begin : genLo

				// Instantiate Adder Tree
				localparam int unsigned  LO_WIDTH = lo_width(i);
				localparam int unsigned  SUM_WIDTH = sumwidth(SIMD, LO_WIDTH);
				uwire [LO_WIDTH -1:0]  arg[SIMD];
				uwire [SUM_WIDTH-1:0]  sum;
				add_multi #(
					.N(SIMD), .DEPTH(PIPELINE_DEPTH-4),
					.ARG_WIDTH(LO_WIDTH),
					.RESET_ZERO(0)
				) reduce (
					.clk, .rst, .en,
					.arg, .sum
				);
				for(genvar  s = 0; s < SIMD; s++)  assign  arg[s] = p3[s][OFFSETS[i]+:LO_WIDTH];
				assign	lo4[i] = sum;

			end : genLo

		end : genLanes

		// Stage #5: Resolve lane totals
		logic signed [NUM_LANES-1:0][ACCU_WIDTH-1:0]  Res5 = '{ default: 'x };
		for(genvar  pe = 1; pe < NUM_LANES; pe++) begin : genRes
			always_ff @(posedge clk) begin
				if(rst)  Res5[pe] <= 'x;
				else if(en) begin
					Res5[pe] <= $signed({ hi4[pe], {(lo_width(pe)){1'b0}} }) + $signed({ 1'b0, lo4[pe] }) - hi4[pe-1];
				end
			end
		end : genRes
		always_ff @(posedge clk) begin
			if(rst)     Res5[0] <= 'x;
			else if(en) Res5[0] <= $signed({ hi4[0], {(lo_width(0)){1'b0}} }) + $signed({ 1'b0, lo4[0] });
		end

		// Output
		for(genvar  pe = PE_BEG; pe < PE_END; pe++) begin
			assign	p[pe] = Res5[pe - PE_BEG + PE_REM];
		end

	end : genPipes

endmodule : mvu
