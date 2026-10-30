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

	int unsigned  TH = 1,	// Tile height: number of interleaved input vectors

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
		if(!(VERSION inside { DSP48E1, DSP48E2, DSP58 })) begin
			$error("%m: VERSION=%0d is not a supported DSP version (1=DSP48E1, 2=DSP48E2, 3=DSP58).", VERSION);
			$finish;
		end
		if(PE < 1) begin
			$error("%m: PE=%0d must be at least 1.", PE);
			$finish;
		end
		if(SIMD < 1) begin
			$error("%m: SIMD=%0d must be at least 1.", SIMD);
			$finish;
		end
		if(WEIGHT_WIDTH < 2) begin
			$error("%m: WEIGHT_WIDTH=%0d is below the minimum of 2 bits.", WEIGHT_WIDTH);
			$finish;
		end
		if(ACTIVATION_WIDTH < 2) begin
			$error("%m: ACTIVATION_WIDTH=%0d is below the minimum of 2 bits.", ACTIVATION_WIDTH);
			$finish;
		end
		if(WEIGHT_WIDTH > A_WIDTH) begin
			$error("%m: WEIGHT_WIDTH=%0d exceeds the A datapath width of %0d bits.", WEIGHT_WIDTH, A_WIDTH);
			$finish;
		end
		if(ACTIVATION_WIDTH - SIGNED_ACTIVATIONS >= B_WIDTH) begin
			$error("%m: ACTIVATION_WIDTH=%0d (%s) does not fit the B datapath of %0d bits (max %0d).",
				ACTIVATION_WIDTH, SIGNED_ACTIVATIONS ? "signed" : "unsigned",
				B_WIDTH, B_WIDTH - 1 + SIGNED_ACTIVATIONS);
			$finish;
		end
		if(ACCU_WIDTH < WEIGHT_WIDTH + ACTIVATION_WIDTH - SIGNED_ACTIVATIONS) begin
			$error("%m: ACCU_WIDTH=%0d is too narrow for a %0d-bit product (%s activations need WEIGHT_WIDTH + ACTIVATION_WIDTH%s = %0d).",
				ACCU_WIDTH, WEIGHT_WIDTH + ACTIVATION_WIDTH - SIGNED_ACTIVATIONS,
				SIGNED_ACTIVATIONS ? "signed" : "unsigned",
				SIGNED_ACTIVATIONS ? " - 1"   : "",
				WEIGHT_WIDTH + ACTIVATION_WIDTH - SIGNED_ACTIVATIONS);
			$finish;
		end
	end

	localparam int unsigned  MIN_LANE_WIDTH = WEIGHT_WIDTH + ACTIVATION_WIDTH - 1;
	// number of lanes: for only 1 lane, NARROW_WEIGHTS makes no difference
	localparam int unsigned  NUM_LANES = A_WIDTH <= WEIGHT_WIDTH? 1 : 1 + (A_WIDTH - !NARROW_WEIGHTS - WEIGHT_WIDTH) / MIN_LANE_WIDTH;
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
	// Depth: 4 cycles for DSP (B1/INMODE + AD/B2 + M + P) + external SIMD reduction
	localparam int unsigned  PIPELINE_DEPTH = 4 + $clog2(SIMD) + (SIMD == 1) + 1;
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

		// Accumulation clearing — shared across SIMD (depends only on L, en, rst)
		uwire  clr;
		if(1) begin : blkClr
			case(TH)
			0: initial begin
				$error("%m: TH must be non-zero.");
				$finish;
			end
			1: assign  clr = L[3];
			2: assign  clr = L[4];
			default: begin
				logic  ClrLine[TH-2] = '{ default: 1 };
				always_ff @(posedge clk) begin
					if(rst)  ClrLine <= '{ default: 1 };
					else if(en) begin
						ClrLine[$high(ClrLine)] <= L[4];
						for(int unsigned  i = 0; i < $high(ClrLine); i++)  ClrLine[i] <= ClrLine[i+1];
					end
				end
				assign  clr = ClrLine[0];
			end
			endcase
		end : blkClr

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

			// Accumulation Connectivity
			uwire [P_WIDTH-1:0]  cc;	// Fabric feedback (not used for TH=1)
			uwire [P_WIDTH-1:0]  pp;

			if(1) begin : blkCrossLane

				// Context Assembly for Cross-Lane Carry Inference
				// For TH<=1: combinational from pp. For TH>=2: registered from cc.
				logic        pp_ctx_msb;
				logic [1:0]  pp_ctx_suf[NUM_LANES-1:1];
				if(1) begin : blkCtx
					uwire [P_WIDTH-1:0]  pp_ctx_src = TH <= 1? pp : cc;

					uwire        pp_ctx0_msb = pp_ctx_src[$left(pp_ctx_src)];
					uwire [1:0]  pp_ctx0_suf[NUM_LANES-1:1];
					for(genvar  i = 1; i < NUM_LANES; i++)  assign  pp_ctx0_suf[i] = pp_ctx_src[OFFSETS[i]+:2];

					if(TH <= 1) begin
						assign	pp_ctx_msb = pp_ctx0_msb;
						assign	pp_ctx_suf = pp_ctx0_suf;
					end
					else begin
						logic        PpCtxMsb = 'x;
						logic [1:0]  PpCtxSuf[NUM_LANES-1:1] = '{ default: 'x };
						always_ff @(posedge clk) begin
							if(rst) begin
								PpCtxMsb <= 'x;
								PpCtxSuf <= '{ default: 'x };
							end
							else if(en) begin
								PpCtxMsb <= pp_ctx0_msb;
								PpCtxSuf <= pp_ctx0_suf;
							end
						end
						assign	pp_ctx_msb = PpCtxMsb;
						assign	pp_ctx_suf = PpCtxSuf;
					end
				end : blkCtx

				// External Canary Pipeline
				// X1: input capture (aligns with DSP stage 1: A1/B1/D1/INMODE)
				// X2: pre-adder (aligns with AD2/B2)
				// X3: multiply (aligns with M3)
				// X4: LSB expectation upon accumulation (aligns with P4)
				logic [1:0]  X1[NUM_LANES-1:1] = '{ default: 0 };
				logic [1:0]  X2[NUM_LANES-1:1] = '{ default: 0 };
				logic [1:0]  X3[NUM_LANES-1:1] = '{ default: 0 };
				logic [1:0]  X4[NUM_LANES-1:1] = '{ default: 0 };
				logic  Clr4 = 1;
				always_ff @(posedge clk) begin
					if(rst) begin
						X1 <= '{ default: 0 };
						X2 <= '{ default: 0 };
						X3 <= '{ default: 0 };
						X4 <= '{ default: 0 };
						Clr4 <= 1;
					end
					else if(en) begin
						X1 <= xx;
						X2 <= X1;
						X3 <= X2;
						foreach(X4[i]) begin
							X4[i] <= X3[i] + (Clr4? 0 : pp_ctx_suf[i]);
						end
						Clr4 <= clr;
					end
				end

				// Derive actual cross-lane overflows
				for(genvar  i = 0; i < NUM_LANES-1; i++) begin
					assign	h3[s][i] = pp[OFFSETS[i+1]+:2] - X4[i+1];
				end

				// Overflow out of high lane
				logic  PZ = 0;
				always_ff @(posedge clk) begin
					if(rst)      PZ <= 0;
					else if(en)  PZ <= Clr4? 0 : pp_ctx_msb;
				end
				assign	h3[s][NUM_LANES-1] =
					( PZ && !pp[$left(pp)-:2])? +1 :
					(!PZ && &pp[$left(pp)-:2])? -1 : 0;

			end : blkCrossLane

			assign	p3[s] = pp;

			// Accumulation Cycling: fabric feedback for C-port (per SIMD)
			if(1) begin : blkAccCycling
				case(TH)
				1: assign  cc = 'x;
				2: assign  cc = pp;
				default: begin : genTHMulti
					logic [P_WIDTH-1:0]  Fb[TH-2] = '{ default: 'x };
					always_ff @(posedge clk) begin
						if(rst)      Fb <= '{ default: 'x };
						else if(en) begin
							Fb[$high(Fb)] <= pp;
							for(int unsigned  i = 0; i < $high(Fb); i++)  Fb[i] <= Fb[i+1];
						end
					end
					assign	cc = Fb[0];
				end : genTHMulti
				endcase
			end : blkAccCycling

			// Zeroing the partial product:
			//   The `zero` input suppresses the current partial product.
			//   Implemented via INMODE[2:1] controlling the pre-adder AD path:
			//     Normal: INMODE[1]=0 (A through), INMODE[2]=1 (D enabled) → AD = D − A
			//     Zero:   INMODE[1]=1 (A gated 0), INMODE[2]=0 (D off)    → AD = 0
			//   INMODEREG=1 latches zero in stage 1, aligning with A1/B1/D1.
			//   Stage 2 computes AD using the registered INMODE → no pipeline hazard.
			if(BEHAVIORAL) begin : genBehav

				// Stage #1: Input Capture (A1REG, B1REG, DREG, INMODEREG)
				logic signed [B_WIDTH-1:0]  B1 = 0;
				always_ff @(posedge clk) begin
					if(rst)      B1    <= 0;
					else if(en)  B1    <= bb;
				end
				logic signed [A_WIDTH-1:0]  A1 = 'x;
				logic signed [A_WIDTH-1:0]  D1 = 'x;
				logic  Zero1 = 'x;
				always_ff @(posedge clk) begin
					if(en) begin
						A1 <= aa;
						D1 <= dd;
						Zero1 <= zero;
					end
				end

				// Stage #2: Pre-adder + B pipeline (ADREG, B2REG)
				logic signed [B_WIDTH-1:0]  B2  = 0;
				always_ff @(posedge clk) begin
					if(rst)      B2    <= 0;
					else if(en)  B2    <= B1;
				end
				logic signed [A_WIDTH-1:0]  AD2 = 'x;
				always_ff @(posedge clk) begin
					if(en)  AD2 <= Zero1? 0 : (D1 - A1);
				end

				// Stage #3: Multiply + Accumulator Staging (MREG, CREG, OPMODEREG)
				logic signed [A_WIDTH+B_WIDTH-1:0]  M3 = 0;
				logic signed [P_WIDTH-1:0]  C3 = 0;
				logic  First3 = 0;
				always_ff @(posedge clk) begin
					if(rst) begin
						M3     <= 0;
						C3     <= 0;
						First3 <= 0;
					end
					else if(en) begin
						M3 <=
// synthesis translate off
							(AD2 === '0) || (B2 === '0)? 0 :
// synthesis translate on
							AD2 * B2;
						C3     <= cc;
						First3 <= clr;
					end
				end

				// Stage #4: Accumulate (PREG)
				logic signed [P_WIDTH-1:0]  P4 = 0;
				always_ff @(posedge clk) begin
					if(rst)      P4 <= 0;
					else if(en)  P4 <= M3 + (First3? 0 : (TH==1? P4 : C3));
				end
				assign	pp = P4;

				end : genBehav
`ifndef VERILATOR
			else begin : genDSP

				// Dynamic OPMODE: clr toggles Z-mux between accumulation source and 0.
				// TH=1:   Z=P(010) or Z=0(000). Single-bit toggle on bit[5].
				// TH>=2:  Z=C(011) or Z=0(000). Two-bit toggle on bits[5:4].
				// clr from blkClr → OPMODEREG (1-beat delay) → controls P4.
				localparam logic [6:0]  OPMODE_INVERSION = TH <= 1? 7'b010_01_01 : 7'b011_01_01;
				uwire [6:0]  opmode = { 1'b0, clr, TH <= 1? 1'b0 : clr, 4'b00_00 };

				// Dynamic INMODE: zero controls AD-path zeroing via INMODE[2:1].
				//   Effective (after IS_INMODE_INVERTED):
				//     Normal (zero=0): {0, 1, 1, 0, 1} → B2, D−A, D enabled, A through, A1
				//     Zero   (zero=1): {0, 1, 0, 1, 1} → B2, D−A, D off,    A gated,   A1
				//   Externally driven: only `zero` on bits [2:1], all else 0.
				//   INMODEREG=1 captures in stage 1 alongside A1/B1/D1.
				//   Stage 2 pre-adder sees the registered INMODE → no pipeline hazard.
				localparam logic [4:0]  INMODE_INVERSION = 5'b0_1_1_0_1;
				uwire [4:0]  inmode = { 3'b0, zero, zero };

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

					// Programmable Inversion Attributes
					.IS_INMODE_INVERTED(INMODE_INVERSION),
					.IS_OPMODE_INVERTED(OPMODE_INVERSION),

					// Register Control Attributes: Pipeline Register Configuration
					.ACASCREG(1),		// A-cascade tap after A1REG
					.ADREG(1),			// Pre-adder output register (stage 2)
					.ALUMODEREG(0),		// Number of pipeline stages for ALUMODE (0 or 1)
					.AREG(2),			// A1REG (stage 1) + A2REG (stage 2)
					.BCASCREG(1),		// B-cascade tap after B1REG
					.BREG(2),			// B1REG (stage 1) + B2REG (stage 2)
					.CARRYINREG(0),		// Number of pipeline stages for CARRYIN (0 or 1)
					.CARRYINSELREG(0),	// Number of pipeline stages for CARRYINSEL (0 or 1)
					.CREG(TH > 1),		// C register: enabled for TH>=2 (C-port feedback)
					.DREG(1),			// D register (stage 1)
					.INMODEREG(1),		// INMODE register (stage 1)
					.MREG(1),			// Multiplier register (stage 3)
					.OPMODEREG(1),		// OPMODE register for clear control
					.PREG(1)			// Accumulator register (stage 4)
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
					.INMODE(inmode),		// 5-bit input: INMODE control input
					.OPMODE(opmode),			// 7-bit input: Operation mode input

					// Data: 30-bit (each) input: Data Ports
					.A({5'b0, aa}),	// 30-bit input: A data input
					.B(				// 18-bit input: B data
// synthesis translate_off
						zero? '0 :
// synthesis translate_on
						bb
					),
					.C(				// 48-bit input: C data input
// synthesis translate_off
						clr? '0 :
// synthesis translate_on
						cc
					),
					.CARRYIN('0),	// 1-bit input: Carry input signal
					.D(dd),			// 25-bit input: D data input

					// Reset/Clock Enable: 1-bit (each) input: Reset/Clock Enable Inputs
					.CEA1(en),			// Stage 1: A1REG
					.CEA2(en),			// Stage 2: A2REG (unused by pre-adder, INMODE[0]=1 selects A1)
					.CEAD(en),			// Stage 2: ADREG
					.CEALUMODE('0),
					.CEB1(en),			// Stage 1: B1REG
					.CEB2(en),			// Stage 2: B2REG
					.CEC(TH > 1? en : 1'b0),	// Stage 3: CREG (TH≥2 feedback)
					.CECARRYIN('0),
					.CECTRL(en),		// OPMODEREG
					.CED(en),			// Stage 1: DREG
					.CEINMODE(en),		// Stage 1: INMODEREG
					.CEM(en),			// Stage 3: MREG
					.CEP(en),			// Stage 4: PREG
					.RSTA('0),			// A path: no reset
					.RSTB(rst),			// B path: reset (B1=B2=0 after rst)
					.RSTC(TH > 1? rst : 1'b0),
					.RSTD('0),			// D path: no reset
					.RSTALLCARRYIN('0),
					.RSTALUMODE('0),
					.RSTCTRL('0),
					.RSTINMODE('0),		// INMODE: no reset (B=0 masks first-beat hazard)
					.RSTM(rst),
					.RSTP(rst)
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

					// Programmable Inversion Attributes
					.IS_ALUMODE_INVERTED('0),
					.IS_CARRYIN_INVERTED('0),
					.IS_CLK_INVERTED('0),
					.IS_INMODE_INVERTED(INMODE_INVERSION),
					.IS_OPMODE_INVERTED({ 2'b00, OPMODE_INVERSION}),
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

					// Register Control Attributes: Pipeline Register Configuration
					.ACASCREG(1),		// A-cascade tap after A1REG
					.ADREG(1),			// Pre-adder output register (stage 2)
					.ALUMODEREG(0),
					.AREG(2),			// A1REG (stage 1) + A2REG (stage 2)
					.BCASCREG(1),		// B-cascade tap after B1REG
					.BREG(2),			// B1REG (stage 1) + B2REG (stage 2)
					.CARRYINREG(0),
					.CARRYINSELREG(0),
					.CREG(TH > 1),		// C register (stage 3, TH≥2 feedback)
					.DREG(1),			// D register (stage 1)
					.INMODEREG(1),		// INMODE register (stage 1)
					.MREG(1),			// Multiplier register (stage 3)
					.OPMODEREG(1),		// OPMODE register for clear control
					.PREG(1)			// Accumulator register (stage 4)
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
					.INMODE(inmode),			// 5-bit input: INMODE control
					.OPMODE({ 2'b00, opmode }),	// 9-bit input: Operation mode

					// Data inputs: Data Ports
					.A({3'b0, aa}),	// 30-bit input: A data
					.B(				// 18-bit input: B data
// synthesis translate_off
						zero? '0 :
// synthesis translate_on
						bb
					),
					.C(				// 48-bit input: C data
// synthesis translate_off
						clr? '0 :
// synthesis translate_on
						cc
					),
					.CARRYIN('0),	// 1-bit input: Carry-in
					.D(dd),			// 27-bit input: D data

					// Reset/Clock Enable inputs
					.CEA1(en),			// Stage 1: A1REG
					.CEA2(en),			// Stage 2: A2REG
					.CEAD(en),			// Stage 2: ADREG
					.CEALUMODE('0),
					.CEB1(en),			// Stage 1: B1REG
					.CEB2(en),			// Stage 2: B2REG
					.CEC(TH > 1? en : 1'b0),	// Stage 3: CREG (TH≥2 feedback)
					.CECARRYIN('0),
					.CECTRL(en),		// OPMODEREG
					.CED(en),			// Stage 1: DREG
					.CEINMODE(en),		// Stage 1: INMODEREG
					.CEM(en),			// Stage 3: MREG
					.CEP(en),			// Stage 4: PREG
					.RSTA('0),			// A path: no reset
					.RSTB(rst),			// B path: reset (B1=B2=0 after rst)
					.RSTC(TH > 1? rst : 1'b0),
					.RSTD('0),			// D path: no reset
					.RSTALLCARRYIN('0),
					.RSTALUMODE('0),
					.RSTCTRL('0),
					.RSTINMODE('0),		// INMODE: no reset (B=0 masks first-beat hazard)
					.RSTM(rst),
					.RSTP(rst)
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

					// Programmable Inversion Attributes
					.IS_ALUMODE_INVERTED('0),
					.IS_CARRYIN_INVERTED('0),
					.IS_CLK_INVERTED('0),
					.IS_INMODE_INVERTED(INMODE_INVERSION),
					.IS_NEGATE_INVERTED('0),
					.IS_OPMODE_INVERTED({ 2'b00, OPMODE_INVERSION}),
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

					// Register Control Attributes: Pipeline Register Configuration
					.ACASCREG(1),		// A-cascade tap after A1REG
					.ADREG(1),			// Pre-adder output register (stage 2)
					.ALUMODEREG(0),
					.AREG(2),			// A1REG (stage 1) + A2REG (stage 2)
					.BCASCREG(1),		// B-cascade tap after B1REG
					.BREG(2),			// B1REG (stage 1) + B2REG (stage 2)
					.CARRYINREG(0),
					.CARRYINSELREG(0),
					.CREG(TH > 1),		// C register (stage 3, TH≥2 feedback)
					.DREG(1),			// D register (stage 1)
					.INMODEREG(1),		// INMODE register (stage 1)
					.MREG(1),			// Multiplier register (stage 3)
					.OPMODEREG(1),		// OPMODE register for clear control
					.PREG(1),			// Accumulator register (stage 4)
					.RESET_MODE("SYNC")
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
					.INMODE(inmode),			// 5-bit input: INMODE control
					.NEGATE('0),				// 3-bit input: Negates the input of the multiplier
					.OPMODE({ 2'b00, opmode }),	// 9-bit input: Operation mode

					// Data inputs: Data Ports
					.A({7'b0, aa}),				// 34-bit input: A data
					.B(						 	// 24-bit input: B data
// synthesis translate_off
						zero? '0 :
// synthesis translate_on
						bb
					),
					.C(							// 58-bit input: C data
// synthesis translate_off
						clr? '0 :
// synthesis translate_on
						cc
					),
					.CARRYIN('0),				// 1-bit input: Carry-in
					.D(dd),						// 27-bit input: D data

					// Reset/Clock Enable inputs
					.ASYNC_RST('0),
					.CEA1(en),			// Stage 1: A1REG
					.CEA2(en),			// Stage 2: A2REG
					.CEAD(en),			// Stage 2: ADREG
					.CEALUMODE('0),
					.CEB1(en),			// Stage 1: B1REG
					.CEB2(en),			// Stage 2: B2REG
					.CEC(TH > 1? en : 1'b0),	// Stage 3: CREG (TH≥2 feedback)
					.CECARRYIN('0),
					.CECTRL(en),		// OPMODEREG
					.CED(en),			// Stage 1: DREG
					.CEINMODE(en),		// Stage 1: INMODEREG
					.CEM(en),			// Stage 3: MREG
					.CEP(en),			// Stage 4: PREG
					.RSTA('0),			// A path: no reset
					.RSTB(rst),			// B path: reset (B1=B2=0 after rst)
					.RSTC(TH > 1? rst : 1'b0),
					.RSTD('0),			// D path: no reset
					.RSTALLCARRYIN('0),
					.RSTALUMODE('0),
					.RSTCTRL('0),
					.RSTINMODE('0),		// INMODE: no reset (B=0 masks first-beat hazard)
					.RSTM(rst),
					.RSTP(rst)
				);
				default: initial begin
					$error("Unknown DSP version.");
					$finish;
				end
				endcase
			end : genDSP
`endif

		end : genSIMD

		// Stage #4: Potentially Multiple Cycles of Cross-SIMD Reduction
		// - binary reduction trees with SIMD leaf nodes for both the core lane outputs and the spill accumulation
		// - balanced tree construction with all fully occupied levels pipelined

		// Context-aligned clear for Hi4 accumulator.
		// Fixed offset PIPELINE_DEPTH-4 from clr (P4 stage) to Hi4 stage.
		// For TH>=2, delay L[PIPELINE_DEPTH-1] by TH-1 beats to match Hi4 ring depth.
		// Reset to 'x (Hi4 has explicit rst) so the delay line maps to SRL.
		uwire  hi_clr;
		if(TH <= 1) begin : genHiClr
			assign	hi_clr = L[PIPELINE_DEPTH-1];
		end : genHiClr
		else if(TH == 2) begin : genHiClr
			assign	hi_clr = L[PIPELINE_DEPTH];
		end : genHiClr
		else begin : genHiClr
			logic [TH-2:0]  HiClrDly = 'x;
			always_ff @(posedge clk) begin
				if(en) begin
					HiClrDly[$high(HiClrDly)] <= L[PIPELINE_DEPTH-1];
					for(int unsigned  i = 0; i < $high(HiClrDly); i++)  HiClrDly[i] <= HiClrDly[i+1];
				end
			end
			assign	hi_clr = HiClrDly[0];
		end : genHiClr

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
					add_multi #(.N(SIMD), .DEPTH(PIPELINE_DEPTH-6), .ARG_WIDTH(2), .ARG_LO(-1), .ARG_HI(1)) reduce (
						.clk, .rst, .en,
						.arg, .sum
					);
					for(genvar  s = 0; s < SIMD; s++)  assign  arg[s] = h3[s][i];

					// High Sideband Accumulation
					// h3 (and thus sum) is context-correct per beat.
					// Hi4 accumulates across SF beats, so for TH>=2 a TH-deep
					// ring keeps contexts separate (same structure as P3).
					logic signed [HI_WIDTH-1:0]  Hi4 = 0;

					// Context-aligned feedback: Hi4 itself for TH=1,
					// TH-deep ring tap for TH>=2.
					uwire signed [HI_WIDTH-1:0]  hh;
					if(TH <= 1) begin : genHiFb
						assign	hh = Hi4;
					end : genHiFb
					else begin : genHiFb
						logic signed [HI_WIDTH-1:0]  HiLine[TH-1] = '{ default: 0 };
						always_ff @(posedge clk) begin
							if(rst)  HiLine <= '{ default: 0 };
							else if(en) begin
								HiLine[0] <= Hi4;
								for(int j = 1; j < TH-1; j++)  HiLine[j] <= HiLine[j-1];
							end
						end
						assign	hh = HiLine[TH-2];
					end : genHiFb

					always_ff @(posedge clk) begin
						if(rst)  Hi4 <= 0;
						else if(en) begin
							automatic logic signed [HI_WIDTH:0]  h = $signed(hi_clr? {(HI_WIDTH){1'b0}} : hh) + $signed(sum);
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
					.N(SIMD), .DEPTH(PIPELINE_DEPTH-5),
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
