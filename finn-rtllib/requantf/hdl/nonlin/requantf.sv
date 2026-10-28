/****************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 *
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 * @brief	FP32 requantization to N bits by clip(round(x*scale+bias)).
 *
 * @description
 *	Performs a floating-point requantization from single-precision FP32
 *	inputs to N-bit integers by computing:
 *		round(x*scale+bias)
 *	with round-half-up (toward +infinity) tie-breaking.
 *
 *	The channel-specific single-precision scale and bias parameters are
 *	rotated round-robin through the list of C/PE channel-fold values.
 *
 *	Each PE lane uses a single DSPFP32 (via fmaf) for the fused
 *	multiply-add. Rounding is implemented by pre-adjusting the bias
 *	at elaboration time:
 *	  - unsigned: bias' = bias + 0.5
 *	  - signed:   bias' = bias + 2^(N-1) + 0.5
 *	The signed adjustment shifts the target range [−2^(N−1), 2^(N−1)−1]
 *	to [0, 2^N−1], allowing a unified unsigned clip path. The MSB of the
 *	extracted result is inverted to recover the two's complement encoding.
 *
 *	The FP32-to-integer conversion exploits small N: only N+2 exponent
 *	values produce non-trivial results, so the clip stage is an
 *	exponent-indexed MUX over fixed mantissa bit-slices rather than a
 *	barrel shifter.
 *
 *	Pipeline (5 stages):
 *	  Stages 1–4: fmaf computes x*scale + bias'       (DSPFP32, lat=4)
 *	  Stage 5:    exponent-indexed extraction + clip    (registered)
 ***************************************************************************/

module requantf #(
	int unsigned  N,       // Output Precision
	int unsigned  C,       // Channel count
	int unsigned  PE = 1,  // Vector parallelism, must divide C

	shortreal  SCALES[PE][C/PE],
	shortreal  BIASES[PE][C/PE],

	bit  SIGNED_OUT = 0,       // 0: unsigned [0, 2^N-1], 1: signed [-2^(N-1), 2^(N-1)-1]
	bit  FORCE_BEHAVIORAL = 0
)(
	input	logic  clk,
	input	logic  rst,

	input	logic [PE-1:0][31:0]  idat,
	input	logic  ivld,

	output	logic [PE-1:0][N-1:0]  odat,
	output	logic  ovld
);
`default_nettype none
	localparam int unsigned  CF = C/PE;  // Channel fold
	localparam int unsigned  LATENCY = 5;

	// Bias pre-adjustment: +0.5 for rounding, +2^(N-1) for signed offset
	localparam shortreal  BIAS_ADJ = SIGNED_OUT? 2.0**(N-1) + 0.5 : 0.5;

	//=== ROM Derivation Function ===========================================
	typedef bit [31:0]  rom_t[CF];
	function automatic rom_t derive_ROM(input shortreal  vals[CF], input shortreal  adj);
		rom_t  res;
		foreach(res[i])  res[i] = $shortrealtobits(vals[i] + adj);
		return  res;
	endfunction : derive_ROM

	//=== Parameter Constraints =============================================
	initial begin
		if(CF*PE != C) begin
			$error("%m: Parallelism PE=%0d does not divide channel count C=%0d.", PE, C);
			$finish;
		end
		if(N < 1) begin
			$error("%m: Output precision N=%0d must be at least 1.", N);
			$finish;
		end
	end

	//=== Channel Selector ==================================================
	int unsigned  cnl_sel;
	if(CF == 1)  assign  cnl_sel = 0;
	else begin : blkCnlCnt
		logic [$clog2(CF)-1:0]  CnlCnt = 0;
		always_ff @(posedge clk) begin
			if(rst)        CnlCnt <= 0;
			else if(ivld)  CnlCnt <= CnlCnt + (CnlCnt != CF-1? 1 : (2**$clog2(CF)-CF)+1);
		end
		assign	cnl_sel = CnlCnt;
	end

	//=== Valid Pipeline ====================================================
	// Stages 1–4 tracked inside fmaf; only the clip stage needs a register.
	uwire  fma_ovld;
	logic  ClipVld = 0;
	always_ff @(posedge clk) begin
		if(rst)  ClipVld <= 0;
		else     ClipVld <= fma_ovld;
	end
	assign	ovld = ClipVld;

	//=== PE Compute Lanes ==================================================
	for(genvar  pe = 0; pe < PE; pe++) begin : genPE

		//--- Stage 1–4: Fused Multiply-Add ---------------------------------
		uwire [31:0]  fma_out;
		if(1) begin : blkFMA
			// Pre-compute bias ROM with rounding/signed adjustment
			localparam rom_t  SCALE_ROM = derive_ROM(SCALES[pe], 0.0);
			localparam rom_t  BIAS_ROM  = derive_ROM(BIASES[pe], BIAS_ADJ);

			uwire [31:0]  scale = SCALE_ROM[cnl_sel];
			uwire [31:0]  bias  = BIAS_ROM [cnl_sel];
			uwire  fma_ovld_pe;
			fmaf #(.OP("ADD"), .FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)) fma (
				.clk, .rst,
				.a(idat[pe]), .b(scale), .c(bias), .ivld,
				.r(fma_out), .ovld(fma_ovld_pe)
			);
			if(pe == 0)  assign  fma_ovld = fma_ovld_pe;
		end : blkFMA

		//--- Stage 5: FP32 to N-bit Integer Clip ---------------------------
		// After bias pre-adjustment, in-range values are non-negative and
		// in [0, 2^N). The clip extracts N unsigned integer bits from the
		// FP32 value by truncation (floor for non-negative), then saturates.
		//
		// Exponent mapping (value = man × 2^(exp−150)):
		//   exp < 127:    |value| < 1   → 0
		//   exp ∈ [127, 127+N−1]:       → exponent-indexed bit-slice
		//   exp ≥ 127+N:                → saturate to 2^N−1
		//   sign = 1:                   → 0 (negative after adjustment)
		logic [N-1:0]  R5 = 'x;
		if(1) begin : blkClip
			uwire         sign = fma_out[31];
			uwire [ 7:0]  exp  = fma_out[30:23];
			uwire [23:0]  man  = { 1'b1, fma_out[22:0] };

			// Exponent-indexed MUX: for exp = 127+e, the integer part
			// is man >> (23−e), of which we take the low N bits.
			// e ranges 0..N−1; for typical N ≤ 24, shift is always ≥ 0.
			uwire [N-1:0]  mag;
			if(N == 1)  assign  mag = 1;  // only reached when !neg && !zero && !ovf
			else begin : genClipN
				// Exponent-indexed MUX over fixed mantissa bit-slices.
				// For exp = 127+e, the integer part is man >> (23−e).
				// The N-bit target implicitly truncates upper bits.
				localparam int unsigned  IDX_W = $clog2(N) > 0? $clog2(N) : 1;
				localparam int unsigned  MUX_SZ = 2**IDX_W;
				uwire [N-1:0]  mux_arm[MUX_SZ];
				for(genvar  e = 0; e < MUX_SZ; e++) begin : genArm
					assign  mux_arm[e] = e < N? man >> (23 - e) : '1;
				end : genArm

				uwire [IDX_W-1:0]  idx = exp - 127;
				assign  mag = mux_arm[idx];
			end : genClipN

			uwire  neg  = sign;
			uwire  zero = (exp < 127);
			uwire  ovf  = (exp >= 127 + N);

			always_ff @(posedge clk) begin
				if(rst)  R5 <= 'x;
				else begin
					R5 <=
						(neg || zero)? '0 :
						ovf          ? '1 :
						mag;
				end
			end
		end : blkClip

		// Signed output: invert MSB to convert offset-binary → two's complement
		if(SIGNED_OUT && N > 1)
			assign	odat[pe] = { ~R5[N-1], R5[N-2:0] };
		else if(SIGNED_OUT)
			assign	odat[pe] = ~R5;
		else
			assign	odat[pe] = R5;

	end : genPE

`default_nettype wire
endmodule : requantf
