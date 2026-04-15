/******************************************************************************
 * Copyright (C) 2024, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	LUT-based dot product with fused accumulation.
 * @details	Drop-in replacement for DSP-based compute cores in the MVU.
 *		Uses a generated compressor tree for the reduction.
 *
 *		This file is a TEMPLATE — $COMP_MODULE_NAME$ is substituted
 *		at code generation time with the config-specific compressor
 *		module name (e.g. comp_8xs2s2).
 *****************************************************************************/

module dotp_comp #(
	int unsigned  PE,
	int unsigned  SIMD,
	int unsigned  WEIGHT_WIDTH,
	int unsigned  ACTIVATION_WIDTH,
	int unsigned  ACCU_WIDTH,
	bit  SIGNED_ACTIVATIONS = 0,
	int unsigned  COMP_PIPELINE_DEPTH = 1
)(
	// Global Control
	input	logic  clk,
	input	logic  rst,
	input	logic  en,

	// Input
	input	logic  last,
	input	logic  zero,
	input	logic signed [PE-1:0][SIMD-1:0][WEIGHT_WIDTH-1:0]   w,
	input	logic        [SIMD-1:0][ACTIVATION_WIDTH-1:0]       a,

	// Output
	output	logic  vld,
	output	logic signed [PE-1:0][ACCU_WIDTH-1:0]  p
);

	initial begin
		if(COMP_PIPELINE_DEPTH < 1) begin
			$error("%m: COMP_PIPELINE_DEPTH (%0d) must be >= 1.", COMP_PIPELINE_DEPTH);
			$finish;
		end
	end

	//-----------------------------------------------------------------------
	// Operand Mapping
	//
	// The `mul_comp_map` interface handles partial-product broadcasting
	// mul_comp_map requires NA >= NB.  Weights are always signed.
	// If activations are wider, swap operands so that ia gets the wider one.
	localparam bit  SWAPPED = ACTIVATION_WIDTH > WEIGHT_WIDTH;

	localparam int unsigned  NA = SWAPPED ? ACTIVATION_WIDTH : WEIGHT_WIDTH;
	localparam int unsigned  NB = SWAPPED ? WEIGHT_WIDTH     : ACTIVATION_WIDTH;
	localparam bit  SIGNED_A    = SWAPPED ? SIGNED_ACTIVATIONS : 1;  // weights always signed
	localparam bit  SIGNED_B    = SWAPPED ? 1 : SIGNED_ACTIVATIONS;

	// Input to Matric Broadcasting
	uwire [NA-1:0]  map0_ia = SWAPPED ? NA'(a[0])    : NA'(w[0][0]);
	uwire [NB-1:0]  map0_ib = SWAPPED ? NB'(w[0][0]) : NB'(a[0]);
	mul_comp_map #(.NA(NA), .NB(NB), .SIGNED_A(SIGNED_A), .SIGNED_B(SIGNED_B))
		map0 (.ia(map0_ia), .ib(map0_ib));
	localparam int unsigned  NM = $bits(map0.oa);

	//-----------------------------------------------------------------------
	// Pipeline shift register for last -> vld
/* verilator lint_off LITENDIAN */
	logic [1:COMP_PIPELINE_DEPTH]  L = '0;
/* verilator lint_on LITENDIAN */
	always_ff @(posedge clk) begin
		if(rst)      L <= '0;
		else if(en) begin
			L[1] <= last;
			for(int unsigned  i = 2; i <= COMP_PIPELINE_DEPTH; i++)
				L[i] <= L[i-1];
		end
	end
	assign	vld = L[COMP_PIPELINE_DEPTH];

	//-----------------------------------------------------------------------
	// PE-parallel compressor instances
	//-----------------------------------------------------------------------
	for(genvar  pe = 0; pe < PE; pe++) begin : genPE

		// Partial product matrix broadcasting
		uwire [NM-1:0]  oa[SIMD];
		uwire [NM-1:0]  ob[SIMD];
		for(genvar  i = 0; i < SIMD; i++) begin : genMap
			uwire [NA-1:0]  map_ia = SWAPPED ? NA'(a[i])    : NA'(w[pe][i]);
			uwire [NB-1:0]  map_ib = SWAPPED ? NB'(w[pe][i]) : NB'(a[i]);
			mul_comp_map #(.NA(NA), .NB(NB), .SIGNED_A(SIGNED_A), .SIGNED_B(SIGNED_B))
				map_i (.ia(map_ia), .ib(map_ib));
			assign	oa[i] = map_i.oa;
			assign	ob[i] = map_i.ob;
		end : genMap

		// Flatten all matrices column by column
		logic [SIMD*NM-1:0]  comp_a;
		logic [SIMD*NM-1:0]  comp_b;
		always_comb begin : blkFlatten
			automatic int unsigned  src_idx[SIMD] = '{ default: 0 };
			automatic int unsigned  dst_idx = 0;
			for(int unsigned  col = 0; col < map0.columns(); col++) begin
				for(int unsigned  k = 0; k < SIMD; k++) begin
					for(int unsigned  row = 0; row < map0.height(col); row++) begin
						comp_a[dst_idx] = oa[k][src_idx[k]];
						comp_b[dst_idx] = ob[k][src_idx[k]];
						src_idx[k]++;
						dst_idx++;
					end
				end
			end
		end : blkFlatten

		// Compressor with fused accumulation
		// $COMP_MODULE_NAME$ is replaced at code generation time with the
		// config-specific compressor module (e.g. comp_8xs2s2).
		uwire [ACCU_WIDTH-1:0]  comp_out;
		$COMP_MODULE_NAME$ comp_inst (
			.clk,
			.in(comp_b),
			.in_2(comp_a),
			.rst(rst || last),
			.en_neg(rst || zero),
			.en(en),
			.out(comp_out)
		);

		assign	p[pe] = $signed(comp_out);

	end : genPE

endmodule : dotp_comp
