/******************************************************************************
 * Copyright (C) 2026, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Standalone testbench for add_multi compressor (comp_NuW_dD).
 *		Tests the compressor directly without requiring add_multi.sv.
 *
 * Template placeholders expanded by run_add_multi_comp_tests.sh:
 *   {n}           - Number of addends
 *   {arg_width}   - Bit width of each addend
 *   {depth}       - Pipeline depth of compressor
 *   {label}       - Configuration label (e.g. n8_w4_p2)
 *   {comp_module} - Generated compressor module name (e.g. comp_8u4_d0)
 *****************************************************************************/

module add_multi_comp_{label}_tb;

	localparam int unsigned  N         = {n};
	localparam int unsigned  ARG_WIDTH = {arg_width};
	localparam int unsigned  DEPTH     = {depth};
	localparam int unsigned  IN_WIDTH  = N * ARG_WIDTH;
	// Use same formula as mvu_pkg::sumwidth() for consistency
	localparam int unsigned  SUM_WIDTH = $clog2(N) + ARG_WIDTH;
	localparam int unsigned  ROUNDS    = 257;

	//-----------------------------------------------------------------------
	// Global Control
	logic  clk = 0;
	always #5ns clk = !clk;

	logic  rst = 1;
	initial begin
		repeat(16) @(posedge clk);
		rst <= 0;
	end

	bit  done = 0;
	always_comb begin
		if(done)  $finish;
	end

	//-----------------------------------------------------------------------
	// DUT: direct compressor instantiation
	logic [IN_WIDTH-1:0]   in;
	logic [SUM_WIDTH-1:0]  out;

	{comp_module} dut (
		.clk,
		.in,
		.out
	);

	//-----------------------------------------------------------------------
	// Transpose function: convert row-major to column-major format.
	//
	// The compressor expects inputs in column-major (bit-slice) order:
	//   in[0..N-1]       = bit 0 of all N addends
	//   in[N..2N-1]      = bit 1 of all N addends
	//   ...
	//   in[(W-1)*N..W*N-1] = bit W-1 of all N addends
	//
	// This matches the transpose in add_multi.sv CATCH_COMP macro:
	//   assign in[j*N+i] = arg[i][j];
	//
	// Without this transpose, addend bits would be misaligned and produce
	// incorrect sums.
	//-----------------------------------------------------------------------
	function automatic logic [IN_WIDTH-1:0] transpose(
		input logic [IN_WIDTH-1:0] row_major
	);
		logic [IN_WIDTH-1:0] col_major;
		for(int i = 0; i < N; i++) begin
			for(int j = 0; j < ARG_WIDTH; j++) begin
				col_major[j*N + i] = row_major[i*ARG_WIDTH + j];
			end
		end
		return col_major;
	endfunction

	//-----------------------------------------------------------------------
	// Input Feed
	int  Q[$];
	initial begin
		in = 'x;
		@(posedge clk iff !rst);

		repeat(ROUNDS) begin
			automatic logic [IN_WIDTH-1:0]  aa;
			automatic int  exp = 0;
			void'(std::randomize(aa));

			// Compute expected sum from row-major input
			for(int unsigned i = 0; i < N; i++) begin
				exp += aa[i*ARG_WIDTH +: ARG_WIDTH];
			end

			// Transpose to column-major before feeding compressor
			in <= transpose(aa);
			Q.push_back(exp);
			@(posedge clk);
		end

		in <= 'x;
		repeat(DEPTH + 10) @(posedge clk);

		assert(Q.size == 0) else begin
			$error("Missing %0d outputs.", Q.size);
		end
		done = 1;
	end

	//-----------------------------------------------------------------------
	// Output Checker
	int unsigned  Checks = 0;
	int unsigned  Errors = 0;
	initial begin
		@(posedge clk iff !rst);
		repeat(DEPTH) @(posedge clk);
		repeat(ROUNDS) @(posedge clk) begin
			automatic int  exp = Q.pop_front();
			automatic int  hav = out;
			assert(hav == exp) else begin
				$error("Output mismatch %0d instead of %0d.", hav, exp);
				$stop;
				Errors <= Errors + 1;
			end
			Checks <= Checks + 1;
		end
	end

	final begin
		$display("Performed %0d checks with %0d errors.", Checks, Errors);
		assert(Checks == ROUNDS) else  $error("Unexpected number of checks: %0d instead of %0d.", Checks, ROUNDS);
	end

endmodule : add_multi_comp_{label}_tb
