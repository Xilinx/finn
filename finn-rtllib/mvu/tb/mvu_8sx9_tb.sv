/******************************************************************************
 * Copyright (C) 2022, Advanced Micro Devices, Inc.
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
 * @brief	Testbench for MVU core compute kernel.
 *****************************************************************************/

module mvu_8sx9_tb();

//-------------------- Simulation parameters --------------------\\
	// Matrix & parallelism config
	localparam int unsigned MH = 256;
	localparam int unsigned PE = 16;
	localparam int unsigned MW = 600;
	localparam int unsigned SIMD = 60;
	localparam int unsigned SEGMENTLEN = 4;
	// Bit-width config
	localparam int unsigned ACTIVATION_WIDTH = 8;
	localparam int unsigned WEIGHT_WIDTH = 4;
	localparam bit SIGNED_ACTIVATIONS = 1;
	// Simulation constants
	localparam int unsigned NF = MH/PE;
	localparam int unsigned SF = MW/SIMD;
	localparam int unsigned NUM_OF_DSP = SIMD/3;

	typedef logic [SIMD-1:0][ACTIVATION_WIDTH-1:0] activation_t;
	typedef activation_t activation_vector_t[SF];

	function activation_vector_t init_ACTIVATIONS;
		automatic activation_vector_t res;
		std::randomize(res);
		return res;
	endfunction : init_ACTIVATIONS

	typedef logic [PE-1:0][SIMD-1:0][WEIGHT_WIDTH-1:0] weight_t;
	typedef weight_t weight_matrix_t[NF][SF];

	function weight_matrix_t init_WEIGHTS;
		automatic weight_matrix_t res;
		std::randomize(res);
		return res;
	endfunction : init_WEIGHTS;

	typedef logic signed [PE-1:0][57:0] output_t;
	typedef output_t output_vector_t [NF];

	function output_vector_t check_output(activation_vector_t a, weight_matrix_t w);
		automatic output_vector_t res = '{default: 0};
		for (int j = 0; j<MH; j++) begin
			for (int i = 0; i<MW; i++) begin
				res[j/PE][j%PE] = $signed(res[j/PE][j%PE]) + $signed(a[i/SIMD][i%SIMD]) * $signed(w[j/PE][i/SIMD][j%PE][i%SIMD]);
			end
		end
		return res;
	endfunction : check_output;

	logic clk = 0;
	always #5ns clk = !clk;

	logic rst;
	initial begin
		rst = 1;
		repeat(16) @(posedge clk);
		rst <= 0;
	end

	logic last;
	logic zero;
	logic vld;
	activation_t a;
	weight_t w;
	output_t p;
	// Reference signals
	activation_vector_t ACTIVATIONS; //   [SF-1:0][SIMD-1:0][ACTIVATION_WIDTH-1:0]
	weight_matrix_t WEIGHTS; //           [NF-1:0][SF-1:0][PE-1:0][SIMD-1:0][WEIGHT_WIDTH-1:0]
	output_vector_t GOLDEN_OUTPUT; //     [NF-1:0][PE-1:0][57:0]
	// Counter for number of outputs (NF dimension) that are produced
	int NF_CNT = 0;

	initial begin
		ACTIVATIONS = init_ACTIVATIONS();
		WEIGHTS = init_WEIGHTS();
		GOLDEN_OUTPUT = check_output(ACTIVATIONS, WEIGHTS);
		last = 0;
		zero = 0;
		a = 'x;
		w = 'x;

		@(posedge clk iff !rst);

		for (int j=0; j<NF; j++) begin
			for (int i=0; i<SF; i++) begin
				last <= (i==SF-1) ? 1 : 0;
				a <= ACTIVATIONS[i];
				w <= WEIGHTS[j][i];
				@(posedge clk iff en);
			end
		end

		last <= 0;
		zero <= 1;

		// Continue until all NF outputs are produced & compared
		@(posedge clk && (NF_CNT==NF));

		$finish;
	end

	logic en = 0;
	always_ff @(posedge clk) begin
		en <= ($urandom()%7 > 1) && !rst;
	end

	// Compare computed output against golden output when vld flag is raised by DUT
	always_ff @(posedge clk iff (vld && en)) begin
		foreach(p[i]) begin
			assert ($signed(p[i]) == $signed(GOLDEN_OUTPUT[NF_CNT][i])) $display(">>> [t=%0t] Test succeeded (NF=%0d)! Computed / GOLDEN = %0d / %0d", $time, NF_CNT, $signed(p[i]), $signed(GOLDEN_OUTPUT[NF_CNT][i]));
			else begin
				$error(">>> [t=%0t] TEST failed (NF=%0d)! Computed / GOLDEN = %0d / %0d", $time, NF_CNT, $signed(p[i]), $signed(GOLDEN_OUTPUT[NF_CNT][i]));
				$stop;
			end
		end
		NF_CNT += 1;
	end

	// Instantiate DUT
	mvu_8sx9 #(
		.PE(PE),
		.SIMD(SIMD),
		.WEIGHT_WIDTH(WEIGHT_WIDTH),
		.SIGNED_ACTIVATIONS(SIGNED_ACTIVATIONS),
		.ACTIVATION_WIDTH(ACTIVATION_WIDTH),
		.SEGMENTLEN(SEGMENTLEN)
	)
	dut (
		.clk, .rst, .en, .last, .zero, .a, .w, .vld, .p
	);

endmodule : mvu_8sx9_tb
