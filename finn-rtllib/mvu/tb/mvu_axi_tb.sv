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
 * @brief	Testbench for MVU AXI wrapper module.
 *****************************************************************************/

module mvu_axi_tb();

//-------------------- Simulation parameters --------------------\\
	// Matrix & parallelism config
	localparam bit IS_MVU = 1;
	localparam string COMPUTE_CORE = "mvu_4sx4u";
	localparam int unsigned MW = 96;
	localparam int unsigned MH = 32;
	localparam int unsigned SIMD = 48;
	localparam int unsigned PE = 16;
	localparam int unsigned SEGMENTLEN = 2;
	localparam bit FORCE_BEHAVIORAL = 0;
	localparam bit M_REG_LUT = 1;
	// Bit-width config
	localparam int unsigned ACTIVATION_WIDTH = 4;
	localparam int unsigned WEIGHT_WIDTH = 4;
	localparam int unsigned ACCU_WIDTH = ACTIVATION_WIDTH+WEIGHT_WIDTH+$clog2(MW);
	localparam bit SIGNED_ACTIVATIONS = 1;
	// Simulation constants
	localparam int unsigned NF = MH/PE;
	localparam int unsigned SF = MW/SIMD;
	localparam int unsigned WEIGHT_WIDTH_BA = (PE*SIMD*WEIGHT_WIDTH+7)/8*8;
	localparam int unsigned ACTIVATION_WIDTH_BA = (SIMD*ACTIVATION_WIDTH+7)/8*8;
	localparam int unsigned WEIGHT_WIDTH_BA_DELTA = WEIGHT_WIDTH_BA - PE*SIMD*WEIGHT_WIDTH;
	localparam int unsigned ACTIVATION_WIDTH_BA_DELTA = ACTIVATION_WIDTH_BA - SIMD*ACTIVATION_WIDTH;
	localparam int unsigned OUTPUT_STREAM_WIDTH_BA = (PE*ACCU_WIDTH + 7)/8 * 8;

	// Generate clk and reset signal
	logic clk = 0;
	always #5ns clk = !clk;

	logic ap_rst_n = 0;
	initial begin
		repeat(16) @(posedge clk);
		ap_rst_n <= 1;
	end

	uwire ap_clk = clk;

	// Generate shared Activations
	typedef logic [SIMD-1:0][ACTIVATION_WIDTH-1:0] activation_t;
	typedef activation_t activation_vector_t[SF];

	function activation_vector_t init_ACTIVATIONS;
		automatic activation_vector_t res;
		std::randomize(res);
		return res;
	endfunction : init_ACTIVATIONS

	activation_vector_t ACTIVATIONS = init_ACTIVATIONS();

	// Run parallel instances across DSP versions and NARROW_WEIGHTS
	bit [2:1][1:0]  done = { 2: 2'b00, 1: 2'b01 }; // [ver][narrow]
	always_comb begin
		if(&done) begin
			$display("Test completed.");
			$finish;
		end
	end

	for(genvar  ver = 1; ver <= 2; ver++) begin : genVersion
		for(genvar  narrow = (ver == 1); narrow <= 1; narrow++) begin : genNarrowWide

		// Activations Feed
		struct {
			activation_t dat;
			logic vld;
			logic rdy;
		} activations;

		initial begin
			activations.vld = 0;
			activations.dat = 'X;
			@(posedge clk iff ap_rst_n);

			for(int unsigned  i = 0; i < SF; i++) begin
				while($urandom()%7 == 0) @(posedge clk);
				activations.dat <= ACTIVATIONS[i];
				activations.vld <= 1;
				@(posedge clk iff activations.rdy);
				activations.dat <= 'x;
				activations.vld <= 0;
			end
		end

		// Instance-specifc Weights (may be narrow)
		typedef logic [PE-1:0][SIMD-1:0][WEIGHT_WIDTH-1:0] weight_t;
		typedef weight_t weight_matrix_t[NF][SF];

		function weight_matrix_t init_WEIGHTS;
			automatic weight_matrix_t  res;
			std::randomize(res);
			if(narrow) begin  // increment all weights of -8
				for(int unsigned  nf = 0; nf < NF; nf++) begin
					for(int unsigned  sf = 0; sf < SF; sf++) begin
						for(int unsigned  pe = 0; pe < PE; pe++) begin
							for(int unsigned  simd = 0; simd < SIMD; simd++) begin
								if(res[nf][sf][pe][simd] == (1 << (WEIGHT_WIDTH-1))) begin
									res[nf][sf][pe][simd]++;
								end
							end
						end
					end
				end
			end
			return res;
		endfunction : init_WEIGHTS;

		weight_matrix_t WEIGHTS = init_WEIGHTS();

		// Weight Feed
		struct {
			weight_t dat;
			logic vld;
			logic rdy;
		} weights;

		initial begin
			weights.vld = 0;
			weights.dat = 'X;
			@(posedge clk iff ap_rst_n);

			weights.vld <= 1;
			for(int unsigned  i = 0; i < NF; i++) begin
				for(int unsigned  j = 0; j < SF; j++) begin
					weights.dat <= WEIGHTS[i][j];
					@(posedge clk iff weights.rdy);
				end
			end
			weights.vld <= 0;
			weights.dat <= 'x;
		end

		// Function to compute golden output
		// a: [SF][SIMD-1:0][ACTIVATION_WIDTH-1:0]
		// a: [SF][PE*SIMD-1:0][ACTIVATION_WIDTH-1:0]
		// w: [NF][SF][PE-1:0][SIMD-1:0][WEIGHT_WIDTH-1:0]
		typedef logic signed [PE-1:0][ACCU_WIDTH-1:0] output_t;
		typedef output_t output_vector_t [NF];

		struct {
			output_t dat;
			logic vld;
			logic rdy;
		} outputs;

		function output_vector_t check_output(activation_vector_t a, weight_matrix_t w);
			automatic output_vector_t res = '{default: 0};
			// The input stream will have the channels interleaved for VVU when PE>1
			// Hence, we need to 'untangle' the input stream, i.e. [..][SIMD*PE][..] --> [..][PE][SIMD][..]
			// Note that for each 'SIMD' (S) and 'PE' (P) element, we have something like:
			// (S_0, P_0), ..., (S_0, P_i), (S_1, P_0), ..., (S_1, P_i), ..., (S_i, P_i) which we need to 'untangle' to
			// (S_0, P_0), ..., (S_i, P_0), (S_0, P_1), ..., (S_i,, P_1), ..., (S_i, P_i)
			for (int i = 0; i < NF; i++) begin
				for (int j = 0; j < SF; j++) begin
					for (int k = 0; k < PE; k++) begin
						for (int l = 0; l < SIMD; l++) begin
							if (SIGNED_ACTIVATIONS)
								res[i][k] = $signed(res[i][k]) + $signed(a[j][l]) * $signed(w[i][j][k][l]);
							else
								res[i][k] = $signed(res[i][k]) + $signed({1'b0, a[j][l]}) * $signed(w[i][j][k][l]);
						end
					end
				end
			end
			return res;
		endfunction : check_output;

		output_vector_t  GOLDEN_OUTPUT = check_output(ACTIVATIONS, WEIGHTS);
		initial begin
			outputs.rdy = 0;
			@(posedge clk iff ap_rst_n);

			for(int unsigned  nf = 0; nf < NF; nf++) begin
				while($urandom()%13 == 0) @(posedge clk);
				outputs.rdy <= 1;
				@(posedge clk iff outputs.vld);
				outputs.rdy <= 0;

				// Compare produced outputs against golden outputs
				foreach(outputs.dat[i]) begin
					assert ($signed(outputs.dat[i]) == $signed(GOLDEN_OUTPUT[nf][i])) begin
						$display(">>> [t=%0t] Test succeeded (nf=%0d)! Computed / GOLDEN = %0d / %0d", $time, nf, $signed(outputs.dat[i]), $signed(GOLDEN_OUTPUT[nf][i]));
					end
					else begin
						$error(">>> [t=%0t] TEST failed (nf=%0d)! Computed / GOLDEN = %0d / %0d", $time, nf, $signed(outputs.dat[i]), $signed(GOLDEN_OUTPUT[nf][i]));
						$stop;
					end
				end
			end

			done[ver][narrow] = 1;
		end

		// Instantiate DUT
		mvu_vvu_axi #(
			.IS_MVU(IS_MVU),
			.COMPUTE_CORE(ver == 1? "mvu_4sx4u_dsp48e1" : "mvu_4sx4u_dsp48e2"),
			.MW(MW),
			.MH(MH),
			.PE(PE),
			.SIMD(SIMD),
			.ACTIVATION_WIDTH(ACTIVATION_WIDTH),
			.WEIGHT_WIDTH(WEIGHT_WIDTH),
			.ACCU_WIDTH(ACCU_WIDTH),
			.NARROW_WEIGHTS(narrow),
			.SIGNED_ACTIVATIONS(SIGNED_ACTIVATIONS),
			.SEGMENTLEN(SEGMENTLEN),
			.FORCE_BEHAVIORAL(FORCE_BEHAVIORAL),
			.M_REG_LUT(M_REG_LUT)
		)
		dut (
			.ap_clk, .ap_rst_n, .s_axis_weights_tdata({ {WEIGHT_WIDTH_BA_DELTA{1'b0}}, weights.dat }), .s_axis_weights_tvalid(weights.vld),
			.s_axis_weights_tready(weights.rdy), .s_axis_input_tdata({ {ACTIVATION_WIDTH_BA_DELTA{1'b0}}, activations.dat }), .s_axis_input_tvalid(activations.vld),
			.s_axis_input_tready(activations.rdy), .m_axis_output_tdata(outputs.dat), .m_axis_output_tvalid(outputs.vld),
			.m_axis_output_tready(outputs.rdy)
		);

		end : genNarrowWide
	end : genVersion

endmodule : mvu_axi_tb
