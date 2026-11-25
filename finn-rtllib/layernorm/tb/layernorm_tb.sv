/****************************************************************************
 * Copyright (C) 2025-2026, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 ***************************************************************************/

 module layernorm_tb;

	localparam int unsigned  ROUNDS = 19;
	localparam bit  FORCE_BEHAVIORAL = 1;

	typedef struct {
		int unsigned  n;
		int unsigned  simd;
		int unsigned  refinements;
	} cfg_t;
	localparam int unsigned  TESTS = 18;
	localparam cfg_t  TEST_CFG[TESTS] = '{
		'{  4,  4, 1 },  // NN=1
		'{ 10,  5, 1 },  // NN=2
		'{ 18,  6, 1 },  // NN=3
		'{ 42,  7, 1 },  // NN=6
		'{ 64,  8, 1 },  // NN=8
		'{ 81,  9, 1 },  // NN=9
		'{100, 10, 1 },  // NN=10
		'{ 44,  4, 1 },  // NN=11
		'{ 60,  5, 1 },  // NN=12
		'{  4,  4, 2 },  // NN=1:  6 DSPs unrolled
		'{ 10,  5, 2 },  // NN=2:  3 rings of 2 passes
		'{ 18,  6, 2 },  // NN=3:  2 rings of 3 passes
		'{ 40,  8, 2 },  // NN=5
		'{ 42,  7, 2 },  // NN=6:  single ring of 6 passes
		'{ 80,  4, 2 },  // NN=20
		'{ 84,  4, 2 },  // NN=21: overlapped
		'{ 92,  4, 2 },  // NN=23
		'{ 96,  4, 2 }   // NN=24: exclusive
	};

	//-----------------------------------------------------------------------
	// Global Control
	logic  clk = 0;
	always #5ns clk = !clk;
	logic  rst = 1;
	initial begin
		repeat(12) @(posedge clk);
		rst <= 0;
	end

	//-----------------------------------------------------------------------
	// Test Instantiations
	bit [TESTS-1:0]  done = '0;
	always_comb begin
		if(&done) $finish();
	end
	for(genvar  test = 0; test < TESTS; test++) begin : genTests
		localparam int unsigned  N    = TEST_CFG[test].n;
		localparam int unsigned  SIMD = TEST_CFG[test].simd;
		localparam int unsigned  REFINEMENTS = TEST_CFG[test].refinements;
		typedef shortreal  vec_t[N];

		// DUT
		logic [SIMD-1:0][31:0]  xdat;
		logic  xvld;
		uwire  xrdy;
		uwire [SIMD-1:0][31:0]  ydat;
		uwire  yvld;
		logic  yrdy;
		layernorm #(
			.N(N), .SIMD(SIMD),
			.NUM_RSQRT_REFINEMENTS(REFINEMENTS),
			.FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)
		) dut (
			.clk, .rst,
			.xdat, .xvld, .xrdy,
			.ydat, .yvld, .yrdy
		);

		// Stimulus
		vec_t  X[ROUNDS];
		initial begin
			xdat = 'x;
			xvld = 0;
			@(posedge clk iff !rst);

			for(int unsigned  r = 0; r < ROUNDS; r++) begin
				static shortreal  b;
				static shortreal  s;
				b = $urandom()%129 - 53.0;
				s = ($urandom()%29 + 1) / 1.3;
				foreach(X[r][i])  X[r][i] = s*($urandom()%1537 - 757.0) + b;

				for(int unsigned  i = 0; i < N; i += SIMD) begin
					while($urandom()%23 == 0) @(posedge clk);
					xvld <= 1;
					for(int unsigned  j = 0; j < SIMD; j++) begin
						xdat[j] <= $shortrealtobits(X[r][i+j]);
					end
					@(posedge clk iff xrdy);
					xdat <= 'x;
					xvld <= 0;
				end
			end

			$display("[%0d] Input feed done.", test);
		end

		// Output Checker
		vec_t  y, exp;
		initial begin
			yrdy = 0;
			@(posedge clk iff !rst);
			repeat(187) @(posedge clk);

			for(int unsigned  r = 0; r < ROUNDS; r++) begin
				static shortreal  m, s;

				for(int unsigned  i = 0; i < N; i += SIMD) begin
					while($urandom()%5 == 0) @(posedge clk);
					yrdy <= 1;
					@(posedge clk iff yvld);
					foreach(ydat[j])  y[i+j] = $bitstoshortreal(ydat[j]);
					yrdy <= 0;
				end

				m = 0.0;
				foreach(X[r][i])  m += X[r][i];
				m /= N;

				s = 0.0;
				foreach(exp[i]) begin
					exp[i] = X[r][i] - m;
					s += exp[i] * exp[i];
				end
				s = 1/$sqrt(s/N);

				foreach(exp[i])  exp[i] *= s;

				foreach(y[i]) begin
					static shortreal  err;
					err = (y[i]-exp[i])/exp[i];
					err *= err;
					assert(err < 1e-5) else begin
						$error("[%0d] Output mismatch: %7.4f instead of %7.4f", test, y[i], exp[i]);
						$stop;
					end

					// Error relative to unit output scale with two refinements
					if(REFINEMENTS > 1) begin
						err = (y[i]-exp[i]) / ((exp[i] < -1.0)? -exp[i] : (exp[i] > 1.0)? exp[i] : 1.0);
						assert((-1e-5 < err) && (err < 1e-5)) else begin
							$error("[%0d] Output inaccurate: %e instead of %e", test, y[i], exp[i]);
							$stop;
						end
					end
				end
			end
			repeat(5) @(posedge clk);

			$display("[%0d] Test completed.", test);
			done[test] <= 1;
		end

	end : genTests

endmodule : layernorm_tb
