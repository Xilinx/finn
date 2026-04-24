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
	} cfg_t;
	localparam int unsigned  TESTS = 9;
	localparam cfg_t  TEST_CFG[TESTS] = '{
		'{  4,  4 },  // NN=1
		'{ 10,  5 },  // NN=2
		'{ 18,  6 },  // NN=3
		'{ 42,  7 },  // NN=6
		'{ 64,  8 },  // NN=8
		'{ 81,  9 },  // NN=9
		'{100, 10 },  // NN=10
		'{ 44,  4 },  // NN=11
		'{ 60,  5 }   // NN=12
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
		typedef shortreal  vec_t[N];

		// DUT
		logic [SIMD-1:0][31:0]  xdat;
		logic  xvld;
		uwire  xrdy;
		uwire [SIMD-1:0][31:0]  ydat;
		uwire  yvld;
		logic  yrdy;
		layernorm #(.N(N), .SIMD(SIMD), .FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)) dut (
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
				end
			end
			repeat(5) @(posedge clk);

			$display("[%0d] Test completed.", test);
			done[test] <= 1;
		end

	end : genTests

endmodule : layernorm_tb
