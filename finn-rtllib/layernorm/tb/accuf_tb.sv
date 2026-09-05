/****************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 ***************************************************************************/

module accuf_tb;

	localparam bit  FORCE_BEHAVIORAL = 1;

	typedef struct {
		shortreal  scale;
		shortreal  bias;
	} cfg_t;
	localparam int unsigned  TESTS = 3;
	localparam cfg_t  TEST_CFG[TESTS] = '{
		cfg_t'{ scale: 1.0, bias: 0.0 },
		cfg_t'{ scale: 0.2, bias: 0.0 },
		cfg_t'{ scale: 0.0, bias: 4.0 }
	};

	// Global Control
	logic  clk = 0;
	always #5ns clk = !clk;
	logic  rst = 1;
	initial begin
		repeat(12) @(posedge clk);
		rst <= 0;
	end

	// Test Instantiation
	bit [TESTS-1:0]  done = '0;
	always_comb begin
		if(&done) $finish;
	end
	for(genvar  test = 0; test < TESTS; test++) begin : genTests
		localparam cfg_t  CFG = TEST_CFG[test];

		// DUT
		uwire [31:0]  a;
		logic  avld;
		logic  alst;
		uwire [31:0]  s;
		uwire  svld;
		accuf #(.SCALE(CFG.scale), .BIAS(CFG.bias), .FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)) dut (
			.clk, .rst,
			.a, .avld, .alst,
			.s, .svld
		);

		// Stimulus
		shortreal  Q[$];
		shortreal  fa, fs;
		assign	a = $shortrealtobits(fa);
		assign	fs = $bitstoshortreal(s);
		initial begin
			automatic shortreal  s = CFG.bias;

			fa = 'x;
			avld = 0;
			alst = 'x;
			@(posedge clk iff !rst);

			for(int unsigned  i = 0; i < 417; i++) begin
				while($urandom()%23 == 0) @(posedge clk);
				avld <= 1;
				fa <= i;
				alst <= (i == 416) || ($urandom()%11 == 0);
				@(posedge clk);
				avld <= 0;

				s += (CFG.scale == 0.0? fa : CFG.scale) * fa;
				if(alst) begin
					Q.push_back(s);
					s = CFG.bias;
				end
			end

			repeat(5) @(posedge clk);
			assert(Q.size() == 0) else begin
				$error("Test #%0d: Missing output.", test);
				$stop;
			end
			$display("Test #%0d completed.", test);
			done[test] = 1;
		end

		// Checker
		always_ff @(posedge clk iff svld) begin
			automatic shortreal  exp, err;
			assert(Q.size) else begin
				$error("Test #%0d: Spurious output.", test);
				$stop;
			end
			exp = Q.pop_front();
			err = fs - exp;
			err *= err;
			assert(err < 1e-8) else begin
				$error("Test #%0d: Output mismatch: %f instead of %f", test, fs, exp);
				$stop;
			end
		end

	end : genTests

endmodule : accuf_tb
