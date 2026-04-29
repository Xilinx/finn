/****************************************************************************
 * Copyright (C) 2025-2026, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 ***************************************************************************/

module rsqrtf_tb;

	localparam bit  FORCE_BEHAVIORAL = 0;
	localparam int unsigned  MIN_SUSTAINABLE_INTERVAL =  1;
	localparam int unsigned  MAX_SUSTAINABLE_INTERVAL = 15;
	localparam int unsigned  TEST_COUNT = MAX_SUSTAINABLE_INTERVAL - MIN_SUSTAINABLE_INTERVAL + 1;
	localparam int unsigned  ROUNDS = 137;

	// Global Control
	logic  clk = 0;
	always #5ns clk = !clk;
	logic  rst = 1;
	initial begin
		repeat(12) @(posedge clk);
		rst <= 0;
	end

	bit [MAX_SUSTAINABLE_INTERVAL:MIN_SUSTAINABLE_INTERVAL]  done = '0;
	always_comb begin
		if(&done)  $finish;
	end

	// Reference Compute
	function shortreal q_rsqrt(input shortreal  x);
		automatic shortreal  y = $bitstoshortreal('h5f3759df - ($shortrealtobits(x) >> 1));
		return  y * (1.5 - (0.5 * x * y * y));
	endfunction : q_rsqrt

	for(genvar  t = MIN_SUSTAINABLE_INTERVAL; t <= MAX_SUSTAINABLE_INTERVAL; t++) begin : genTests

		// DUT
		shortreal  fx;
		uwire [31:0]  x = $shortrealtobits(fx);
		logic  xvld;
		uwire [31:0]  r;
		uwire  rvld;
		uwire  xrdy;
		rsqrtf #(
			.SUSTAINABLE_INTERVAL(t),
			.FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)
		) dut (
			.clk, .rst,
			.x, .xvld, .xrdy,
			.r, .rvld
		);
		shortreal  fr;
		assign	fr = $bitstoshortreal(r);

		// Stimulus
		shortreal  Q[$];
		initial begin
			automatic int unsigned  Round2Cycles = 0;
			fx = 'x;
			xvld = 0;
			@(posedge clk iff !rst);

			// Round 1: intermittent feed with occasional stalls
			for(int unsigned  i = 1; i <= ROUNDS; i++) begin
				while($urandom()%23 == 0) @(posedge clk);
				fx <= i;
				xvld <= 1;
				@(posedge clk iff xrdy);
				Q.push_back(fx);
				fx <= 'x;
				xvld <= 0;
			end
			repeat(12) @(posedge clk);

			// Round 2: feed as fast as the DUT accepts input
			xvld <= 1;
			for(int unsigned  i = 0; i < ROUNDS;) begin
				fx <= ROUNDS + i;
				@(posedge clk);
				Round2Cycles++;
				if(xrdy) begin
					Q.push_back(fx);
					i++;
				end
			end
			xvld <= 0;
			fx <= 'x;

			$display("Test #%0d: Round-2 cycles/input = %0d/%0d = %0.2f", t, Round2Cycles, ROUNDS, real'(Round2Cycles)/ROUNDS);

			repeat(32) @(posedge clk);
			assert(Q.size() == 0) else begin
				$error("Test #%0d: Missing %0d outputs.", t, Q.size());
				$stop;
			end

			done[t] = 1;
		end

		// Checker
		int unsigned  Checks = 0;
		always_ff @(posedge clk iff rvld) begin
			automatic shortreal  x, exp, err;
			assert(Q.size()) else begin
				$error("Test #%0d: Spurious output.", t);
				$stop;
			end
			x = Q.pop_front();
			exp = q_rsqrt(x);

			err = fr - exp;
			err *= err;
			assert(err < 1e-8) else begin
				$error("Test #%0d: Output mismatch for %f: %f instead of %f", t, x, fr, exp);
				$stop;
			end
			Checks <= Checks + 1;
		end

		final begin
			assert(Checks == 2*ROUNDS)  $display("Test #%0d: Successfully performed %0d checks.", t, Checks);
			else  $error("Test #%0d: Unexpected number of checks: %0d instead of %0d.", t, Checks, 2*ROUNDS);
		end
	end : genTests

endmodule : rsqrtf_tb
