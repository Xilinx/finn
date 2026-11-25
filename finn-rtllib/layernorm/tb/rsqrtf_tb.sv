/****************************************************************************
 * Copyright (C) 2025-2026, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 *
 * @description
 *	Exercises rsqrtf for both ITERATIONS across all SUSTAINABLE_INTERVALs
 *	up to beyond the threshold to its exclusive single-DSP structure:
 *	 - Round 1: intermittent feed with occasional stalls,
 *	 - Round 2: feed as fast as the DUT accepts input,
 *	 - Round 3: feed paced at the SUSTAINABLE_INTERVAL with random extra
 *	            gaps, which must never meet a revoked readiness.
 *	Outputs are checked bit-exact against an fp32 model of the computation.
 *	The latency of paced inputs from their acceptance to output is checked
 *	against 12 cycles per iteration plus, for structures with an input skid,
 *	2 cycles of skid and at most 3 (7 for the paired DSPs) cycles of slot
 *	arbitration, which are the worst cases over all input sequences.
 ***************************************************************************/

module rsqrtf_tb_case #(
	int unsigned  SUSTAINABLE_INTERVAL,
	int unsigned  ITERATIONS,
	bit  FORCE_BEHAVIORAL
)(
	input	logic  clk,
	input	logic  rst,
	output	logic  done
);

	localparam int unsigned  ROUNDS = 137;
	localparam int unsigned  PASSES = 3*ITERATIONS;
	localparam bit  SKID =
		(SUSTAINABLE_INTERVAL > 1) && (SUSTAINABLE_INTERVAL < 4*PASSES - 3);
	localparam int unsigned  MIN_LATENCY = 4*PASSES;
	localparam bit  PAIRED = (ITERATIONS == 1) && (SUSTAINABLE_INTERVAL == 2);
	localparam int unsigned  MAX_LATENCY = MIN_LATENCY + (!SKID? 0 : PAIRED? 9 : 5);

	// Reference Compute in fp32 with explicit rounding of every operation
	function logic [31:0] q_rsqrt(input logic [31:0]  x);
		automatic logic [31:0]  x2 = { 1'b0, x[30:23]-8'd1, x[22:0] };
		automatic logic [31:0]  y = 'h5f3759df - (x >> 1);
		for(int unsigned  i = 0; i < ITERATIONS; i++) begin
			automatic logic [31:0]  h = $shortrealtobits($bitstoshortreal(x2) * $bitstoshortreal(y));
			automatic logic [31:0]  t = $shortrealtobits($bitstoshortreal(h) * $bitstoshortreal(y));
			automatic logic [31:0]  u = $shortrealtobits(1.5 - $bitstoshortreal(t));
			y = $shortrealtobits($bitstoshortreal(u) * $bitstoshortreal(y));
		end
		return  y;
	endfunction : q_rsqrt

	// Positive normal fp32 of random magnitude
	function shortreal rand_x();
		return  $bitstoshortreal({ 1'b0, 8'(20 + $urandom()%210), 23'($urandom()) });
	endfunction : rand_x

	// DUT
	shortreal  fx;
	uwire [31:0]  x = $shortrealtobits(fx);
	logic  xvld;
	uwire [31:0]  r;
	uwire  rvld;
	uwire  xrdy;
	rsqrtf #(
		.SUSTAINABLE_INTERVAL(SUSTAINABLE_INTERVAL),
		.ITERATIONS(ITERATIONS),
		.FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)
	) dut (
		.clk, .rst,
		.x, .xvld, .xrdy,
		.r, .rvld
	);

	// Stimulus
	typedef struct {
		shortreal     x;
		int unsigned  t;	// acceptance cycle
		bit           p;	// paced at SUSTAINABLE_INTERVAL
	} job_t;
	job_t  Q[$];
	int unsigned  Cycle = 0;
	always_ff @(posedge clk)  Cycle <= Cycle + 1;

	initial begin
		automatic int unsigned  Round2Cycles = 0;
		done = 0;
		fx = 'x;
		xvld = 0;
		@(posedge clk iff !rst);

		// Round 1: intermittent feed with occasional stalls
		for(int unsigned  i = 1; i <= ROUNDS; i++) begin
			while($urandom()%23 == 0) @(posedge clk);
			fx <= (i < 32)? i : rand_x();
			xvld <= 1;
			@(posedge clk iff xrdy);
			Q.push_back('{ x: fx, t: Cycle, p: 0 });
			fx <= 'x;
			xvld <= 0;
		end
		repeat(4*PASSES) @(posedge clk);

		// Round 2: feed as fast as the DUT accepts input
		xvld <= 1;
		for(int unsigned  i = 0; i < ROUNDS;) begin
			fx <= rand_x();
			@(posedge clk);
			Round2Cycles++;
			if(xrdy) begin
				Q.push_back('{ x: fx, t: Cycle, p: 0 });
				i++;
			end
		end
		xvld <= 0;
		fx <= 'x;
		$display(
			"Test II=%0d/ITERATIONS=%0d: Round-2 cycles/input = %0d/%0d = %0.2f",
			SUSTAINABLE_INTERVAL, ITERATIONS, Round2Cycles, ROUNDS, real'(Round2Cycles)/ROUNDS
		);
		@(posedge clk iff Q.size() == 0);
		repeat(4) @(posedge clk);

		// Round 3: paced feed at sustainable interval with random extra gaps
		for(int unsigned  i = 0; i < 4*ROUNDS; i++) begin
			fx <= rand_x();
			xvld <= 1;
			@(posedge clk);
			assert(xrdy) else begin
				$error("Test II=%0d/ITERATIONS=%0d: Revoked readiness.", SUSTAINABLE_INTERVAL, ITERATIONS);
				$stop;
			end
			Q.push_back('{ x: fx, t: Cycle, p: 1 });
			fx <= 'x;
			xvld <= 0;
			repeat(SUSTAINABLE_INTERVAL-1 + ($urandom()%3 == 0? $urandom()%3 : 0)) @(posedge clk);
		end

		repeat(4*PASSES + 8) @(posedge clk);
		assert(Q.size() == 0) else begin
			$error("Test II=%0d/ITERATIONS=%0d: Missing %0d outputs.", SUSTAINABLE_INTERVAL, ITERATIONS, Q.size());
			$stop;
		end

		done = 1;
	end

	// Checker
	int unsigned  Checks = 0;
	int unsigned  MinLatency = '1;
	int unsigned  MaxLatency = 0;
	always_ff @(posedge clk iff rvld) begin
		automatic job_t  job;
		automatic logic [31:0]  exp;
		automatic int unsigned  lat;
		assert(Q.size()) else begin
			$error("Test II=%0d/ITERATIONS=%0d: Spurious output.", SUSTAINABLE_INTERVAL, ITERATIONS);
			$stop;
		end
		job = Q.pop_front();
		exp = q_rsqrt($shortrealtobits(job.x));
		assert(r == exp) else begin
			$error(
				"Test II=%0d/ITERATIONS=%0d: Output mismatch for %e: %e (%08x) instead of %e (%08x)",
				SUSTAINABLE_INTERVAL, ITERATIONS, job.x, $bitstoshortreal(r), r, $bitstoshortreal(exp), exp
			);
			$stop;
		end
		lat = Cycle - job.t;
		if(job.p) begin
			assert((MIN_LATENCY <= lat) && (lat <= MAX_LATENCY)) else begin
				$error(
					"Test II=%0d/ITERATIONS=%0d: Latency of %0d outside of [%0d, %0d].",
					SUSTAINABLE_INTERVAL, ITERATIONS, lat, MIN_LATENCY, MAX_LATENCY
				);
				$stop;
			end
			if(lat < MinLatency)  MinLatency <= lat;
			if(lat > MaxLatency)  MaxLatency <= lat;
		end
		Checks <= Checks + 1;
	end

	final begin
		assert(Checks == 6*ROUNDS) begin
			$display(
				"Test II=%0d/ITERATIONS=%0d: Successfully performed %0d checks, paced latency %0d..%0d.",
				SUSTAINABLE_INTERVAL, ITERATIONS, Checks, MinLatency, MaxLatency
			);
		end
		else begin
			$error(
				"Test II=%0d/ITERATIONS=%0d: Unexpected number of checks: %0d instead of %0d.",
				SUSTAINABLE_INTERVAL, ITERATIONS, Checks, 6*ROUNDS
			);
		end
	end

endmodule : rsqrtf_tb_case

module rsqrtf_tb #(
	bit  FORCE_BEHAVIORAL = 0
);

	localparam int unsigned  MAX_SUSTAINABLE_INTERVAL[2] = '{ 13, 25 };

	// Global Control
	logic  clk = 0;
	always #5ns clk = !clk;
	logic  rst = 1;
	initial begin
		repeat(12) @(posedge clk);
		rst <= 0;
	end

	uwire [1:2][MAX_SUSTAINABLE_INTERVAL[1]:1]  done;
	for(genvar  it = 1; it <= 2; it++) begin : genIterations
		for(genvar  ii = 1; ii <= MAX_SUSTAINABLE_INTERVAL[1]; ii++) begin : genTests
			if(ii > MAX_SUSTAINABLE_INTERVAL[it-1])  assign  done[it][ii] = 1;
			else begin : genCase
				rsqrtf_tb_case #(
					.SUSTAINABLE_INTERVAL(ii),
					.ITERATIONS(it),
					.FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)
				) test (
					.clk, .rst,
					.done(done[it][ii])
				);
			end : genCase
		end : genTests
	end : genIterations

	always_comb begin
		if(&done)  $finish;
	end

endmodule : rsqrtf_tb
