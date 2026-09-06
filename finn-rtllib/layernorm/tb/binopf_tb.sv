/****************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 ***************************************************************************/

module binopf_tb;

	localparam bit  FORCE_BEHAVIORAL = 1;
	typedef struct {
		string     op;
		shortreal  scale;
		bit        delay;
	} cfg_t;
	localparam int unsigned  TESTS = 8;
	localparam cfg_t  CFGS[TESTS] = '{
		'{ "ADD", 1.0, 0 },
		'{ "ADD", 1.0, 1 },
		'{ "ADD", 1.3, 0 },
		'{ "SUB", 1.0, 0 },
		'{ "SUB", 0.4, 0 },
		'{ "SBR", 1.0, 0 },
		'{ "SBR", 3.7, 0 },
		'{ "MUL", 1.0, 0 }
	};

	// Global Control
	logic  clk = 0;
	always #5ns clk = !clk;
	logic  rst = 1;
	initial begin
		repeat(12) @(posedge clk);
		rst <= 0;
	end

	// Test Instantiations
	bit [TESTS-1:0]  done = '0;
	always_comb begin
		if(&done) $finish;
	end
	for(genvar  test = 0; test < TESTS; test++) begin : genTests
		localparam cfg_t  CFG = CFGS[test];
		localparam shortreal  SCALE = CFG.scale;

		function shortreal compute_ref(input shortreal  a, input shortreal  b);
			unique case(CFG.op)
			"ADD":  return  a + SCALE*b;
			"SUB":  return  a - SCALE*b;
			"SBR":  return  SCALE*b - a;
			"MUL":  return  a * b;
			endcase
		endfunction : compute_ref

		// DUT
		logic  avld;
		shortreal  a;
		logic  bload;
		shortreal  b;
		uwire  rvld;
		shortreal  r;
		if(1) begin : blkDUT
			uwire [31:0]  aa = $shortrealtobits(a);
			uwire [31:0]  bb = $shortrealtobits(b);
			uwire [31:0]  rr;
			binopf #(
				.OP(CFG.op), .B_SCALE(SCALE),
				.A_MATCH_OP_DELAY(CFG.delay),
				.FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)
			) dut (
				.clk, .rst,
				.avld, .a(aa), .bload, .b(bb),
				.rvld, .r(rr)
			);
			assign	r = $bitstoshortreal(rr);
		end : blkDUT

		// Stimulus
		shortreal  B0;
		shortreal  Q[$];
		initial begin

			avld = 0;
			a = 'x;
			bload = 0;
			b = 'x;
			@(posedge clk iff !rst);

			// Fork off Background Update of `b` Input
			fork
				forever begin
					automatic shortreal  val = $urandom()%10000 - 5000.0;
					bload <= 1;
					b <= val;
					B0 <= val;
					@(posedge clk);
					bload <= 0;
					b <= 'x;
					while($urandom()%37 != 0) @(posedge clk);
				end
			join_none

			// Run a Series of `a` Values
			repeat(1739) begin
				while($urandom()%17 == 0) @(posedge clk);

				avld <= 1;
				a <= $urandom()%10000 - 5000.0;
				@(posedge clk);
				fork begin
					automatic shortreal  a0 = a;
					if(CFG.delay)  repeat(2) @(posedge clk);
					Q.push_back(compute_ref(a0, B0));
				end join_none
				avld <= 0;
				a <= 'x;
			end

			repeat(7) @(posedge clk);
			assert(Q.size() == 0) else begin
				$error("Test #%0d: Missing output.", test);
				$stop;
			end
			$display("Test #%0d completed.", test);
			done[test] = 1;
		end

		// Checker
		always_ff @(posedge clk iff rvld) begin
			automatic shortreal  exp, err;
			assert(Q.size) else begin
				$error("Test #%0d: Spurious output.", test);
				$stop;
			end
			exp = Q.pop_front();
			err = r - exp;
			err *= err;
			assert((err < 1e-5) || ($shortrealtobits(r) == $shortrealtobits(exp))) else begin
				$error(
					"Test #%0d: Output mismatch: %f/%08x instead of %f/%08x",
					test, r, $shortrealtobits(r), exp, $shortrealtobits(exp)
				);
				$stop;
			end
		end

	end : genTests

endmodule : binopf_tb
