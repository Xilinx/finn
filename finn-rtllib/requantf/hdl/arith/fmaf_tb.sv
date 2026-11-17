/****************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 *
 * @brief	Self-checking testbench for fmaf.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 ***************************************************************************/

module fmaf_tb;
`default_nettype none

	localparam bit  FORCE_BEHAVIORAL = 1;

	typedef struct {
		string  op;
	} cfg_t;
	localparam int unsigned  TESTS = 3;
	localparam cfg_t  CFGS[TESTS] = '{
		'{ "ADD" },
		'{ "SBR" },
		'{ "SUB" }
	};

	// Global Control
	logic  clk = 0;
	always #5ns clk = !clk;
	logic  rst = 1;
	initial begin
		repeat(3) @(posedge clk);
		rst <= 0;
	end

	// Test Instantiations
	bit [TESTS-1:0]  done = '0;
	always_comb begin
		if(&done)  $finish;
	end

	for(genvar  test = 0; test < TESTS; test++) begin : genTests
		localparam cfg_t  CFG = CFGS[test];

		function shortreal compute_ref(input shortreal  a, b, c);
			unique case(CFG.op)
			"ADD":  return  a * b + c;
			"SBR":  return  a * b - c;
			"SUB":  return  c - a * b;
			endcase
		endfunction : compute_ref

		// DUT wiring
		shortreal  fa, fb, fc, fr;
		logic  avld;
		uwire  rvld;
		if(1) begin : blkDUT
			uwire [31:0]  aa = $shortrealtobits(fa);
			uwire [31:0]  bb = $shortrealtobits(fb);
			uwire [31:0]  cc = $shortrealtobits(fc);
			uwire [31:0]  rr;
			fmaf #(.OP(CFG.op), .FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)) dut (
				.clk, .rst,
				.a(aa), .b(bb), .c(cc), .ivld(avld),
				.r(rr), .ovld(rvld)
			);
			assign	fr = $bitstoshortreal(rr);
		end : blkDUT

		// Reference queue
		shortreal  Q[$];

		// Stimulus
		initial begin
			avld = 0;
			fa = 0.0;
			fb = 0.0;
			fc = 0.0;
			@(posedge clk iff !rst);
			repeat(2) @(posedge clk);

			//--- Directed tests ------------------------------------------------
			// Identity: 1*x + 0
			fa <= 7.5;  fb <= 1.0;  fc <= 0.0;  avld <= 1;
			@(posedge clk);
			Q.push_back(compute_ref(fa, fb, fc));

			// Pure add: 0*x + c  (effectively c through the pipeline)
			fa <= 0.0;  fb <= 0.0;  fc <= 42.0;  avld <= 1;
			@(posedge clk);
			Q.push_back(compute_ref(fa, fb, fc));

			// Negative operands
			fa <= -3.0;  fb <= 5.0;  fc <= 10.0;  avld <= 1;
			@(posedge clk);
			Q.push_back(compute_ref(fa, fb, fc));

			// Large values
			fa <= 1000.0;  fb <= 1000.0;  fc <= -500000.0;  avld <= 1;
			@(posedge clk);
			Q.push_back(compute_ref(fa, fb, fc));

			// Small fractional
			fa <= 0.125;  fb <= 0.25;  fc <= 0.5;  avld <= 1;
			@(posedge clk);
			Q.push_back(compute_ref(fa, fb, fc));

			avld <= 0;
			fa <= 0.0;  fb <= 0.0;  fc <= 0.0;
			@(posedge clk);

			//--- Random tests --------------------------------------------------
			repeat(1739) begin
				while($urandom()%7 == 0) @(posedge clk);

				fa <= $urandom()%10000 - 5000.0;
				fb <= $urandom()%10000 - 5000.0;
				fc <= $urandom()%10000 - 5000.0;
				avld <= 1;
				@(posedge clk);
				Q.push_back(compute_ref(fa, fb, fc));
				avld <= 0;
			end

			repeat(8) @(posedge clk);
			assert(Q.size() == 0) else begin
				$error("Test #%0d (%s): Missing %0d outputs.", test, CFG.op, Q.size());
				$stop;
			end
			$display("Test #%0d (%s) completed.", test, CFG.op);
			done[test] = 1;
		end

		// Checker: allow small relative error since the reference computes
		// a*b and +c as two separately-rounded shortreal operations, while
		// the DUT pipeline may round the intermediate product differently.
		always_ff @(posedge clk iff rvld) begin
			automatic shortreal  exp, diff, mag;
			assert(Q.size) else begin
				$error("Test #%0d (%s): Spurious output.", test, CFG.op);
				$stop;
			end
			exp = Q.pop_front();
			diff = fr - exp;
			if(diff < 0)  diff = -diff;
			mag = exp < 0? -exp : exp;
			// Accept if absolute error < 1 or relative error < 2^-20
			assert(diff < 1.0 || diff < mag * 9.6e-7) else begin
				$error(
					"Test #%0d (%s): Output mismatch: %f/%08x instead of %f/%08x",
					test, CFG.op, fr, $shortrealtobits(fr), exp, $shortrealtobits(exp)
				);
				$stop;
			end
		end

	end : genTests

`default_nettype wire
endmodule : fmaf_tb
