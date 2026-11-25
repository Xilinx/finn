/****************************************************************************
 * Copyright (C) 2026, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Testbench for pwpolyf: FP32 piecewise polynomial activation.
 * @author	Shane Fleming <shane.fleming@amd.com>
 *
 * @description
 *	Tests all four activation functions (gelu, silu, sigmoid, tanh) in
 *	parallel using random FP32 stimulus with online shortreal-based
 *	checking against a reference function.
 ***************************************************************************/

module pwpolyf_tb;

	localparam int unsigned  TEST_COUNT = 4;
	localparam string  FUNCS[TEST_COUNT] = '{"gelu", "silu", "sigmoid", "tanh"};
	localparam int unsigned  RUNS = 4096;

	// Global Control
	logic  clk = 0;
	always #5ns clk = !clk;
	logic  rst = 1;
	initial begin
		repeat(12) @(posedge clk);
		rst <= 0;
	end

	bit [TEST_COUNT-1:0]  done = '0;
	always_comb begin
		if(&done)  $finish;
	end

	for(genvar  t = 0; t < TEST_COUNT; t++) begin : genTests
		localparam string  FUNC = FUNCS[t];

		// DUT wired for PE=1
		logic [31:0]  xdat;
		logic  xvld;
		uwire  xrdy;
		uwire [31:0]  ydat;
		uwire  yvld;
		logic  yrdy;

		pwpolyf #(.PE(1), .FUNC(FUNC)) dut (
			.clk, .rst,
			.xdat, .xvld, .xrdy,
			.ydat, .yvld, .yrdy
		);
		shortreal  y;
		assign  y = $bitstoshortreal(ydat);

		// Reference function -- compute in real, cast to shortreal
		function automatic shortreal ref_func(input shortreal x);
			automatic real  xr = real'(x);
			automatic real  yr;
			if(xr >= 8.0)
				return (FUNC == "gelu" || FUNC == "silu")? x : shortreal'(1.0);
			if(xr <= -8.0)
				return (FUNC == "tanh")? shortreal'(-1.0) : shortreal'(0.0);
			if(FUNC == "gelu") begin
				automatic real  t = $tanh($sqrt(2.0/3.14159265358979) * (xr + 0.044715*xr*xr*xr));
				yr = 0.5 * xr * (1.0 + t);
			end
			else if(FUNC == "silu")  yr = xr / (1.0 + $exp(-xr));
			else if(FUNC == "sigmoid")  yr = 1.0 / (1.0 + $exp(-xr));
			else  yr = $tanh(xr);
			return shortreal'(yr);
		endfunction

		// Online checking state
		shortreal  ExpQ[$];

		// Stimulus driver
		initial begin
			xdat = '0;
			xvld = 0;
			@(posedge clk iff !rst);

			repeat(RUNS) begin
				automatic logic [31:0]  vbits;

				// Cover range [-8, 8) across all 5 octaves (exp 125..129)
				vbits = 32'h40000000 + ($urandom() % 32'h01800000);  // [2.0, 6.0) range
				if($urandom() % 2)  vbits[31] = 1;  // random sign
				if($urandom() % 4 == 0) vbits = 32'h3F800000;  // 1.0
				if($urandom() % 8 == 0) vbits = 32'h00000000;  // 0.0
				if($urandom() % 8 == 0) vbits = 32'h40E00000 | ($urandom() % 32'h00100000);  // [7.0, 7.5)

				while($urandom() % 17 == 0) @(posedge clk);

				xdat <= vbits;
				xvld <= 1;

				@(posedge clk iff xrdy);
				ExpQ.push_back(ref_func($bitstoshortreal(vbits)));

				xvld <= 0;
			end
		end

		always_ff @(posedge clk iff yvld && yrdy) begin
			automatic shortreal  exp, err;
			assert(ExpQ.size) else begin
				$error("[%s] Spurious output.", FUNC);
				$stop;
			end
			exp = ExpQ.pop_front();
			err = y - exp;
			err *= err;
			assert((err < 1e-3) || ($shortrealtobits(y) == $shortrealtobits(exp))) else begin
				$error("[%s] Output mismatch: %f/%08x instead of %f/%08x",
					FUNC, y, $shortrealtobits(y), exp, $shortrealtobits(exp));
				$stop;
			end
		end

		// Output collector -- drives yrdy backpressure
		initial begin
			yrdy = 0;
			@(posedge clk iff !rst);

			repeat(RUNS) begin
				while($urandom() % 17 == 0) @(posedge clk);
				yrdy <= 1;
				@(posedge clk iff yvld);
				yrdy <= 0;
			end

			// Verify all expected outputs were consumed
			@(posedge clk);
			assert(ExpQ.size() == 0) else begin
				$error("[%s] Missing %0d outputs.", FUNC, ExpQ.size());
				$stop;
			end

			$display("PWPOLYF[%s]: %0d outputs verified online.", FUNC, RUNS);
			done[t] = 1;
		end

	end : genTests

endmodule : pwpolyf_tb
