/******************************************************************************
 * Copyright (C) 2026, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Testbench for requant_axi.
 *****************************************************************************/

module requant_axi_tb;

	localparam int unsigned  ROUNDS = 311;

	typedef struct {
		int unsigned  version;
		int unsigned  k;
		int unsigned  n;
		int unsigned  c;
		shortreal     scales[8];
		shortreal     biases[8];
		bit           throttled_in;
		bit           throttled_out;
	} cfg_t;

	localparam int unsigned  TEST_CNT = 7;
	localparam cfg_t  TESTS[TEST_CNT] = '{
		'{ version: 1, k: 12, n: 8, c: 5,
		   scales: '{ 0.0625, 0.1250, 0.2500, 0.5000, 0.7500, 0.0, 0.0, 0.0 },
		   biases: '{ 12.0, 8.0, 4.0, 0.0, -2.0, 0.0, 0.0, 0.0 },
		   throttled_in: 1, throttled_out: 1 },
		'{ version: 2, k: 16, n: 6, c: 5,
		   scales: '{ 0.03125, 0.0625, 0.1250, 0.2500, 0.5000, 0.0, 0.0, 0.0 },
		   biases: '{ 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0 },
		   throttled_in: 1, throttled_out: 1 },
		'{ version: 3, k: 10, n: 5, c: 5,
		   scales: '{ 1.0000, 0.7500, 0.5000, 0.2500, 0.1250, 0.0, 0.0, 0.0 },
		   biases: '{ -4.0, -2.0, 0.0, 2.0, 4.0, 0.0, 0.0, 0.0 },
		   throttled_in: 0, throttled_out: 1 },
		'{ version: 1, k: 9, n: 5, c: 5,
		   scales: '{ 1.2500, 1.0000, 0.7500, 0.5000, 0.2500, 0.0, 0.0, 0.0 },
		   biases: '{ -1.0, -0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0 },
		   throttled_in: 1, throttled_out: 0 },
		'{ version: 2, k: 14, n: 7, c: 5,
		   scales: '{ 0.2000, 0.4000, 0.6000, 0.8000, 1.0000, 0.0, 0.0, 0.0 },
		   biases: '{ -3.0, -1.5, 0.0, 1.5, 3.0, 0.0, 0.0, 0.0 },
		   throttled_in: 1, throttled_out: 0 },
		'{ version: 3, k: 11, n: 6, c: 5,
		   scales: '{ 0.015625, 0.03125, 0.0625, 0.1250, 0.2500, 0.0, 0.0, 0.0 },
		   biases: '{ 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0 },
		   throttled_in: 0, throttled_out: 0 },
		'{ version: 1, k: 8, n: 4, c: 5,
		   scales: '{ 0.5000, 0.3750, 0.2500, 0.1250, 0.0625, 0.0, 0.0, 0.0 },
		   biases: '{ 0.0, 0.5, 1.0, 1.5, 2.0, 0.0, 0.0, 0.0 },
		   throttled_in: 1, throttled_out: 1 }
	};

	//-----------------------------------------------------------------------
	// Global Control
	logic  clk = 0;
	always #5ns clk = !clk;
	logic  rst = 1;
	initial begin
		repeat(8) @(posedge clk);
		rst <= 0;
	end

	//-----------------------------------------------------------------------
	// Parallel test instances
	bit [TEST_CNT-1:0]  done = '0;
	always_comb begin
		if(&done)  $finish;
	end

	for(genvar  t = 0; t < TEST_CNT; t++) begin : genTests
		localparam cfg_t  CFG = TESTS[t];
		localparam int unsigned  VERSION = CFG.version;
		localparam int unsigned  K = CFG.k;
		localparam int unsigned  N = CFG.n;
		localparam int unsigned  C = CFG.c;
		localparam int unsigned  PE = 1;
		localparam int unsigned  CF = C/PE;
		localparam shortreal  SCALES[PE][CF] = '{ '{ CFG.scales[0], CFG.scales[1], CFG.scales[2], CFG.scales[3], CFG.scales[4] } };
		localparam shortreal  BIASES[PE][CF] = '{ '{ CFG.biases[0], CFG.biases[1], CFG.biases[2], CFG.biases[3], CFG.biases[4] } };
		localparam bit  THROTTLED_IN  = CFG.throttled_in;
		localparam bit  THROTTLED_OUT = CFG.throttled_out;
		localparam int unsigned  I_WIDTH = (PE*K+7)/8 * 8;
		localparam int unsigned  O_WIDTH = (PE*N+7)/8 * 8;

		// DUT Instantiation
		logic  irdy;
		logic  ivld;
		logic signed [I_WIDTH-1:0]  idat;
		logic  ordy;
		logic  ovld;
		logic        [O_WIDTH-1:0]  odat;
		requant_axi #(
			.VERSION(VERSION),
			.K(K), .N(N), .C(C), .PE(PE),
			.SCALES(SCALES), .BIASES(BIASES)
		) dut (
			.ap_clk(clk), .ap_rst_n(!rst),
			.s_axis_tready(irdy), .s_axis_tvalid(ivld), .s_axis_tdata(idat),
			.m_axis_tready(ordy), .m_axis_tvalid(ovld), .m_axis_tdata(odat)
		);

		// Input and Reference Feed
		int unsigned  RefQ[$];
		initial begin
			automatic int unsigned  CntCF = 0;
			idat = 'x;
			ivld = 0;
			ordy = 0;
			$display(
				"[%0d] requant_axi test: VERSION=%0d K=%0d N=%0d C=%0d throttle=%0b/%0b",
				t, VERSION, K, N, C, THROTTLED_IN, THROTTLED_OUT
			);

			@(posedge clk iff !rst);
			repeat(ROUNDS) begin
				automatic logic signed [K-1:0]  x;
				automatic int  exp;

				void'(std::randomize(x));
				exp = int'(SCALES[0][CntCF]*x + BIASES[0][CntCF]);
				if(exp < 0)         exp = 0;
				if(exp > 2**N - 1)  exp = 2**N - 1;

				while(THROTTLED_IN && (($urandom()%7) == 0)) @(posedge clk);

				ivld <= 1;
				idat <= x;
				@(posedge clk iff irdy);
				ivld <=  0;
				idat <= 'x;

				RefQ.push_back(exp);
				CntCF = (CntCF + 1) % CF;
			end
		end

		// Output Checker
		initial begin
			ordy = 0;
			@(posedge clk iff !rst);

			repeat(ROUNDS) begin
				automatic int unsigned  exp;

				while(THROTTLED_OUT && (($urandom()%7) == 0)) @(posedge clk);
				ordy <= 1;
				@(posedge clk iff ovld);
				ordy <= 0;

				assert(!$isunknown(odat[N-1:0])) else begin
					$error("[%0d] Unknown output while ovld.", t);
					$stop;
				end
				assert(RefQ.size() > 0) else begin
					$error("[%0d] Spurious output.", t);
					$stop;
				end

				exp = RefQ.pop_front();
				assert(odat[N-1:0] == exp) else begin
					$error("[%0d] Output mismatch: %0d instead of %0d.", t, odat[N-1:0], exp);
					$stop;
				end
			end

			@(posedge clk);
			$display("[%0d] Completed %0d ops.", t, ROUNDS);
			done[t] <= 1;
		end

		final begin
			assert(RefQ.size() == 0) else begin
				$error("[%0d] Missing %0d outputs.", t, RefQ.size());
				$stop;
			end
		end

	end : genTests

endmodule : requant_axi_tb
