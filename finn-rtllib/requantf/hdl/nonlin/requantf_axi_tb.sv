/****************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 *
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 * @brief	Testbench for requantf_axi.
 ***************************************************************************/

module requantf_axi_tb;
`default_nettype none

	localparam int unsigned  ROUNDS = 311;
	localparam bit  FORCE_BEHAVIORAL = 1;

	typedef struct {
		int unsigned  n;
		int unsigned  c;
		shortreal     scales[5];
		shortreal     biases[5];
		bit           throttled_in;
		bit           throttled_out;
		bit           signed_out;
	} cfg_t;

	localparam int unsigned  TEST_CNT = 7;
	localparam cfg_t  TESTS[TEST_CNT] = '{
		// Unsigned tests
		'{ n: 8, c: 5,
		   scales: '{ 0.0625, 0.125, 0.25, 0.5, 0.75 },
		   biases: '{ 12.0, 8.0, 4.0, 0.0, -2.0 },
		   throttled_in: 1, throttled_out: 1, signed_out: 0 },
		'{ n: 4, c: 5,
		   scales: '{ 0.03125, 0.0625, 0.125, 0.25, 0.5 },
		   biases: '{ 7.5, 4.0, 2.0, 1.0, 0.0 },
		   throttled_in: 0, throttled_out: 1, signed_out: 0 },
		'{ n: 6, c: 5,
		   scales: '{ 1.5, 0.75, 0.375, 0.1875, 0.09375 },
		   biases: '{ -10.0, -4.0, 0.0, 2.0, 4.0 },
		   throttled_in: 1, throttled_out: 0, signed_out: 0 },
		// Power-of-two scales
		'{ n: 8, c: 5,
		   scales: '{ 0.25, 0.5, 1.0, 2.0, 4.0 },
		   biases: '{ 0.0, 0.0, 0.0, 0.0, 0.0 },
		   throttled_in: 0, throttled_out: 0, signed_out: 0 },
		// Signed tests
		'{ n: 8, c: 5,
		   scales: '{ 0.0625, 0.125, 0.25, 0.5, 0.75 },
		   biases: '{ -14.0, -7.0, 0.0, 3.5, 7.0 },
		   throttled_in: 1, throttled_out: 1, signed_out: 1 },
		'{ n: 5, c: 5,
		   scales: '{ 0.375, 0.1875, 0.09375, 0.046875, 0.5 },
		   biases: '{ -4.0, -2.0, 0.0, 1.5, 3.0 },
		   throttled_in: 0, throttled_out: 0, signed_out: 1 },
		'{ n: 4, c: 5,
		   scales: '{ 0.5, 0.25, 0.125, 0.0625, 1.0 },
		   biases: '{ -4.0, -2.0, 0.0, 1.0, 3.0 },
		   throttled_in: 1, throttled_out: 0, signed_out: 1 }
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
		localparam int unsigned  N = CFG.n;
		localparam int unsigned  C = CFG.c;
		localparam int unsigned  PE = 1;
		localparam int unsigned  CF = C/PE;
		localparam bit  SIGNED_OUT = CFG.signed_out;
		localparam shortreal  SCALES[PE][CF] = '{ CFG.scales };
		localparam shortreal  BIASES[PE][CF] = '{ CFG.biases };
		localparam bit  THROTTLED_IN  = CFG.throttled_in;
		localparam bit  THROTTLED_OUT = CFG.throttled_out;
		localparam int unsigned  I_WIDTH = PE*32;
		localparam int unsigned  O_WIDTH = (PE*N+7)/8 * 8;

		// DUT Instantiation
		logic  irdy;
		logic  ivld;
		logic [I_WIDTH-1:0]  idat;
		logic  ordy;
		logic  ovld;
		logic [O_WIDTH-1:0]  odat;
		requantf_axi #(
			.N(N), .C(C), .PE(PE),
			.SCALES(SCALES), .BIASES(BIASES),
			.SIGNED_OUT(SIGNED_OUT),
			.FORCE_BEHAVIORAL(FORCE_BEHAVIORAL)
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
				"[%0d] requantf_axi test: N=%0d C=%0d throttle=%0b/%0b signed_out=%0d",
				t, N, C, THROTTLED_IN, THROTTLED_OUT, SIGNED_OUT
			);

			@(posedge clk iff !rst);
			repeat(ROUNDS) begin
				automatic shortreal  xf;
				automatic int  exp;

				xf = ($urandom() % 100000) / 100.0 - 500.0;
				exp = $rtoi($floor(SCALES[0][CntCF] * xf + BIASES[0][CntCF] + 0.5));
				if(SIGNED_OUT) begin
					if(exp < -(2**(N-1)))  exp = -(2**(N-1));
					if(exp > 2**(N-1)-1)   exp = 2**(N-1)-1;
				end
				else begin
					if(exp < 0)         exp = 0;
					if(exp > 2**N - 1)  exp = 2**N - 1;
				end

				while(THROTTLED_IN && (($urandom()%7) == 0)) @(posedge clk);

				ivld <= 1;
				idat <= $shortrealtobits(xf);
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
				automatic int  exp;

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
				// Allow ±1 tolerance: FP32 FMA rounding may differ from shortreal reference
				if(SIGNED_OUT) begin
					automatic int  got = $signed(odat[N-1:0]);
					assert(got == exp || got == exp+1 || got == exp-1) else begin
						$error("[%0d] Output mismatch: %0d instead of %0d.", t, got, exp);
						$stop;
					end
				end
				else begin
					automatic int  got = odat[N-1:0];
					assert(got == exp || got == exp+1 || got == exp-1) else begin
						$error("[%0d] Output mismatch: %0d instead of %0d.", t, got, exp);
						$stop;
					end
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

`default_nettype wire
endmodule : requantf_axi_tb
