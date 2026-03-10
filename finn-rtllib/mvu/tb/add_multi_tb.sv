/******************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * @brief	Testbench for pipelined multi-input adder tree.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 *****************************************************************************/

module add_multi_tb;

	import  mvu_pkg::*;

	typedef struct {
		int unsigned  n;
		int unsigned  d;
		int  arg_lo;
		int  arg_hi;
	} test_cfg_t;

	localparam int unsigned  ROUNDS = 509;
	localparam int unsigned  TESTS = 6;
	localparam test_cfg_t  TEST_CFG[TESTS] = '{
		'{  7, 4, -1,  1},
		'{ 16, 2, -1,  1},
		'{ 33, 7, -1,  1},
		'{  5, 0,  0,  7},
		'{  8, 4,  0, 16},
		'{ 31, 5,  0, 33}
	};

	logic  clk = 0;
	always #5ns clk = !clk;
	logic  rst = 1;
	initial begin
		repeat(13) @(posedge clk);
		rst <= 0;
	end

	bit [TESTS-1:0]  done = '0;
	always_comb begin
		if(&done)  $finish();
	end

	for(genvar  test = 0; test < TESTS; test++) begin : genTests
		localparam test_cfg_t  CFG = TEST_CFG[test];
		localparam int unsigned  N = CFG.n;
		localparam int unsigned  D = CFG.d;
		localparam int  ARG_LO = CFG.arg_lo;
		localparam int  ARG_HI = CFG.arg_hi;

		localparam int unsigned  ARG_WIDTH = bitwidth(ARG_LO, ARG_HI);
		localparam int unsigned  SUM_WIDTH = sumwidth(N, ARG_WIDTH, ARG_LO, ARG_HI);

		// DUT
		logic [ARG_WIDTH-1:0]  arg[N];
		uwire [SUM_WIDTH-1:0]  sum;
		add_multi #(.N(N), .DEPTH(D), .ARG_WIDTH(ARG_WIDTH), .ARG_LO(ARG_LO), .ARG_HI(ARG_HI)) dut (
			.clk, .rst, .en('1),
			.arg, .sum
		);

		// Stimulus
		typedef struct {
			bit  vld;
			int  sum;
		} ref_t;
		ref_t  feed;
		initial begin
			arg  = '{ default: 'x };
			feed = '{ vld: 0, sum: 0 };
			@(posedge clk iff !rst);

			repeat(ROUNDS) begin
				automatic type(arg)  arg0;
				automatic int  sum0 = 0;
				foreach(arg0[i]) begin
					automatic int  val = ARG_LO + $urandom()%(ARG_HI-ARG_LO+1);
					arg0[i] = val;
					sum0 += val;
				end
				arg <= arg0;
				feed <= '{ vld: 1, sum: sum0 };
				@(posedge clk);
			end
			arg <= '{ default: 0 };
			feed <= '{ vld: 0, sum: 0 };

			repeat(7) @(posedge clk);
			done[test] <= 1;
		end

		// Checker
		int unsigned  Cnt = 0;
		initial begin
			automatic ref_t  Q[D+1];
			@(posedge clk iff !rst);
			forever @(posedge clk) begin
				for(int unsigned  i = 0; i < D; i++)  Q[i] = Q[i+1];
				Q[D] = feed;
				assert(!Q[0].vld || (sum == Q[0].sum[SUM_WIDTH-1:0])) else begin
					$error("Test %0d: Received %0d instead of %0d.", test, sum, Q[0].sum);
					$stop;
				end
				Cnt++;
			end
		end

		final begin
			$display("Test %0d: %0d successful checks.", test, Cnt);
		end

	end : genTests

endmodule : add_multi_tb
